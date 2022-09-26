#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import os
import random
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import tqdm
from gym import spaces
from torch import nn
from torch.utils.data._utils.collate import default_collate
from torch.optim.lr_scheduler import LambdaLR

from habitat import Config, VectorEnv, logger
from habitat.utils import profiling_wrapper
from habitat.utils.visualizations.utils import observations_to_image
from habitat_baselines.common.base_trainer import BaseRLTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat.core.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.transformer.transformer_rollout_storage import RolloutStorage
from habitat_baselines.common.tensorboard_utils import (
    TensorboardWriter,
    get_writer,
)
from habitat_baselines.rl.ddppo.algo import DDPPO
from habitat_baselines.rl.ddppo.ddp_utils import (
    EXIT,
    add_signal_handlers,
    get_distrib_size,
    init_distrib_slurm,
    is_slurm_batch_job,
    load_resume_state,
    rank0_only,
    requeue_job,
    save_resume_state,
)
from habitat_baselines.rl.ddppo.policy import (  # noqa: F401.
    PointNavResNetPolicy,
)
from habitat_baselines.transformer.transformer_policy import (
    TransformerResNetPolicy,
)
from habitat_baselines.utils.common import (
    ObservationBatchingCache,
    batch_obs,
    generate_video,
    get_num_actions,
    is_continuous_action_space,
)
from habitat_baselines.common.construct_vector_env import construct_envs
from habitat.utils.render_wrapper import overlay_frame


@baseline_registry.register_trainer(name="online_transformer")
class OnlineTransformerTrainer(BaseRLTrainer):
    r"""Trainer class for PPO algorithm
    Paper: https://arxiv.org/abs/1707.06347.
    """
    supported_tasks = ["Nav-v0"]

    SHORT_ROLLOUT_THRESHOLD: float = 0.25
    _is_distributed: bool
    _obs_batching_cache: ObservationBatchingCache
    envs: VectorEnv
    transformer_policy: TransformerResNetPolicy

    def __init__(self, config=None):
        super().__init__(config)
        self.transformer_policy = None
        self.envs = None
        self.obs_transforms = []

        self._static_encoder = False
        self._encoder = None
        self._obs_space = None

        # Distributed if the world size would be
        # greater than 1
        self._is_distributed = get_distrib_size()[2] > 1
        self._obs_batching_cache = ObservationBatchingCache()

        self.using_velocity_ctrl = (
            self.config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS
        ) == ["VELOCITY_CONTROL"]

    @property
    def obs_space(self):
        if self._obs_space is None and self.envs is not None:
            self._obs_space = self.envs.observation_spaces[0]

        return self._obs_space

    @obs_space.setter
    def obs_space(self, new_obs_space):
        self._obs_space = new_obs_space

    def _all_reduce(self, t: torch.Tensor) -> torch.Tensor:
        r"""All reduce helper method that moves things to the correct
        device and only runs if distributed
        """
        if not self._is_distributed:
            return t

        orig_device = t.device
        t = t.to(device=self.device)
        torch.distributed.all_reduce(t)

        return t.to(device=orig_device)
    
    def _setup_transformer_policy(self) -> None:
        r"""Sets up actor critic and agent for PPO.

        Args:
            ppo_cfg: config node with relevant params

        Returns:
            None
        """
        logger.add_filehandler(self.config.LOG_FILE)

        policy = baseline_registry.get_policy(self.config.RL.POLICY.name)
        observation_space = self.obs_space
        self.obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            observation_space, self.obs_transforms
        )

        self.transformer_policy = policy.from_config(
            self.config, observation_space, self.policy_action_space, 
        )
        self.obs_space = observation_space
        self.transformer_policy.to(self.device)

        if (
            self.config.RL.TRANSFORMER.pretrained_encoder
            or self.config.RL.TRANSFORMER.pretrained
        ):
            pretrained_state = torch.load(
                self.config.RL.TRANSFORMER.pretrained_weights, map_location="cpu"
            )

        if self.config.RL.TRANSFORMER.pretrained:
            prefix = ""
            self.transformer_policy.load_state_dict(
                {
                    k[k.find(prefix) + len(prefix) :]: v
                    for k, v in pretrained_state["state_dict"].items()
                }
            )
        elif self.config.RL.TRANSFORMER.pretrained_encoder:
            prefix = "net.visual_encoder."
            self.transformer_policy.net.visual_encoder.load_state_dict(
                {
                    k[k.find(prefix) + len(prefix) :]: v
                    for k, v in pretrained_state["state_dict"].items()
                    if prefix in k
                }
            )

        if not self.config.RL.TRANSFORMER.train_encoder:
            self._static_encoder = True
            for param in self.transformer_policy.net.visual_encoder.parameters():
                param.requires_grad_(False)


    def _init_envs(self, config=None):
        if config is None:
            config = self.config

        self.envs = construct_envs(
            config,
            workers_ignore_signals=is_slurm_batch_job(),
        )

    def _init_train(self):
        resume_state = load_resume_state(self.config)
        if resume_state is not None:
            self.config: Config = resume_state["config"]
            self.using_velocity_ctrl = (
                self.config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS
            ) == ["VELOCITY_CONTROL"]

        # if self.config.RL.DDPPO.force_distributed:
        #     self._is_distributed = True

        if is_slurm_batch_job():
            add_signal_handlers()

        if self._is_distributed:
            local_rank, tcp_store = init_distrib_slurm(
                self.config.RL.TRANSFORMER.distrib_backend
            )
            if rank0_only():
                logger.info(
                    "Initialized Skill Transformer with {} workers".format(
                        torch.distributed.get_world_size()
                    )
                )

            self.config.defrost()
            self.config.TORCH_GPU_ID = local_rank
            self.config.SIMULATOR_GPU_ID = local_rank
            # Multiply by the number of simulators to make sure they also get unique seeds
            self.config.TASK_CONFIG.SEED += (
                torch.distributed.get_rank() * self.config.NUM_ENVIRONMENTS
            )
            self.config.freeze()

            random.seed(self.config.TASK_CONFIG.SEED)
            np.random.seed(self.config.TASK_CONFIG.SEED)
            torch.manual_seed(self.config.TASK_CONFIG.SEED)
            self.num_rollouts_done_store = torch.distributed.PrefixStore(
                "rollout_tracker", tcp_store
            )
            self.num_rollouts_done_store.set("num_done", "0")

        if rank0_only() and self.config.VERBOSE:
            logger.info(f"config: {self.config}")

        
        if torch.cuda.is_available():
            self.device = torch.device("cuda", self.config.TORCH_GPU_ID)
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")
        
        if rank0_only() and not os.path.isdir(self.config.CHECKPOINT_FOLDER):
            os.makedirs(self.config.CHECKPOINT_FOLDER)
        
        profiling_wrapper.configure(
            capture_start_step=self.config.PROFILING.CAPTURE_START_STEP,
            num_steps_to_capture=self.config.PROFILING.NUM_STEPS_TO_CAPTURE,
        )

        self._init_envs(self.config)

        action_space = self.envs.action_spaces[0]
        if self.using_velocity_ctrl:
            # For navigation using a continuous action space for a task that
            # may be asking for discrete actions
            self.policy_action_space = action_space["VELOCITY_CONTROL"]
            action_shape = (2,)
            discrete_actions = False
        else:
            self.policy_action_space = action_space
            if is_continuous_action_space(action_space):
                # Assume ALL actions are NOT discrete
                action_shape = (get_num_actions(action_space),)
                discrete_actions = False
            else:
                # For discrete pointnav
                action_shape = None
                discrete_actions = True

        obs_space = self.obs_space
        storage_cfg = self.config.RL.ROLLOUT_STORAGE
        
        self._setup_transformer_policy()

        logger.info(
            "agent number of parameters: {}".format(
                sum(param.numel() for param in self.transformer_policy.parameters())
            )
        )

        if self._is_distributed:
            self.init_distributed(find_unused_params=True)
            torch.distributed.barrier()

        self._nbuffers = 2 if storage_cfg.use_double_buffered_sampler else 1

        self.rollouts = RolloutStorage(
            self.config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS,
            self.envs.num_envs,
            obs_space,
            self.policy_action_space,
            is_double_buffered=storage_cfg.use_double_buffered_sampler,
            action_shape=action_shape,
            discrete_actions=discrete_actions,
        )
        self.rollouts.to(self.device)

        observations = self.envs.reset()
        batch = batch_obs(
            observations, device=self.device, cache=self._obs_batching_cache
        )
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

        self.rollouts.buffers["observations"][0] = batch  # type: ignore

        self.current_episode_reward = torch.zeros((self.envs.num_envs, 1), device=self.device)
        self.current_timesteps = torch.zeros((self.envs.num_envs, 1), dtype=torch.int64, device=self.device)


        self.running_episode_stats = dict(
            count=torch.zeros(self.envs.num_envs, 1),
            reward=torch.zeros(self.envs.num_envs, 1),
        )
        self.window_episode_stats = defaultdict(
            lambda: deque(maxlen=self.config.RL.TRANSFORMER.reward_window_size)
        )
        self.num_episodes_collected = 0

        self.optimizer = torch.optim.Adam(
            list(filter(lambda p: p.requires_grad, self.transformer_policy.parameters())),
            lr=self.config.RL.TRANSFORMER.lr,
            eps=self.config.RL.TRANSFORMER.eps,
        )

        self.env_time = 0.0
        self.pth_time = 0.0
        self.t_start = time.time()

    def init_distributed(self, find_unused_params: bool = True) -> None:

        if torch.cuda.is_available():
            self.transformer_policy = torch.nn.parallel.DistributedDataParallel(
                self.transformer_policy,
                device_ids=[self.device],
                output_device=self.device,
                find_unused_parameters=find_unused_params,
            )
        else:
            self.transformer_policy = torch.nn.parallel.DistributedDataParallel(
                self.transformer_policy,
                find_unused_parameters=find_unused_params,
            )

    @rank0_only
    @profiling_wrapper.RangeContext("save_checkpoint")
    def save_checkpoint(
        self, file_name: str, extra_state: Optional[Dict] = None
    ) -> None:
        r"""Save checkpoint with specified name.

        Args:
            file_name: file name for checkpoint

        Returns:
            None
        """
        checkpoint = {
            "state_dict": self.transformer_policy.state_dict(),
            "config": self.config,
        }
        if extra_state is not None:
            checkpoint["extra_state"] = extra_state

        torch.save(
            checkpoint, os.path.join(self.config.CHECKPOINT_FOLDER, file_name)
        )

    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        r"""Load checkpoint of specified path as a dict.

        Args:
            checkpoint_path: path of target checkpoint
            *args: additional positional args
            **kwargs: additional keyword args

        Returns:
            dict containing checkpoint info
        """
        return torch.load(checkpoint_path, *args, **kwargs)

    METRICS_BLACKLIST = {"top_down_map", "collisions.is_collision"}

    @classmethod
    def _extract_scalars_from_info(
        cls, info: Dict[str, Any]
    ) -> Dict[str, float]:
        result = {}
        for k, v in info.items():
            if not isinstance(k, str) or k in cls.METRICS_BLACKLIST:
                continue

            if isinstance(v, dict):
                result.update(
                    {
                        k + "." + subk: subv
                        for subk, subv in cls._extract_scalars_from_info(
                            v
                        ).items()
                        if isinstance(subk, str)
                        and k + "." + subk not in cls.METRICS_BLACKLIST
                    }
                )
            # Things that are scalar-like will have an np.size of 1.
            # Strings also have an np.size of 1, so explicitly ban those
            elif np.size(v) == 1 and not isinstance(v, str):
                result[k] = float(v)

        return result

    @classmethod
    def _extract_scalars_from_infos(
        cls, infos: List[Dict[str, Any]]
    ) -> Dict[str, List[float]]:

        results = defaultdict(list)
        for i in range(len(infos)):
            for k, v in cls._extract_scalars_from_info(infos[i]).items():
                results[k].append(v)

        return results

    def _compute_actions_and_step_envs(self, buffer_index: int = 0):
        num_envs = self.envs.num_envs
        env_slice = slice(
            int(buffer_index * num_envs / self._nbuffers),
            int((buffer_index + 1) * num_envs / self._nbuffers),
        )

        t_sample_action = time.time()

        # sample actions
        with torch.no_grad():
            step_batch = self.rollouts.buffers[
                slice(
                    max(0, self.rollouts.current_rollout_step_idxs[buffer_index] - self.config.RL.TRANSFORMER.context_length + 1),
                    self.rollouts.current_rollout_step_idxs[buffer_index] + 1
                ),
                env_slice,
            ]

            profiling_wrapper.range_push("compute actions")
            
            current_timesteps = self.current_timesteps[env_slice]
            valid_context = torch.min(torch.ones_like(current_timesteps) * self.config.RL.TRANSFORMER.context_length, current_timesteps + 1)

            actions = self.transformer_policy.act(
                {k: step_batch["observations"][k].transpose(1,0) for k in step_batch["observations"].keys()},
                prev_actions=step_batch["actions"].transpose(1,0),
                targets=None,
                rtgs=self.config.RL.TRANSFORMER.return_to_go - step_batch["cumulative_rewards"].transpose(1,0), 
                timesteps=current_timesteps.unsqueeze(-1),
                valid_context=valid_context.squeeze(),
            )

        # NB: Move actions to CPU.  If CUDA tensors are
        # sent in to env.step(), that will create CUDA contexts
        # in the subprocesses.
        # For backwards compatibility, we also call .item() to convert to
        # an int
        actions = actions.to(device="cpu")
        self.pth_time += time.time() - t_sample_action

        profiling_wrapper.range_pop()  # compute actions

        t_step_env = time.time()

        for index_env, act in zip(
            range(env_slice.start, env_slice.stop), actions.unbind(0)
        ):
            if act.shape[0] > 1:
                step_action = action_array_to_dict(
                    self.policy_action_space, act
                )
            else:
                step_action = act.item()
            self.envs.async_step_at(index_env, step_action)

        self.env_time += time.time() - t_step_env

        self.rollouts.insert(
            actions=actions,
            timesteps=current_timesteps,
            buffer_index=buffer_index,
        )

    def _collect_environment_result(self, buffer_index: int = 0):
        num_envs = self.envs.num_envs
        env_slice = slice(
            int(buffer_index * num_envs / self._nbuffers),
            int((buffer_index + 1) * num_envs / self._nbuffers),
        )

        t_step_env = time.time()
        outputs = [
            self.envs.wait_step_at(index_env)
            for index_env in range(env_slice.start, env_slice.stop)
        ]

        observations, rewards_l, dones, infos = [
            list(x) for x in zip(*outputs)
        ]

        self.env_time += time.time() - t_step_env

        t_update_stats = time.time()
        batch = batch_obs(
            observations, device=self.device, cache=self._obs_batching_cache
        )
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

        rewards = torch.tensor(
            rewards_l,
            dtype=torch.float,
            device=self.current_episode_reward.device,
        )
        rewards = rewards.unsqueeze(1)

        self.num_episodes_collected += sum(dones)
        if self.num_episodes_collected == 1:
            a=0
        not_done_masks = torch.tensor(
            [[not done] for done in dones],
            dtype=torch.bool,
            device=self.current_episode_reward.device,
        )
        done_masks = torch.logical_not(not_done_masks)

        self.current_episode_reward[env_slice] += rewards
        current_rewards = self.current_episode_reward[env_slice]
        self.current_timesteps[env_slice] += 1
        current_ep_reward = self.current_episode_reward[env_slice]
        self.running_episode_stats["reward"][env_slice] += current_ep_reward.where(done_masks, torch.zeros_like(current_ep_reward)).cpu()  # type: ignore
        self.running_episode_stats["count"][env_slice] += done_masks.float().cpu()   # type: ignore
        for k, v_k in self._extract_scalars_from_infos(infos).items():
            v = torch.tensor(
                v_k,
                dtype=torch.float,
                device=self.current_episode_reward.device,
            ).unsqueeze(1)
            if k not in self.running_episode_stats:
                self.running_episode_stats[k] = torch.zeros_like(
                    self.running_episode_stats["count"]
                )

            self.running_episode_stats[k][env_slice] += v.where(done_masks, torch.zeros_like(v)).cpu()  # type: ignore

        if self._static_encoder:
            with torch.no_grad():
                batch["visual_features"] = self._encoder(batch)

        self.current_episode_reward[env_slice].masked_fill_(done_masks, 0.0)
        self.current_timesteps[env_slice].masked_fill_(done_masks, 0)
        
        self.rollouts.insert(
            next_observations=batch,
            rewards=rewards,
            cumulative_rewards=current_rewards,
            next_masks=not_done_masks,
            buffer_index=buffer_index,
        )

        self.rollouts.advance_rollout(buffer_index)

        self.pth_time += time.time() - t_update_stats

        return env_slice.stop - env_slice.start

    @profiling_wrapper.RangeContext("_collect_rollout_step")
    def _collect_rollout_step(self):
        self._compute_actions_and_step_envs()
        return self._collect_environment_result()

    @profiling_wrapper.RangeContext("_update_agent")
    def _update_agent(self):
        t_update_model = time.time()

        self.rollouts.compute_returns()

        self.transformer_policy.train()

        data_generator = self.rollouts.recurrent_generator(self.config.RL.TRANSFORMER.batch_size, self.config.RL.TRANSFORMER.context_length)

        losses = []
        for it, batch in enumerate(data_generator):
            batch = default_collate(batch)
            
            loss = self.transformer_policy(
                batch["observations"],
                batch["actions"],
                batch["actions"],
                batch["returns"],
                batch["timesteps"][:,-1:,:],
            )

            self.optimizer.zero_grad()
            torch.nn.utils.clip_grad_norm_(self.transformer_policy.parameters(), self.config.RL.TRANSFORMER.grad_norm_clip)
            self.optimizer.step()

            # report progress
            loss = loss.detach().mean() # collapse all losses if they are scattered on multiple gpus
            if rank0_only():
                print(f"epoch {self.num_updates_done+1} iter {it}: train loss {loss.item():.5f}.")
            losses.append(loss)
        losses = torch.mean(torch.stack(losses))

        self.rollouts.after_update()
        self.pth_time += time.time() - t_update_model

        profiling_wrapper.range_push("PPO.update epoch")
        
        return losses

    def _coalesce_post_step(
        self, losses: Dict[str, float], count_steps_delta: int
    ) -> Dict[str, float]:
        stats_ordering = sorted(self.running_episode_stats.keys())
        stats = torch.stack(
            [self.running_episode_stats[k] for k in stats_ordering], 0
        )

        stats = self._all_reduce(stats)

        for i, k in enumerate(stats_ordering):
            self.window_episode_stats[k].append(stats[i])

        if self._is_distributed:
            loss_name_ordering = sorted(losses.keys())
            stats = torch.tensor(
                [losses[k] for k in loss_name_ordering] + [count_steps_delta],
                device="cpu",
                dtype=torch.float32,
            )
            stats = self._all_reduce(stats)
            count_steps_delta = int(stats[-1].item())
            stats /= torch.distributed.get_world_size()

            losses = {
                k: stats[i].item() for i, k in enumerate(loss_name_ordering)
            }

        if self._is_distributed and rank0_only():
            self.num_rollouts_done_store.set("num_done", "0")

        self.num_steps_done += count_steps_delta

        return losses

    @rank0_only
    def _training_log(
        self, writer, losses: Dict[str, float], prev_time: int = 0
    ):
        deltas = {
            k: (
                (v[-1] - v[0]).sum().item()
                if len(v) > 1
                else v[0].sum().item()
            )
            for k, v in self.window_episode_stats.items()
        }
        deltas["count"] = max(deltas["count"], 1.0)

        writer.add_scalar(
            "reward",
            deltas["reward"] / deltas["count"],
            self.num_steps_done,
        )

        # Check to see if there are any metrics
        # that haven't been logged yet
        metrics = {
            k: v / deltas["count"]
            for k, v in deltas.items()
            if k not in {"reward", "count"}
        }

        for k, v in metrics.items():
            writer.add_scalar(f"metrics/{k}", v, self.num_steps_done)
        for k, v in losses.items():
            writer.add_scalar(f"losses/{k}", v, self.num_steps_done)

        fps = self.num_steps_done / ((time.time() - self.t_start) + prev_time)
        writer.add_scalar("metrics/fps", fps, self.num_steps_done)

        # log stats
        if self.num_updates_done % self.config.LOG_INTERVAL == 0:
            logger.info(
                "update: {}\tfps: {:.3f}\t".format(
                    self.num_updates_done,
                    fps,
                )
            )

            logger.info(
                "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\t"
                "frames: {}".format(
                    self.num_updates_done,
                    self.env_time,
                    self.pth_time,
                    self.num_steps_done,
                )
            )

            logger.info(
                "Average window size: {}  {}".format(
                    len(self.window_episode_stats["count"]),
                    "  ".join(
                        "{}: {:.3f}".format(k, v / deltas["count"])
                        for k, v in deltas.items()
                        if k != "count"
                    ),
                )
            )

    def should_end_early(self, rollout_step) -> bool:
        if not self._is_distributed:
            return False
        # This is where the preemption of workers happens.  If a
        # worker detects it will be a straggler, it preempts itself!
        return (
            rollout_step
            >= self.config.RL.PPO.num_steps * self.SHORT_ROLLOUT_THRESHOLD
        ) and int(self.num_rollouts_done_store.get("num_done")) >= (
            self.config.RL.DDPPO.sync_frac * torch.distributed.get_world_size()
        )

    @profiling_wrapper.RangeContext("train")
    def train(self) -> None:
        r"""Main method for training DD/PPO.

        Returns:
            None
        """

        self._init_train()

        count_checkpoints = 0
        prev_time = 0

        lr_scheduler = LambdaLR(
            optimizer=self.optimizer,
            lr_lambda=lambda x: 1 - self.percent_done(),
        )

        resume_state = load_resume_state(self.config)
        if resume_state is not None:
            prefix = ""
            self.transformer_policy.load_state_dict(
                {
                    k[k.find(prefix) + len(prefix) :]: v
                    for k, v in resume_state["state_dict"].items()
                    if prefix in k
                }
            )
            # self.transformer_policy.load_state_dict(resume_state["state_dict"])
            self.optimizer.load_state_dict(resume_state["optim_state"])
            lr_scheduler.load_state_dict(resume_state["lr_sched_state"])

            # requeue_stats = resume_state["requeue_stats"]
            # self.env_time = requeue_stats["env_time"]
            # self.pth_time = requeue_stats["pth_time"]
            # self.num_steps_done = requeue_stats["num_steps_done"]
            # self.num_updates_done = requeue_stats["num_updates_done"]
            # self._last_checkpoint_percent = requeue_stats[
            #     "_last_checkpoint_percent"
            # ]
            # count_checkpoints = requeue_stats["count_checkpoints"]
            # prev_time = requeue_stats["prev_time"]

            # self.running_episode_stats = requeue_stats["running_episode_stats"]
            # self.window_episode_stats.update(
            #     requeue_stats["window_episode_stats"]
            # )

        with (
            get_writer(self.config, flush_secs=self.flush_secs)
            if rank0_only()
            else contextlib.suppress(AttributeError)
        ) as writer:
            while not self.is_done():
                profiling_wrapper.on_start_step()
                profiling_wrapper.range_push("train update")

                if rank0_only() and self._should_save_resume_state():
                    # requeue_stats = dict(
                    #     env_time=self.env_time,
                    #     pth_time=self.pth_time,
                    #     count_checkpoints=count_checkpoints,
                    #     num_steps_done=self.num_steps_done,
                    #     num_updates_done=self.num_updates_done,
                    #     _last_checkpoint_percent=self._last_checkpoint_percent,
                    #     prev_time=(time.time() - self.t_start) + prev_time,
                    #     running_episode_stats=self.running_episode_stats,
                    #     window_episode_stats=dict(self.window_episode_stats),
                    # )
                    save_resume_state(
                        dict(
                            state_dict=self.transformer_policy.state_dict(),
                            optim_state=self.optimizer.state_dict(),
                            lr_sched_state=lr_scheduler.state_dict(),
                            config=self.config,
                            # requeue_stats=requeue_stats,
                        ),
                        self.config,
                    )

                if EXIT.is_set():
                    profiling_wrapper.range_pop()  # train update

                    self.envs.close()

                    requeue_job()

                    return

                self.transformer_policy.eval()
                count_steps_delta = 0
                profiling_wrapper.range_push("rollouts loop")

                profiling_wrapper.range_push("_collect_rollout_step")
                for buffer_index in range(self._nbuffers):
                    self._compute_actions_and_step_envs(buffer_index)

                not_done = True
                while not_done:

                    for buffer_index in range(self._nbuffers):
                        count_steps_delta += self._collect_environment_result(
                            buffer_index
                        )

                        if (buffer_index + 1) == self._nbuffers:
                            profiling_wrapper.range_pop()  # _collect_rollout_step

                        if (buffer_index + 1) == self._nbuffers:
                            profiling_wrapper.range_push(
                                "_collect_rollout_step"
                            )

                        if self.num_episodes_collected >= self.config.RL.ROLLOUT_STORAGE.num_trajs:
                            self.num_episodes_collected = 0
                            not_done = False
                            break

                        self._compute_actions_and_step_envs(buffer_index)

                profiling_wrapper.range_pop()  # rollouts loop

                if self._is_distributed:
                    self.num_rollouts_done_store.add("num_done", 1)

                loss = self._update_agent()

                if self.config.RL.TRANSFORMER.use_linear_lr_decay:
                    lr_scheduler.step()  # type: ignore

                self.num_updates_done += 1
                losses = self._coalesce_post_step(
                    dict(
                        loss=loss,
                    ),
                    count_steps_delta,
                )

                self._training_log(writer, losses, prev_time)

                # checkpoint model
                if rank0_only() and self.should_checkpoint():
                    self.save_checkpoint(
                        f"ckpt.{count_checkpoints}.pth",
                        dict(
                            step=self.num_steps_done,
                            wall_time=(time.time() - self.t_start) + prev_time,
                        ),
                    )
                    count_checkpoints += 1

                profiling_wrapper.range_pop()  # train update

            self.envs.close()

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        pass