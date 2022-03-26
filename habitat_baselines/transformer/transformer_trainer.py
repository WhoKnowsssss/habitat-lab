#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import os
import random
import time
from collections import defaultdict, deque
from typing import Any, ClassVar, Dict, List, Tuple, Union, Optional


import numpy as np
import torch
from tqdm import tqdm
from gym import spaces
from torch import device, nn
from torch.optim.lr_scheduler import LambdaLR
from warmup_scheduler import GradualWarmupScheduler
from torch.utils.data import (
    DataLoader,
    DistributedSampler
)
import torch.multiprocessing as mp

from habitat import Config, VectorEnv, logger
from habitat.utils import profiling_wrapper
from habitat.utils.visualizations.utils import observations_to_image
from habitat_baselines.common.base_trainer import BaseRLTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.common.rollout_storage import RolloutStorage
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
from habitat_baselines.transformer.policy import (
    TransformerResNetPolicy,
)
from habitat_baselines.transformer.dataset import RollingDataset
from habitat_baselines.utils.common import (
    ObservationBatchingCache,
    action_array_to_dict,
    batch_obs,
    generate_video,
    get_num_actions,
    is_continuous_action_space,
)
from habitat_baselines.utils.env_utils import construct_envs


@baseline_registry.register_trainer(name="transformer")
class TransformerTrainer(BaseRLTrainer):
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
            get_env_class(config.ENV_NAME),
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

        world_rank, world_size = None, None
        if self._is_distributed:
            local_rank, tcp_store = init_distrib_slurm(
                self.config.RL.TRANSFORMER.distrib_backend
            )
            world_rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
            if rank0_only():
                logger.info(
                    "Initialized Skill Transformer with {} workers, id={}".format(
                        world_size, world_rank
                    )
                )

            self.config.defrost()
            self.config.TORCH_GPU_ID = local_rank
            self.config.SIMULATOR_GPU_ID = local_rank
            # Multiply by the number of simulators to make sure they also get unique seeds
            # self.config.TASK_CONFIG.SEED += (
            #     world_rank * self.config.NUM_ENVIRONMENTS
            # )
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

        self._init_envs(self.config)

        action_space = self.envs.action_spaces[0]
        if self.using_velocity_ctrl:
            # For navigation using a continuous action space for a task that
            # may be asking for discrete actions
            self.policy_action_space = action_space["VELOCITY_CONTROL"]
        else:
            self.policy_action_space = action_space

        self._setup_transformer_policy()

        logger.info(
            "agent number of parameters: {}".format(
                sum(param.numel() for param in self.transformer_policy.parameters())
            )
        )

        if self._is_distributed:
            self.init_distributed(find_unused_params=True)
            torch.distributed.barrier()

        # self.train_dataset = StateActionReturnDataset.from_config(self.config.RL.TRAJECTORY_DATASET, self.config.RL.TRANSFORMER.context_length*3)
        manager = mp.Manager()
        self.dataset_context = manager.dict()
        self.train_dataset = RollingDataset(self.config.RL.TRAJECTORY_DATASET, 
                                self.config.RL.TRANSFORMER.context_length*3, 
                                (world_size, world_rank, self.config.TASK_CONFIG.SEED), 
                                self.dataset_context,
                                rank0_only()
                            )
            
        self.train_loader = DataLoader(self.train_dataset, pin_memory=True,
                                batch_size=self.config.RL.TRANSFORMER.batch_size,
                                num_workers=self.config.RL.TRANSFORMER.num_workers, 
                                persistent_workers=True
                            )

        self.optimizer = torch.optim.Adam(
            list(filter(lambda p: p.requires_grad, self.transformer_policy.parameters())),
            lr=self.config.RL.TRANSFORMER.lr,
            eps=self.config.RL.TRANSFORMER.eps,
        )

        self.envs.close()



        self.env_time = 0.0
        self.pth_time = 0.0
        self.t_start = time.time()

    # def init_distributed(self, find_unused_params: bool = True) -> None:
    #     r"""Initializes distributed training for the model

    #     1. Broadcasts the model weights from world_rank 0 to all other workers
    #     2. Adds gradient hooks to the model

    #     :param find_unused_params: Whether or not to filter out unused parameters
    #                                before gradient reduction.  This *must* be True if
    #                                there are any parameters in the model that where unused in the
    #                                forward pass, otherwise the gradient reduction
    #                                will not work correctly.
    #     """
    #     class _EvalActionsWrapper(torch.nn.Module):
    #         r"""Wrapper on evaluate_actions that allows that to be called from forward.
    #         This is needed to interface with DistributedDataParallel's forward call
    #         """

    #         def __init__(self, transformer_policy):
    #             super().__init__()
    #             self.transformer_policy = transformer_policy

    #         def forward(self, *args, **kwargs):
    #             return self.transformer_policy.evaluate_actions(*args, **kwargs)
        
    #     # NB: Used to hide the hooks from the nn.Module,
    #     # so they don't show up in the state_dict
    #     class Guard:  # noqa: SIM119
    #         def __init__(self, model, device):
    #             if torch.cuda.is_available():
    #                 self.ddp = torch.nn.parallel.DistributedDataParallel(
    #                     model,
    #                     device_ids=[device],
    #                     output_device=device,
    #                     find_unused_parameters=find_unused_params,
    #                 )
    #             else:
    #                 self.ddp = torch.nn.parallel.DistributedDataParallel(
    #                     model,
    #                     find_unused_parameters=find_unused_params,
    #                 )

    #         def __call__(self, observations, rnn_hidden_states, prev_actions, masks, action):
    #             return self.ddp(observations, rnn_hidden_states, prev_actions, masks, action)

    #     self._evaluate_actions_wrapper = Guard(_EvalActionsWrapper(self.transformer_policy), self.device)  # type: ignore

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
                self.rollouts.current_rollout_step_idxs[buffer_index],
                env_slice,
            ]

            profiling_wrapper.range_push("compute actions")
            (
                values,
                actions,
                actions_log_probs,
                recurrent_hidden_states,
            ) = self.actor_critic.act(
                step_batch["observations"],
                step_batch["recurrent_hidden_states"],
                step_batch["prev_actions"],
                step_batch["masks"],
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
            next_recurrent_hidden_states=recurrent_hidden_states,
            actions=actions,
            action_log_probs=actions_log_probs,
            value_preds=values,
            buffer_index=buffer_index,
        )

    @rank0_only
    def _training_log(
        self, writer, losses: Dict[str, float], prev_time: int = 0
    ):
        # deltas = {
        #     k: (
        #         (v[-1] - v[0]).sum().item()
        #         if len(v) > 1
        #         else v[0].sum().item()
        #     )
        #     for k, v in self.window_episode_stats.items()
        # }
        # deltas["count"] = max(deltas["count"], 1.0)

        # writer.add_scalar(
        #     "reward",
        #     deltas["reward"] / deltas["count"],
        #     self.num_updates_done,
        # )

        # # Check to see if there are any metrics
        # # that haven't been logged yet
        # metrics = {
        #     k: v / deltas["count"]
        #     for k, v in deltas.items()
        #     if k not in {"reward", "count"}
        # }
        # if len(metrics) > 0:
        #     writer.add_scalars("metrics", metrics, self.num_updates_done)

        writer.add_scalars(
            "losses",
            losses,
            self.num_updates_done,
        )

        # log stats
        # if self.num_updates_done % self.config.LOG_INTERVAL == 0:
        #     logger.info(
        #         "update: {}\tfps: {:.3f}\t".format(
        #             self.num_updates_done,
        #             self.num_updates_done
        #             / ((time.time() - self.t_start) + prev_time),
        #         )
        #     )

        #     logger.info(
        #         "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\t"
        #         "frames: {}".format(
        #             self.num_updates_done,
        #             self.env_time,
        #             self.pth_time,
        #             self.num_updates_done,
        #         )
        #     )

        #     logger.info(
        #         "Average window size: {}  {}".format(
        #             len(self.window_episode_stats["count"]),
        #             "  ".join(
        #                 "{}: {:.3f}".format(k, v / deltas["count"])
        #                 for k, v in deltas.items()
        #                 if k != "count"
        #             ),
        #         )
        #     )

    def _run_epoch(
        self, 
        split: str, 
        epoch_num: int=0
    ):
        is_train = split == 'train'
        self.transformer_policy.train(is_train)

        pbar = tqdm(enumerate(self.train_loader)) # #  # if is_train else enumerate(self.train_loader)
        losses = []
        for it, (x, y, r, t) in pbar:

            # place data on the correct device

            x = [{idx2: x[idx1][idx2].to(self.device) for idx2 in x[idx1].keys()} for idx1 in range(len(x))]

            y = y.to(self.device).squeeze(-2)
            r = r.to(self.device).squeeze(-2)
            t = t.to(self.device)

            # forward the model
            with torch.set_grad_enabled(is_train):
                # logits, loss = model(x, y, r)
                loss = self.transformer_policy(x, y, y, r, t)
                loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                losses.append(loss)

            if is_train:

                # backprop and update the parameters
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.transformer_policy.parameters(), self.config.RL.TRANSFORMER.grad_norm_clip)
                self.optimizer.step()

                # report progress
                if rank0_only():
                    pbar.set_description(f"epoch {epoch_num+1} iter {it}: train loss {loss.item():.5f}.")

        losses = torch.mean(torch.stack(losses))
        print("LR:::", self.optimizer.param_groups[0]['lr'])
        return losses

    @profiling_wrapper.RangeContext("train")
    def train(self) -> None:
        r"""Main method for training DD/PPO.

        Returns:
            None
        """

        self._init_train()

        count_checkpoints = 0
        prev_time = 0
        
        lr_scheduler_after = LambdaLR(
            optimizer=self.optimizer,
            lr_lambda=lambda x: 1 - self.percent_done(),
        )
        lr_scheduler = GradualWarmupScheduler(
            self.optimizer, 
            multiplier=1, 
            total_epoch=50, 
            after_scheduler=lr_scheduler_after
        )

        resume_state = load_resume_state(self.config)
        if resume_state is not None:
            self.transformer_policy.load_state_dict(resume_state["state_dict"])
            self.optimizer.load_state_dict(resume_state["optim_state"])
            lr_scheduler_after.load_state_dict(resume_state["lr_sched_state"])
            lr_scheduler.total_epoch = 0

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

                if self.config.RL.TRANSFORMER.use_linear_lr_decay:
                    self.optimizer.step()
                    lr_scheduler.step()  # type: ignore

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
                            lr_sched_state=lr_scheduler_after.state_dict(),
                            config=self.config,
                            # requeue_stats=requeue_stats,
                        ),
                        self.config,
                    )

                if EXIT.is_set():

                    requeue_job()

                    return

                self.train_dataset.set_epoch(self.num_updates_done)
      
                loss = self._run_epoch('train', epoch_num=self.num_updates_done)

                loss = self._all_reduce(loss)



                self.num_updates_done += 1

                self._training_log(writer, {'loss': loss}, prev_time)

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

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        if self._is_distributed:
            raise RuntimeError("Evaluation does not support distributed mode")

        # self.device = torch.device("cpu")

        # Map location CPU is almost always better than mapping to a CUDA device.
        ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")

        if self.config.EVAL.USE_CKPT_CONFIG:
            config = self._setup_eval_config(ckpt_dict["config"])
        else:
            config = self.config.clone()

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        config.freeze()

        if (
            len(self.config.VIDEO_OPTION) > 0
            and self.config.VIDEO_RENDER_TOP_DOWN
        ):
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
            config.freeze()

        if config.VERBOSE:
            logger.info(f"env config: {config}")

        self._init_envs(config)

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
                # Assume NONE of the actions are discrete
                action_shape = (get_num_actions(action_space),)
                discrete_actions = False
            else:
                # For discrete pointnav
                action_shape = (1,)
                discrete_actions = True

        self._setup_transformer_policy()

        # self.transformer_policy.load_state_dict(ckpt_dict["state_dict"])
        
        prefix= ''
        if any(["module." in k for k in ckpt_dict["state_dict"].keys()]):
            prefix = "module."
        
        self.transformer_policy.load_state_dict(
            {
                k[k.find(prefix) + len(prefix) :]: v
                for k, v in ckpt_dict["state_dict"].items()
                if prefix in k and ("action_distri" not in k) and ("critic" not in k)
            }
        )

        observations = self.envs.reset()
        batch = batch_obs(
            observations, device=self.device, cache=self._obs_batching_cache
        )
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        current_episode_reward = torch.zeros(
            self.envs.num_envs, 1, device="cpu"
        )

        prev_actions = torch.zeros(
            self.config.NUM_ENVIRONMENTS, self.config.RL.TRANSFORMER.context_length,
            *action_shape,
            device=self.device,
            dtype=torch.long if discrete_actions else torch.float,
        )
        rtgs = torch.zeros(
            (self.config.NUM_ENVIRONMENTS, self.config.RL.TRANSFORMER.context_length, 1), 
            dtype=torch.int64, 
            device=self.device,
        )
        rtgs[:,-1,0] = self.config.RL.TRANSFORMER.return_to_go
        timesteps = torch.zeros(
            (self.config.NUM_ENVIRONMENTS, 1, 1), 
            dtype=torch.int64, 
            device=self.device,
        )
        batch_list = [batch]

        valid_context = torch.ones(
            (self.config.NUM_ENVIRONMENTS), 
            dtype=torch.int64, 
            device=self.device,
        )

        not_done_masks = torch.zeros(
            self.config.NUM_ENVIRONMENTS,
            1,
            device=self.device,
            dtype=torch.bool,
        )

        stats_episodes: Dict[
            Any, Any
        ] = {}  # dict of dicts that stores stats per episode

        rgb_frames = [
            [] for _ in range(self.config.NUM_ENVIRONMENTS)
        ]  # type: List[List[np.ndarray]]
        if len(self.config.VIDEO_OPTION) > 0:
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)

        number_of_eval_episodes = self.config.TEST_EPISODE_COUNT
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = sum(self.envs.number_of_episodes)
        else:
            total_num_eps = sum(self.envs.number_of_episodes)
            if total_num_eps < number_of_eval_episodes:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes"
                    ", dataset only has {total_num_eps}."
                )
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_eval_episodes = total_num_eps

        pbar = tqdm(total=number_of_eval_episodes)
        self.transformer_policy.eval()

        name_list = list(range(10,110,10))
        name_list.reverse()
        name__ = name_list.pop()
        gt = torch.load('data/temp_data/{}.pt'.format(name__), map_location=torch.device('cpu'))
        self.gt_actions = gt["actions"]
        self.gt_observations = gt["obs"]
        obs_fix = []
        idxxx=0

            
        while (
            len(stats_episodes) < number_of_eval_episodes
            and self.envs.num_envs > 0
        ):
            current_episodes = self.envs.current_episodes()


            batch_list2 = self.gt_observations[max(0,idxxx+1-30):idxxx+1]
            for k in batch_list2[0].keys():
                batch_list2[-1][k] = batch_list2[-1][k].to(prev_actions.device).unsqueeze(0)
                print(torch.sum(torch.abs(batch_list2[-1][k] - batch_list[-1][k])))
                # batch_list[i]['robot_head_depth'] = batch_list[i]['robot_head_depth']
            # batch_list = batch_list2
            with torch.no_grad(): 
                actions = self.transformer_policy.act(
                    batch_list,
                    prev_actions=prev_actions,
                    targets=None, 
                    rtgs=rtgs, 
                    timesteps=timesteps,
                    valid_context=valid_context,
                )
            print(actions)
            if idxxx < 0:
                pass
                actions = torch.tensor(self.gt_actions[idxxx], device=prev_actions.device).unsqueeze(0)
            else:
                pass
                # gt["obs_fix"] = obs_fix
                # torch.save(gt, 'data/temp_data/{}_fix.pt'.format(name__))
                # name__ = name_list.pop()
                # gt = torch.load('data/temp_data/{}.pt'.format(name__), map_location=torch.device('cpu'))
                # self.gt_actions = gt["actions"]
                # self.gt_observations = gt["obs"]
                # idxxx = 0
                # obs_fix = []
                # continue
            differnce = actions - torch.tensor(self.gt_actions[idxxx], device=prev_actions.device).unsqueeze(0)
            print(differnce, self.gt_actions[idxxx])
            gt_observations = self.gt_observations[idxxx]
            # print(gt_observations['robot_head_depth'] - observations[0]['robot_head_depth'])
            idxxx += 1
            # if timesteps[0,0,0].item() == 49:
            #     print()
            prev_actions = torch.cat((prev_actions[:,1:,:], actions.unsqueeze(1)), dim=1)  # type: ignore
            # NB: Move actions to CPU.  If CUDA tensors are
            # sent in to env.step(), that will create CUDA contexts
            # in the subprocesses.
            # For backwards compatibility, we also call .item() to convert to
            # an int
            if actions[0].shape[0] > 1:
                step_data = [
                    action_array_to_dict(self.policy_action_space, a)
                    for a in actions.to(device="cpu")
                ]
            else:
                step_data = [a.item() for a in actions.to(device="cpu")]

            outputs = self.envs.step(step_data)

            observations, rewards_l, dones, infos = [
                list(x) for x in zip(*outputs)
            ]

            obs_fix.append(observations[0])

            batch = batch_obs(
                observations,
                device=self.device,
                cache=self._obs_batching_cache,
            )
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)

            batch_list = batch_list + [batch]
            batch_list = batch_list[-self.config.RL.TRANSFORMER.context_length:]

            not_done_masks = torch.tensor(
                [[not done] for done in dones],
                dtype=torch.bool,
                device="cpu",
            )

            rewards = torch.tensor(
                rewards_l, dtype=torch.float, device=self.device
            )
            rtgs = torch.cat((rtgs[:,1:,:], (rtgs[:,-1,0]-rewards).reshape(-1,1,1)), dim=1)

            valid_context = torch.minimum(valid_context + 1, torch.tensor(self.config.RL.TRANSFORMER.context_length, device=self.device))


            rewards = rewards.cpu().unsqueeze(-1)
            timesteps = timesteps + 1
            current_episode_reward += rewards
            next_episodes = self.envs.current_episodes()
            envs_to_pause = []
            n_envs = self.envs.num_envs
            for i in range(n_envs):
                if (
                    next_episodes[i].scene_id,
                    next_episodes[i].episode_id,
                ) in stats_episodes:
                    envs_to_pause.append(i)

                # episode ended
                if not not_done_masks[i].item():
                    pbar.update()
                    episode_stats = {}
                    episode_stats["reward"] = current_episode_reward[i].item()
                    episode_stats.update(
                        self._extract_scalars_from_info(infos[i])
                    )
                    current_episode_reward[i] = 0
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[
                        (
                            current_episodes[i].scene_id,
                            current_episodes[i].episode_id,
                        )
                    ] = episode_stats

                    valid_context[i] = 1
                    timesteps[i,0,0] = 0
                    rtgs[i,-1,0] = self.config.RL.TRANSFORMER.return_to_go

                    
                    if len(self.config.VIDEO_OPTION) > 0:
                        generate_video(
                            video_option=self.config.VIDEO_OPTION,
                            video_dir=self.config.VIDEO_DIR,
                            images=rgb_frames[i],
                            episode_id=current_episodes[i].episode_id,
                            checkpoint_idx=checkpoint_index,
                            metrics=self._extract_scalars_from_info(infos[i]),
                            tb_writer=writer,
                        )

                        rgb_frames[i] = []


                    num_episodes = len(stats_episodes)
                    aggregated_stats = {}
                    for stat_key in next(iter(stats_episodes.values())).keys():
                        aggregated_stats[stat_key] = (
                            sum(v[stat_key] for v in stats_episodes.values())
                            / num_episodes
                        )

                    for k, v in aggregated_stats.items():
                        logger.info(f"Average episode {k}: {v:.4f}")

                    

                # episode continues
                elif len(self.config.VIDEO_OPTION) > 0:
                    # TODO move normalization / channel changing out of the policy and undo it here
                    frame = observations_to_image(
                        {k: v[i] for k, v in batch.items()}, infos[i]
                    )
                    rgb_frames[i].append(frame)

            not_done_masks = not_done_masks.to(device=self.device)
            (
                self.envs,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                rtgs,
                batch_list,
                valid_context,
                timesteps,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                rtgs,
                batch_list,
                valid_context,
                timesteps,
                rgb_frames,
            )

        

        
        
        num_episodes = len(stats_episodes)
        aggregated_stats = {}
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = (
                sum(v[stat_key] for v in stats_episodes.values())
                / num_episodes
            )

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        step_id = checkpoint_index
        if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
            step_id = ckpt_dict["extra_state"]["step"]

        writer.add_scalars(
            "eval_reward",
            {"average reward": aggregated_stats["reward"]},
            step_id,
        )

        rewards = np.array([v['reward'] for v in stats_episodes.values()])
        rewards.tofile('rewards_{}.csv'.format(self.config.RL.TRANSFORMER.return_to_go),sep=',')

        

        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        if len(metrics) > 0:
            writer.add_scalars("eval_metrics", metrics, step_id)

        self.envs.close()
    
    
    @staticmethod
    def _pause_envs(
        envs_to_pause: List[int],
        envs,
        not_done_masks,
        current_episode_reward,
        prev_actions,
        rtgs,
        batch_list,
        valid_context,
        timesteps,
        rgb_frames,
    ):
        # pausing self.envs with no new episode
        if len(envs_to_pause) > 0:
            state_index = list(range(envs.num_envs))
            for idx in reversed(envs_to_pause):
                state_index.pop(idx)
                envs.pause_at(idx)

            # indexing along the batch dimensions
            not_done_masks = not_done_masks[state_index]
            current_episode_reward = current_episode_reward[state_index]
            prev_actions = prev_actions[state_index]
            rtgs = rtgs[state_index]
            valid_context = valid_context[state_index]
            timesteps = timesteps[state_index]

            for i, batch in enumerate(batch_list):
                for k, v in batch.items():
                    batch_list[i][k] = v[state_index]

            rgb_frames = [rgb_frames[i] for i in state_index]

        return (
            envs,
            not_done_masks,
            current_episode_reward,
            prev_actions,
            rtgs,
            batch_list,
            valid_context,
            timesteps,
            rgb_frames,
        )
