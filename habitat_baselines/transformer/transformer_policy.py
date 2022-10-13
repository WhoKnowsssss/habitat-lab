#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc
import time
from typing import Dict, List, Optional, Tuple

import torch
from gym import spaces
from torch import device, nn as nn
import torch.nn.functional as F
import numpy as np

from habitat.config import Config
from habitat.tasks.nav.nav import (
    EpisodicCompassSensor,
    EpisodicGPSSensor,
    HeadingSensor,
    ImageGoalSensor,
    IntegratedPointGoalGPSAndCompassSensor,
    PointGoalSensor,
    ProximitySensor,
)
from habitat_baselines.rl.ddppo.policy.resnet_policy import ResNetEncoder
from habitat_baselines.rl.ddppo.policy import resnet
from habitat_baselines.rl.ppo.policy import Policy
from habitat_baselines.transformer.transformer_model import (
    GPTConfig,
    GPT,
)
from habitat_baselines.transformer.pure_bc_model import LSTMBC
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.utils.common import get_num_actions
from habitat_baselines.rl.transformer_policy.action_distribution import (
    ActionDistribution,
)
from habitat_baselines.rl.ppo import NetPolicy
from habitat_baselines.common.tensor_dict import TensorDict

from .focal_loss import FocalLoss

@baseline_registry.register_policy
class TransformerResNetPolicy(NetPolicy):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int = 512,
        context_length: int = 30,
        max_episode_step: int = 200,
        n_layer: int = 6,
        n_head: int = 8,
        model_type: str = "reward_conditioned",
        resnet_baseplanes: int = 32,
        backbone: str = "resnet18",
        force_blind_policy: bool = False,
        policy_config: Config = None,
        fuse_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        include_visual_keys = policy_config.include_visual_keys
        super().__init__(
            TransformerResnetNet(
                observation_space=observation_space,
                action_space=action_space,  # for previous action
                hidden_size=hidden_size,
                context_length=context_length,
                max_episode_step=max_episode_step,
                model_type=model_type,
                n_head=n_head,
                n_layer=n_layer,
                backbone=backbone,
                resnet_baseplanes=resnet_baseplanes,
                force_blind_policy=force_blind_policy,
                discrete_actions=False,
                fuse_keys=[
                    k for k in fuse_keys if k not in include_visual_keys
                ],
                include_visual_keys=include_visual_keys,
            ),
            action_space=action_space,
            policy_config=policy_config,
        )
        self.boundaries_mean = torch.tensor(
            [-1.0, -0.8, -0.6, -0.4, -0.2, 0., 0.2, 0.4, 0.6, 0.8, 1.0]
        ).cuda()
        self.boundaries = torch.tensor(
            [-1.1, -0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9, 1.1]
        ).cuda()
        self.loss_vars = nn.parameter.Parameter(torch.zeros((3,)))
        self.focal_loss = FocalLoss(
            alpha=(1-torch.tensor([0.05,0.0125,0.0125,0.0125,0.0125,0.8,0.0125,0.0125,0.0125,0.0125,0.05])), gamma=5).cuda()
        self.focal_loss_loc = FocalLoss(gamma=5).cuda()
        self.focal_loss_pick = FocalLoss(
            alpha=(1-torch.tensor([0.8,0.1,0.1])), gamma=5).cuda()

        if self.action_distribution_type == "categorical":
            self.len_logit = [11*7, 3, 11*2]
        elif self.action_distribution_type == "gaussian":
            self.len_logit = [7, 1, 2]
        elif self.action_distribution_type == "mixed":
            self.len_logit = [11*7, 3, 2]
        else:
            raise NotImplementedError

    @classmethod
    def from_config(
        cls,
        config: Config,
        observation_space: spaces.Dict,
        action_space,
        **kwargs,
    ):
        return cls(
            observation_space=observation_space,
            action_space=action_space,
            hidden_size=config.RL.TRANSFORMER.hidden_size,
            context_length=config.RL.TRANSFORMER.context_length,
            max_episode_step=config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS,
            model_type=config.RL.TRANSFORMER.model_type,
            n_head=config.RL.TRANSFORMER.n_head,
            n_layer=config.RL.TRANSFORMER.n_layer,
            backbone=config.RL.TRANSFORMER.backbone,
            force_blind_policy=config.FORCE_BLIND_POLICY,
            policy_config=config.RL.POLICY,
            fuse_keys=config.TASK_CONFIG.GYM.OBS_KEYS,
        )

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        (
            value,
            action,
            action_log_probs,
            rnn_hidden_states,
        ) = super().act(
            observations,
            rnn_hidden_states,
            prev_actions,
            masks,
            deterministic=deterministic,
        )
        if self.action_distribution_type == "mixed":
            action[:,:7] = self.boundaries_mean[action[:,:7].to(torch.long)]
            action[:,7] = (action[:,7] == 1).int() \
                            + 2*(action[:,7] == 0).int() \
                            + 3*(action[:,7] == 2).int() - 2
        return (
            value,
            action,
            action_log_probs,
            rnn_hidden_states,
        )

    def evaluate_actions(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        action,
        rnn_build_seq_info=None,
        evaluate_aux_losses=True,
    ):
        if self.action_distribution_type == "mixed":
            action[:,:7] = torch.bucketize(action[:,:7], self.boundaries) - 1
            action[:,7] = (action[:,7] == 0).int() \
                            + 2*(sampled_action[:,7] == -1).int() \
                            + 3*(sampled_action[:,7] == 1).int() - 1
        return super().evaluate_actions(
            observations,
            rnn_hidden_states,
            prev_actions,
            masks,
            action,
            rnn_build_seq_info=rnn_build_seq_info,
            evaluate_aux_losses=evaluate_aux_losses,
        )

    def forward(
        self,
        states, 
        actions, 
        targets, 
        rtgs,
        timesteps,
    ):
        t1 = time.perf_counter()
        features = self.net(states, None, actions, None, rtgs=rtgs, offline_training=True)
        # if we are given some desired targets also calculate the loss
        loss = None
        loss_dict = None

        if self.action_distribution_type == "categorical":
            distribution = self.action_distribution(features)
            logits = distribution.probs
        elif self.action_distribution_type == "gaussian":
            distribution = self.action_distribution(features)
            logits = distribution.mean
        elif self.action_distribution_type == "mixed":
            logits = self.action_distribution(features, return_logits=True)
        else:
            raise NotImplementedError

        #======================== separate logits ==========================
        logits_arm, logits_pick, logits_loc = torch.split(logits, self.len_logit, -1)
        
        #=========================== locomotion ============================
        temp_target = targets[:,:,8:10]
        loss1 = F.mse_loss(logits_loc, temp_target)

        #=========================== arm action ============================
        temp_target = torch.bucketize(targets[:,:,:7], self.boundaries) - 1
        logits_arm = logits_arm.view(*logits_arm.shape[:2], 7, 11)
        loss2 = self.focal_loss(logits_arm[:,:,:,:].permute(0,3,1,2), temp_target[:,:,:7])
        accuracy2 = torch.sum(torch.argmax(logits_arm[:,:,:,:], dim=-1) == temp_target[:,:,:7]) / np.prod(temp_target[:,:,:7].shape)

        #========================= gripper action ==========================
        loss3 = self.focal_loss_pick(logits_pick.permute(0,2,1), targets[:,:,7].long())
        accuracy3 = torch.sum(torch.argmax(logits_pick[:,:,:], dim=-1) == targets[:,:,7].long()) / np.prod(targets[:,:,7].shape)

        #========================== stop action ============================
        # loss4 = F.cross_entropy(logits_stop.permute(0,2,1), targets[:,:,10].long(), label_smoothing=0.05)
        # accuracy4 = torch.sum(torch.argmax(logits_stop[:,:,:], dim=-1) == targets[:,:,10].long()) / np.prod(targets[:,:,10].shape)

        loss_dict = {
            "locomotion": loss1.detach().item(), 
            "arm": loss2.detach().item(), 
            "pick": loss3.detach().item(), 
            # "place": loss4.detach().item(),
            # "accuracy_nav": accuracy1.detach().item(),
            "accuracy_pick": accuracy3.detach().item(),
            "accuracy_arm": accuracy2.detach().item(),
            # "accuracy_place": accuracy4.detach().item(),
            } 
        loss1 = torch.exp(-self.loss_vars[0]) * loss1 + self.loss_vars[0]
        loss2 = torch.exp(-self.loss_vars[1]) * loss2 + self.loss_vars[1]
        loss3 = torch.exp(-self.loss_vars[2]) * loss3 + self.loss_vars[2]
        # loss4 = torch.exp(-self.loss_vars[2]) * loss4 + self.loss_vars[2]
        loss = loss1 + loss2 + loss3 #+ loss4
        return loss, loss_dict

class TransformerResnetNet(nn.Module):
    """Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int,
        context_length: int,
        max_episode_step: int,
        model_type: str,
        n_head: int,
        n_layer: int,
        backbone,
        resnet_baseplanes,
        force_blind_policy: bool = False,
        discrete_actions: bool = True,
        fuse_keys: Optional[List[str]] = None,
        include_visual_keys: Optional[List[str]] = None,
    ):
        super().__init__()
        self.context_length = context_length

        self.discrete_actions = discrete_actions
        if discrete_actions:
            num_actions = action_space.n + 1
        else:
            num_actions = get_num_actions(action_space)

        # self._n_prev_action = 32
        # rnn_input_size = self._n_prev_action
        rnn_input_size = 0
        self.include_visual_keys = include_visual_keys

        self._fuse_keys = fuse_keys
        if self._fuse_keys is not None:
            rnn_input_size += sum(
                [observation_space.spaces[k].shape[0] for k in self._fuse_keys]
            )

        self._hidden_size = hidden_size

        if force_blind_policy:
            use_obs_space = spaces.Dict({})
        elif (
            self.include_visual_keys is not None
            and len(self.include_visual_keys) != 0
        ):
            use_obs_space = spaces.Dict(
                {
                    k: v
                    for k, v in observation_space.spaces.items()
                    if k in ["robot_head_rgb"]
                }
            )
        else:
            use_obs_space = observation_space

        # self.visual_encoder_rgb = ResNetEncoder(
        #     use_obs_space,
        #     baseplanes=resnet_baseplanes,
        #     ngroups=resnet_baseplanes // 2,
        #     make_backbone=getattr(resnet, backbone),
        # )

        if force_blind_policy:
            use_obs_space = spaces.Dict({})
        elif (
            self.include_visual_keys is not None
            and len(self.include_visual_keys) != 0
        ):
            use_obs_space = spaces.Dict(
                {
                    k: v
                    for k, v in observation_space.spaces.items()
                    if k in ["robot_head_depth"]
                }
            )
        else:
            use_obs_space = observation_space

        self.visual_encoder = ResNetEncoder(
            use_obs_space,
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes // 2,
            make_backbone=getattr(resnet, backbone),
        )

        if not self.visual_encoder.is_blind:
            # self.visual_fc_rgb = nn.Sequential(
            #     nn.Flatten(),
            #     nn.Linear(
            #         np.prod(self.visual_encoder.output_shape), hidden_size//2
            #     ),
            #     nn.ReLU(True),
            # )
            self.visual_fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(
                    np.prod(self.visual_encoder.output_shape), hidden_size // 2
                ),
                nn.ReLU(True),
            )
        self._hxs_dim = (self._hidden_size // 2) + rnn_input_size + num_actions
        self._num_actions = num_actions
        mconf = GPTConfig(
            num_actions,
            context_length,
            num_states=[
                (0 if self.is_blind else self._hidden_size // 2),
                rnn_input_size,
            ],
            n_layer=n_layer,
            n_head=n_head,
            n_embd=self._hidden_size,
            model_type=model_type,
            max_timestep=max_episode_step,
        )  # 6,8
        self.state_encoder = GPT(mconf)

        # self.state_encoder = LSTMBC(mconf)

        self.train()

    @property
    def hidden_state_hxs_dim(self):
        return self._hxs_dim

    @property
    def num_recurrent_layers(self):
        return self.context_length

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states,
        prev_actions,
        masks,
        rnn_build_seq_info=None,
        # targets=None,
        rtgs=None,
        # timesteps=None,
        offline_training=False
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        x = []
        B = prev_actions.shape[0]
        # if len(observations["joint"].shape) == len(prev_actions.shape):
        #     observations = {
        #         k: observations[k].reshape(-1, *observations[k].shape[2:])
        #         for k in observations.keys()
        #     }

        if not self.is_blind:
            if "visual_features" in observations:
                visual_feats = observations["visual_features"]
            else:
                # visual_feats = self.visual_encoder_rgb(observations)
                # visual_feats = self.visual_fc_rgb(visual_feats)
                # x.append(visual_feats)
                visual_feats = self.visual_encoder(observations)
                visual_feats = self.visual_fc(visual_feats)
                x.append(visual_feats)
                # visual_feats = self.visual_encoder_rgb(observations)
                # visual_feats = self.visual_fc_rgb(visual_feats)
                # x.append(visual_feats)

        if self._fuse_keys is not None:
            # observations["obj_start_gps_compass"] = torch.stack(
            #     [
            #         observations["obj_start_gps_compass"][:, 0],
            #         torch.cos(observations["obj_start_gps_compass"][:, 1]),
            #         torch.sin(observations["obj_start_gps_compass"][:, 1]),
            #     ]
            # ).permute(1, 0)
            # observations["obj_goal_gps_compass"] = torch.stack(
            #     [
            #         observations["obj_goal_gps_compass"][:, 0],
            #         torch.cos(observations["obj_goal_gps_compass"][:, 1]),
            #         torch.sin(observations["obj_goal_gps_compass"][:, 1]),
            #     ]
            # ).permute(1, 0)
            fuse_states = torch.cat(
                [observations[k] for k in self._fuse_keys], dim=-1
            )
            x.append(fuse_states)

        x = torch.cat(x, dim=1)
        x = x.reshape(B, -1, *x.shape[1:])

        if offline_training:
            # Move valid state-action-reward pair to the left
            out = self.state_encoder(
                x,
                prev_actions,
                rtgs=None,
            )
            return out

        rnn_hidden_states *= masks.view(-1, 1, 1)

        current_context = torch.argmax(
            (rnn_hidden_states.sum(-1) == 0).float(), -1
        )
        action_dim = self._num_actions

        # Write obs to context
        rnn_hidden_states[
            torch.arange(B), current_context, :-action_dim
        ] = x.view(B, -1)

        # Write actions to context
        rnn_hidden_states[
            torch.arange(B), current_context, -action_dim:
        ] = prev_actions.view(B, -1)

        out = self.state_encoder(
            rnn_hidden_states[..., :-action_dim],
            rnn_hidden_states[..., -action_dim:],
            rtgs=None,
        )

        # out = self.state_encoder(
        #     x,
        #     prev_actions,
        #     rtgs=None,
        # )

        return out[torch.arange(B), current_context], rnn_hidden_states, {}
        # return out[torch.arange(B), masks-1], rnn_hidden_states, {}