#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc
from os import times
from typing import Dict, List, Optional, Tuple

import torch
from gym import spaces
from torch import device, nn as nn
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
from habitat_baselines.rl.transformer_policy.transformer_model import (
    GPTConfig,
    GPT,
)
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.utils.common import get_num_actions
from habitat_baselines.rl.transformer_policy.action_distribution import (
    ActionDistribution,
)
from habitat_baselines.rl.ppo import NetPolicy
from habitat_baselines.common.tensor_dict import TensorDict


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
        # rtgs=None,
        # timesteps=None,
        # current_context=None,
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

        return out[torch.arange(B), current_context + 1], rnn_hidden_states, {}
