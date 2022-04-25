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
from habitat.tasks.nav.object_nav_task import ObjectGoalSensor
from habitat_baselines.rl.ddppo.policy.resnet_policy import ResNetEncoder
from habitat_baselines.rl.ddppo.policy import resnet
from habitat_baselines.rl.ppo.policy import Policy
from habitat_baselines.transformer.transformer_model import GPTConfig, GPT
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.utils.common import get_num_actions


@baseline_registry.register_policy
class TransformerResNetPolicy(nn.Module, Policy):
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
        normalize_visual_inputs: bool = False,
        force_blind_policy: bool = False,
        policy_config: Config = None,
        fuse_keys: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__()
        self.context_length = context_length
        
        if policy_config is not None:
            discrete_actions = (
                policy_config.action_distribution_type == "categorical"
            )
            self.action_distribution_type = (
                policy_config.action_distribution_type
            )
            include_visual_keys = policy_config.include_visual_keys
        else:
            discrete_actions = True
            self.action_distribution_type = "categorical"
            include_visual_keys = None

        self.net = TransformerResnetNet(
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
            normalize_visual_inputs=normalize_visual_inputs,
            force_blind_policy=force_blind_policy,
            discrete_actions=discrete_actions,
            fuse_keys=fuse_keys,
            include_visual_keys=include_visual_keys
        )

    @classmethod
    def from_config(
        cls,
        config: Config,
        observation_space: spaces.Dict,
        action_space,
    ):
        return cls(
            observation_space=observation_space,
            action_space=action_space,
            hidden_size=config.RL.TRANSFORMER.hidden_size,
            context_length=config.RL.TRANSFORMER.context_length,
            max_episode_step=config.RL.TRANSFORMER.max_episode_step,
            model_type=config.RL.TRANSFORMER.model_type,
            n_head=config.RL.TRANSFORMER.n_head,
            n_layer=config.RL.TRANSFORMER.n_layer,
            backbone=config.RL.TRANSFORMER.backbone,
            normalize_visual_inputs=any(["rgb" in s for s in observation_space.spaces.keys()]),
            force_blind_policy=config.FORCE_BLIND_POLICY,
            policy_config=config.RL.POLICY,
            fuse_keys=config.RL.get("GYM_OBS_KEYS", None),
        )

    def act(
        self,
        observations,
        prev_actions=None,
        targets=None, 
        rtgs=None, 
        timesteps=None,
        valid_context=None,
    ):
        self.net.eval()
        logits, _ = self.net(observations, prev_actions=prev_actions, targets=targets, rtgs=rtgs, timesteps=timesteps, current_context=valid_context)

        logits = logits[range(logits.shape[0]), valid_context-1]
        return logits

    def forward(
        self,
        states, 
        actions, 
        targets, 
        rtgs,
        timesteps,
    ):
        _, loss = self.net(states, prev_actions=actions, targets=targets, rtgs=rtgs, timesteps=timesteps)

        return loss

class Net(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        pass

    @property
    @abc.abstractmethod
    def output_size(self):
        pass

    @property
    @abc.abstractmethod
    def is_blind(self):
        pass

class TransformerResnetNet(Net):
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
        normalize_visual_inputs: bool,
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

        if (
            IntegratedPointGoalGPSAndCompassSensor.cls_uuid
            in observation_space.spaces
        ):
            n_input_goal = (
                observation_space.spaces[
                    IntegratedPointGoalGPSAndCompassSensor.cls_uuid
                ].shape[0]
                + 1
            )
            self.tgt_embeding = nn.Linear(n_input_goal, 32)
            rnn_input_size += 32

        if ObjectGoalSensor.cls_uuid in observation_space.spaces:
            self._n_object_categories = (
                int(
                    observation_space.spaces[ObjectGoalSensor.cls_uuid].high[0]
                )
                + 1
            )
            self.obj_categories_embedding = nn.Embedding(
                self._n_object_categories, 32
            )
            rnn_input_size += 32

        if EpisodicGPSSensor.cls_uuid in observation_space.spaces:
            input_gps_dim = observation_space.spaces[
                EpisodicGPSSensor.cls_uuid
            ].shape[0]
            self.gps_embedding = nn.Linear(input_gps_dim, 32)
            rnn_input_size += 32

        if PointGoalSensor.cls_uuid in observation_space.spaces:
            input_pointgoal_dim = observation_space.spaces[
                PointGoalSensor.cls_uuid
            ].shape[0]
            self.pointgoal_embedding = nn.Linear(input_pointgoal_dim, 32)
            rnn_input_size += 32

        if HeadingSensor.cls_uuid in observation_space.spaces:
            input_heading_dim = (
                observation_space.spaces[HeadingSensor.cls_uuid].shape[0] + 1
            )
            assert input_heading_dim == 2, "Expected heading with 2D rotation."
            self.heading_embedding = nn.Linear(input_heading_dim, 32)
            rnn_input_size += 32

        if ProximitySensor.cls_uuid in observation_space.spaces:
            input_proximity_dim = observation_space.spaces[
                ProximitySensor.cls_uuid
            ].shape[0]
            self.proximity_embedding = nn.Linear(input_proximity_dim, 32)
            rnn_input_size += 32

        if EpisodicCompassSensor.cls_uuid in observation_space.spaces:
            assert (
                observation_space.spaces[EpisodicCompassSensor.cls_uuid].shape[
                    0
                ]
                == 1
            ), "Expected compass with 2D rotation."
            input_compass_dim = 2  # cos and sin of the angle
            self.compass_embedding = nn.Linear(input_compass_dim, 32)
            rnn_input_size += 32

        if ImageGoalSensor.cls_uuid in observation_space.spaces:
            goal_observation_space = spaces.Dict(
                {"rgb": observation_space.spaces[ImageGoalSensor.cls_uuid]}
            )
            self.goal_visual_encoder = ResNetEncoder(
                goal_observation_space,
                baseplanes=resnet_baseplanes,
                ngroups=resnet_baseplanes // 2,
                make_backbone=getattr(resnet, backbone),
                normalize_visual_inputs=normalize_visual_inputs,
            )

            self.goal_visual_fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(
                    np.prod(self.goal_visual_encoder.output_shape), hidden_size
                ),
                nn.ReLU(True),
            )

            rnn_input_size += hidden_size

        self._hidden_size = hidden_size
        
        if force_blind_policy:
            use_obs_space = spaces.Dict({})
        elif self.include_visual_keys is not None and len(self.include_visual_keys) != 0:
            use_obs_space = spaces.Dict(
                {
                    k: v
                    for k, v in observation_space.spaces.items()
                    if k in ['robot_head_depth']
                }
            )
        else:
            use_obs_space = observation_space
            
        self.visual_encoder_depth = ResNetEncoder(
            use_obs_space,
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes // 2,
            make_backbone=getattr(resnet, backbone),
            normalize_visual_inputs=normalize_visual_inputs,
        )

        if force_blind_policy:
            use_obs_space = spaces.Dict({})
        elif self.include_visual_keys is not None and len(self.include_visual_keys) != 0:
            use_obs_space = spaces.Dict(
                {
                    k: v
                    for k, v in observation_space.spaces.items()
                    if k in ['robot_head_rgb']
                }
            )
        else:
            use_obs_space = observation_space
            
        self.visual_encoder = ResNetEncoder(
            use_obs_space,
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes // 2,
            make_backbone=getattr(resnet, backbone),
            normalize_visual_inputs=normalize_visual_inputs,
        )

        if not self.visual_encoder.is_blind:
            self.visual_fc_depth = nn.Sequential(
                nn.Flatten(),
                nn.Linear(
                    np.prod(self.visual_encoder.output_shape), hidden_size
                ),
                nn.ReLU(True),
            )
            self.visual_fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(
                    np.prod(self.visual_encoder.output_shape), hidden_size//2
                ),
                nn.ReLU(True),
            )
        mconf = GPTConfig(num_actions, context_length, num_states=[(0 if self.is_blind else self._hidden_size//2), rnn_input_size],
                  n_layer=n_layer, n_head=n_head, n_embd=self._hidden_size, model_type=model_type, max_timestep=max_episode_step) # 6,8
        self.state_encoder = GPT(mconf)

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        prev_actions,
        targets=None,
        rtgs=None,
        timesteps=None,
        current_context=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        x = []
        B = prev_actions.shape[0]
        observations = {k: observations[k].reshape(-1, *observations[k].shape[2:]) for k in observations.keys()}

        if not self.is_blind:
            if "visual_features" in observations:
                visual_feats = observations["visual_features"]
            else:
                visual_feats = self.visual_encoder_depth(observations)

            visual_feats = self.visual_fc_depth(visual_feats)
            x.append(visual_feats)
            visual_feats = self.visual_encoder(observations)
            visual_feats = self.visual_fc(visual_feats)
            x.append(visual_feats)

        if self._fuse_keys is not None:
            fuse_states = torch.cat(
                [observations[k] for k in self._fuse_keys], dim=-1
            )
            x.append(fuse_states)

        if IntegratedPointGoalGPSAndCompassSensor.cls_uuid in observations:
            goal_observations = observations[
                IntegratedPointGoalGPSAndCompassSensor.cls_uuid
            ]
            if goal_observations.shape[1] == 2:
                # Polar Dimensionality 2
                # 2D polar transform
                goal_observations = torch.stack(
                    [
                        goal_observations[:, 0],
                        torch.cos(-goal_observations[:, 1]),
                        torch.sin(-goal_observations[:, 1]),
                    ],
                    -1,
                )
            else:
                assert (
                    goal_observations.shape[1] == 3
                ), "Unsupported dimensionality"
                vertical_angle_sin = torch.sin(goal_observations[:, 2])
                # Polar Dimensionality 3
                # 3D Polar transformation
                goal_observations = torch.stack(
                    [
                        goal_observations[:, 0],
                        torch.cos(-goal_observations[:, 1])
                        * vertical_angle_sin,
                        torch.sin(-goal_observations[:, 1])
                        * vertical_angle_sin,
                        torch.cos(goal_observations[:, 2]),
                    ],
                    -1,
                )

            x.append(self.tgt_embeding(goal_observations))

        if PointGoalSensor.cls_uuid in observations:
            goal_observations = observations[PointGoalSensor.cls_uuid]
            x.append(self.pointgoal_embedding(goal_observations))

        if ProximitySensor.cls_uuid in observations:
            sensor_observations = observations[ProximitySensor.cls_uuid]
            x.append(self.proximity_embedding(sensor_observations))

        if HeadingSensor.cls_uuid in observations:
            sensor_observations = observations[HeadingSensor.cls_uuid]
            sensor_observations = torch.stack(
                [
                    torch.cos(sensor_observations[0]),
                    torch.sin(sensor_observations[0]),
                ],
                -1,
            )
            x.append(self.heading_embedding(sensor_observations))

        if ObjectGoalSensor.cls_uuid in observations:
            object_goal = observations[ObjectGoalSensor.cls_uuid].long()
            x.append(self.obj_categories_embedding(object_goal).squeeze(dim=1))

        if EpisodicCompassSensor.cls_uuid in observations:
            compass_observations = torch.stack(
                [
                    torch.cos(observations[EpisodicCompassSensor.cls_uuid]),
                    torch.sin(observations[EpisodicCompassSensor.cls_uuid]),
                ],
                -1,
            )
            x.append(
                self.compass_embedding(compass_observations.squeeze(dim=1))
            )

        if EpisodicGPSSensor.cls_uuid in observations:
            x.append(
                self.gps_embedding(observations[EpisodicGPSSensor.cls_uuid])
            )

        if ImageGoalSensor.cls_uuid in observations:
            goal_image = observations[ImageGoalSensor.cls_uuid]
            goal_output = self.goal_visual_encoder({"rgb": goal_image})
            x.append(self.goal_visual_fc(goal_output))

        # if self.discrete_actions:
        #     prev_actions = prev_actions.squeeze(-1)
        #     start_token = torch.zeros_like(prev_actions)
        #     prev_actions = self.prev_action_embedding(
        #         torch.where(masks.view(-1), prev_actions + 1, start_token)
        #     )
        # else:
        #     prev_actions = self.prev_action_embedding(
        #         masks * prev_actions.float()
        #     )

        outs = torch.cat(x, dim=1)
        outs = outs.reshape(B, -1, *outs.shape[1:])

        assert (outs.shape[1] <= self.context_length), "Input Dimension Error"
        assert ((targets is not None) != (current_context is not None)), "Training or Evaluating? "
        # if outs.shape[1] < self.context_length and (current_context is not None):
        #     outs = torch.cat((torch.zeros((outs.shape[0], self.context_length - outs.shape[1], outs.shape[2]), device=outs.device), outs), dim=1)
        # if prev_actions.shape[1] < self.context_length and (current_context is not None):
        #     prev_actions = torch.cat((torch.zeros((prev_actions.shape[0], self.context_length - prev_actions.shape[1], prev_actions.shape[2]), device=prev_actions.device), prev_actions), dim=1)
        # if rtgs.shape[1] < self.context_length and (current_context is not None):
        #     rtgs = torch.cat((torch.zeros((rtgs.shape[0], self.context_length - rtgs.shape[1], rtgs.shape[2]), device=rtgs.device), rtgs), dim=1)

        # Move valid state-action-reward pair to the left
        if current_context is not None:
            rtgs_ = torch.zeros_like(rtgs)
            prev_actions_ = torch.zeros_like(prev_actions)
            outs_ = torch.zeros_like(outs)
            for i in range(current_context.shape[0]):
                rtgs_[i,:current_context[i],:] = rtgs[i,-current_context[i]:,:]
                prev_actions_[i,:current_context[i],:] = prev_actions[i,-current_context[i]:,:]
                outs_[i,:current_context[i],:] = outs[i,-current_context[i]:,:]
            logits, loss = self.state_encoder(outs_, prev_actions_, rtgs=rtgs_, timesteps=timesteps)
        else:
            logits, loss = self.state_encoder(outs, prev_actions, targets=prev_actions, rtgs=rtgs, timesteps=timesteps)

        return logits, loss