#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from gym import spaces
from torch import nn as nn
from torch.nn import functional as F

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
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ddppo.erik_policy import resnet
from habitat_baselines.rl.ddppo.erik_policy.rnn_state_encoder import (
    build_rnn_state_encoder,
)
from habitat_baselines.rl.ddppo.policy.running_mean_and_var import (
    RunningMeanAndVar,
)
from habitat_baselines.rl.models.cpc_aux_loss import ActionEmbedding
from habitat_baselines.rl.ppo import Net, NetPolicy
from habitat_baselines.utils.common import get_num_actions, sum_tensor_list


def _apply_projections(inputs, projection_weights):
    inp = torch.cat(inputs, -1)
    proj_weight = torch.cat(projection_weights, -1)

    return F.linear(inp, proj_weight, None)


@baseline_registry.register_policy
class ErikPointNavResNetPolicy(NetPolicy):
    def __init__(
        self,
        aux_loss_config,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int = 512,
        num_recurrent_layers: int = 1,
        rnn_type: str = "GRU",
        resnet_baseplanes: int = 32,
        backbone: str = "resnet18",
        normalize_visual_inputs: bool = False,
        force_blind_policy: bool = False,
        policy_config: Config = None,
        fuse_keys: Optional[List[str]] = None,
        **kwargs
    ):
        if policy_config is not None:
            self.action_distribution_type = (
                policy_config.action_distribution_type
            )
            include_visual_keys = policy_config.include_visual_keys
        else:
            self.action_distribution_type = "categorical"
            include_visual_keys = None

        if fuse_keys is None:
            fuse_keys = []

        super().__init__(
            aux_loss_config,
            PointNavResNetNet(
                observation_space=observation_space,
                action_space=action_space,  # for previous action
                hidden_size=hidden_size,
                num_recurrent_layers=num_recurrent_layers,
                rnn_type=rnn_type,
                backbone=backbone,
                resnet_baseplanes=resnet_baseplanes,
                normalize_visual_inputs=normalize_visual_inputs,
                force_blind_policy=force_blind_policy,
                fuse_keys=fuse_keys,
                include_visual_keys=include_visual_keys,
            ),
            action_space,
            policy_config=policy_config,
        )

    @classmethod
    def from_config(
        cls,
        config: Config,
        observation_space: spaces.Dict,
        action_space,
    ):
        if "GYM" in config.TASK_CONFIG:
            fuse_keys = config.TASK_CONFIG.GYM.OBS_KEYS
        else:
            fuse_keys = config.RL.GYM_OBS_KEYS

        return cls(
            aux_loss_config=config.RL.auxiliary_losses,
            observation_space=observation_space,
            action_space=action_space,
            hidden_size=config.RL.PPO.hidden_size,
            rnn_type=config.RL.DDPPO.rnn_type,
            num_recurrent_layers=config.RL.DDPPO.num_recurrent_layers,
            backbone=config.RL.DDPPO.backbone,
            normalize_visual_inputs="rgb" in observation_space.spaces,
            force_blind_policy=config.FORCE_BLIND_POLICY,
            policy_config=config.RL.POLICY,
            fuse_keys=fuse_keys,
        )


class ResNetEncoder(nn.Module):
    def __init__(
        self,
        observation_space: spaces.Dict,
        baseplanes: int = 32,
        ngroups: int = 32,
        spatial_size: int = 128,
        make_backbone=None,
        normalize_visual_inputs: bool = False,
    ):
        super().__init__()

        # Determine which visual observations are present
        self.rgb_keys = [k for k in observation_space.spaces if "rgb" in k]
        self.depth_keys = [k for k in observation_space.spaces if "depth" in k]

        # Count total # of channels for rgb and for depth
        self._n_input_rgb, self._n_input_depth = [
            # sum() returns 0 for an empty list
            sum([observation_space.spaces[k].shape[2] for k in keys])
            for keys in [self.rgb_keys, self.depth_keys]
        ]

        if normalize_visual_inputs:
            self.running_mean_and_var: nn.Module = RunningMeanAndVar(
                self._n_input_depth + self._n_input_rgb
            )
        else:
            self.running_mean_and_var = nn.Sequential()

        if not self.is_blind:
            input_channels = self._n_input_depth + self._n_input_rgb
            self.backbone = make_backbone(input_channels, baseplanes, ngroups)
            self._mode = self.backbone._mode

            mock_input = self._build_mock_input(observation_space)
            
            with torch.no_grad():
                # if self._mode == "v1":
                #     mock_input = F.avg_pool2d(mock_input, 2)

                mock_output = self.backbone(mock_input)

                (
                    _,
                    final_channels,
                    final_spatial_h,
                    final_spatial_w,
                ) = mock_output.size()

            after_compression_flat_size = 1024
            channels_per_group = 16
            num_compression_channels = int(
                channels_per_group
                * round(
                    after_compression_flat_size
                    / (final_spatial_h * final_spatial_w)
                    / channels_per_group
                )
            )
            self.compression = nn.Sequential(
                resnet.conv3x3(
                    final_channels,
                    num_compression_channels,
                    dilation=2 if self._mode == "v2" else 1,
                ),
                nn.GroupNorm(
                    num_compression_channels // channels_per_group,
                    num_compression_channels,
                ),
                nn.ReLU(True),
            )

            self.output_shape = (
                num_compression_channels,
                final_spatial_h,
                final_spatial_w,
            )
            with torch.no_grad():
                assert (
                    self.compression(mock_output).size()[1:]
                    == self.output_shape
                )

    @property
    def is_blind(self):
        return self._n_input_rgb + self._n_input_depth == 0

    def _build_mock_input(self, observation_space) -> torch.Tensor:
        obs = {
            k: torch.from_numpy(np.ones((1, *v.shape), dtype=v.dtype))
            for k, v in observation_space.spaces.items()
        }
        return self._build_cnn_input(obs)

    def layer_init(self):
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    layer.weight, nn.init.calculate_gain("relu")
                )
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

    def _build_cnn_input(
        self, observations: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        float_type = (
            torch.float16 if torch.is_autocast_enabled() else torch.float32
        )
        cnn_input = []
        for k in self.rgb_keys:
            rgb_observations = observations[k]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            rgb_observations = rgb_observations.permute(0, 3, 1, 2)
            rgb_observations = rgb_observations.to(
                dtype=float_type, copy=True
            ).mul_(
                1.0 / 255.0
            )  # normalize RGB
            cnn_input.append(rgb_observations)

        for k in self.depth_keys:
            depth_observations = observations[k]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            depth_observations = depth_observations.permute(0, 3, 1, 2).to(
                dtype=float_type
            )
            cnn_input.append(depth_observations)

        return torch.cat(cnn_input, dim=1)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:  # type: ignore
        if self.is_blind:
            return None

        x = self._build_cnn_input(observations)

        # if self._mode == "v1":
        #     x = F.avg_pool2d(x, 2)

        x = self.running_mean_and_var(x)
        x = self.backbone(x)
        x = self.compression(x)
        return x


class PointNavResNetNet(Net):
    """Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    prev_action_embedding: nn.Module

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int,
        num_recurrent_layers: int,
        rnn_type: str,
        backbone,
        resnet_baseplanes,
        normalize_visual_inputs: bool,
        force_blind_policy: bool = False,
        fuse_keys: Optional[List[str]] = None,
        include_visual_keys: Optional[List[str]] = None,
    ):
        super().__init__()
        if action_space is not None:
            self.prev_action_embedding: nn.Module = ActionEmbedding(
                action_space, dim_per_action=16
            )
            self._n_prev_action = self.prev_action_embedding.output_size
        else:
            self.prev_action_embedding = None
            self._n_prev_action = 0

        rnn_input_size = self._n_prev_action  # test

        # Only fuse the 1D state inputs. Other inputs are processed by the
        # visual encoder
        if fuse_keys is None:
            fuse_keys = []

        self._fuse_keys: List[str] = sorted(
            k for k in fuse_keys if len(observation_space.spaces[k].shape) == 1
        )
        if len(self._fuse_keys) != 0:
            self._fuse_projections = nn.ModuleDict(
                {
                    k: nn.Linear(
                        observation_space.spaces[k].shape[0],
                        hidden_size,
                        bias=False,
                    )
                    for k in self._fuse_keys
                }
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
        elif include_visual_keys is not None and len(include_visual_keys) != 0:
            use_obs_space = spaces.Dict(
                {
                    k: v
                    for k, v in observation_space.spaces.items()
                    if k in include_visual_keys
                }
            )
        else:
            use_obs_space = observation_space

        self.visual_encoders = nn.ModuleDict()
        self.visual_fcs = nn.ModuleDict()

        total_obs_space = spaces.Dict(use_obs_space.spaces)
        for k in ["head", "arm", "all"]:
            this_obs_space = spaces.Dict(
                {
                    ok: v
                    for ok, v in total_obs_space.items()
                    if k in ok or k == "all"
                }
            )
            total_obs_space = spaces.Dict(
                {
                    ok: v
                    for ok, v in total_obs_space.items()
                    if ok not in this_obs_space.spaces
                }
            )
            visual_encoder = ResNetEncoder(
                this_obs_space,
                baseplanes=resnet_baseplanes,
                ngroups=resnet_baseplanes // 2,
                make_backbone=getattr(resnet, backbone),
                normalize_visual_inputs=normalize_visual_inputs,
            )

            if not visual_encoder.is_blind:
                visual_fc = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(
                        np.prod(visual_encoder.output_shape),
                        hidden_size,
                    ),
                    nn.ReLU(True),
                )

                self.visual_encoders[k] = visual_encoder
                self.visual_fcs[k] = visual_fc

        self.rnn_input_size = (
            0
            if self.is_blind and len(self._fuse_keys) == 0
            else self._hidden_size
        ) + rnn_input_size
        self.state_encoder = build_rnn_state_encoder(
            self.rnn_input_size,
            self._hidden_size,
            rnn_type=rnn_type,
            num_layers=num_recurrent_layers,
        )

        self.train()
        self.apply(self.init_fn)

    def init_fn(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(m.weight, 0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return (
            len(self.visual_encoders) == 0
            or list(self.visual_encoders.values())[0].is_blind
        )

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states,
        prev_actions,
        masks,
        rnn_build_seq_info=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        aux_loss_state = {}
        x = []
        if not self.is_blind:
            all_visual_feats = []
            for i, k in enumerate(self.visual_encoders.keys()):
                enc = self.visual_encoders[k]
                fc = self.visual_fcs[k]
                all_visual_feats.append(fc(enc(observations)))

            visual_feats = sum_tensor_list(all_visual_feats)

            aux_loss_state["perception_embed"] = visual_feats
            x.append(visual_feats)

        if len(self._fuse_keys) != 0:
            fuse_states = _apply_projections(
                [observations[k] for k in self._fuse_keys],
                [self._fuse_projections[k].weight for k in self._fuse_keys],
            )

            x = [sum_tensor_list(x + [fuse_states])]

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

        if self.prev_action_embedding is not None:
            prev_actions = self.prev_action_embedding(prev_actions, masks)

            x.append(prev_actions)

        out = torch.cat(x, dim=1)
        aux_loss_state["rnn_input"] = out
        out, rnn_hidden_states = self.state_encoder(
            out, rnn_hidden_states, masks, rnn_build_seq_info
        )
        aux_loss_state["rnn_output"] = out

        return out, rnn_hidden_states, aux_loss_state
