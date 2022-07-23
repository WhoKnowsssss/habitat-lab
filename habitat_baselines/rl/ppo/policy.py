#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc
from typing import Union

import torch
from gym import spaces
from torch import nn as nn

from habitat.config import Config
from habitat.tasks.nav.nav import (
    ImageGoalSensor,
    IntegratedPointGoalGPSAndCompassSensor,
    PointGoalSensor,
)
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.models.rnn_state_encoder import (
    build_rnn_state_encoder,
)
from habitat_baselines.rl.models.simple_cnn import SimpleCNN
from habitat_baselines.utils.common import (
    ActionDistributionNet,
    CategoricalNet,
    GaussianNet,
    get_num_action_logits,
    get_num_actions,
)


class Policy(abc.ABC):
    action_distribution: nn.Module
    supports_pausing: bool = True

    def __init__(self):
        pass

    @property
    def should_load_agent_state(self):
        return True

    @property
    def num_recurrent_layers(self) -> int:
        return 0

    def do_pause(self, state_index):
        pass

    def forward(self, *x):
        raise NotImplementedError

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def from_config(cls, config, observation_space, action_space, **kwargs):
        pass


class NetPolicy(nn.Module, Policy):
    action_distribution: ActionDistributionNet

    def __init__(self, aux_loss_config, net, action_space, policy_config=None):
        super().__init__()
        self.net = net
        self.action_distribution: Union[CategoricalNet, GaussianNet]

        # if policy_config is None:
        #     self.action_distribution_type = "categorical"
        # else:
        #     self.action_distribution_type = (
        #         policy_config.action_distribution_type
        #     )

        # if self.action_distribution_type == "categorical":
        #     self.action_distribution = CategoricalNet(
        #         self.net.output_size, self.dim_actions
        #     )
        # elif self.action_distribution_type == "gaussian":
        #     self.action_distribution = GaussianNet(
        #         self.net.output_size,
        #         self.dim_actions,
        #         policy_config.ACTION_DIST,
        #     )
        # else:
        #     ValueError(
        #         f"Action distribution {self.action_distribution_type}"
        #         "not supported."
        #     )

        # self.critic = CriticHead(self.net.output_size)

        if self.action_distribution_type == "categorical":
            self.dim_actions = action_space.n
            self.num_actions = 1
            self.action_distribution = CategoricalNet(
                self.net.output_size, self.dim_actions
            )
        elif self.action_distribution_type == "gaussian":
            self.dim_actions = get_num_action_logits(action_space)
            self.num_actions = get_num_actions(action_space)
            self.action_distribution = GaussianNet(
                self.net.output_size,
                self.dim_actions,
                policy_config.ACTION_DIST,
            )
        else:
            ValueError(
                f"Action distribution {self.action_distribution_type}"
                "not supported."
            )

        self.critic = CriticHead(self.net.output_size)
        self.aux_losses = nn.ModuleDict(
            {
                k: baseline_registry.get_auxiliary_loss(k)(
                    action_space,
                    self.net.output_size,
                    self.net.output_size,
                    **aux_loss_config.get(k),
                )
                for k in (
                    aux_loss_config.enabled
                    if aux_loss_config is not None
                    else []
                )
            }
        )

    @property
    def should_load_agent_state(self):
        return True

    @property
    def num_recurrent_layers(self) -> int:
        return self.net.num_recurrent_layers

    def forward(self, *x):
        raise NotImplementedError

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        features, rnn_hidden_states, _ = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        if deterministic:
            if self.action_distribution_type == "categorical":
                action = distribution.mode()
            elif self.action_distribution_type == "gaussian":
                action = distribution.mean
        else:
            action = distribution.sample()

        action_log_probs = distribution.log_probs(action)

        return (
            value,
            action,
            action_log_probs,
            rnn_hidden_states,
        )

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        features, _, _ = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        return self.critic(features)

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
        features, rnn_hidden_states, aux_loss_state = self.net(
            observations,
            rnn_hidden_states,
            prev_actions,
            masks,
            rnn_build_seq_info,
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy()

        aux_loss_res = {}
        if evaluate_aux_losses:
            for k, v in self.aux_losses.items():
                aux_loss_res[k] = v(
                    aux_loss_state,
                    dict(
                        observations=observations,
                        rnn_hidden_states=rnn_build_seq_info,
                        prev_actions=prev_actions,
                        masks=masks,
                        actions=action,
                        rnn_build_seq_info=rnn_build_seq_info,
                    ),
                )

        return (
            value,
            action_log_probs,
            distribution_entropy,
            rnn_hidden_states,
            aux_loss_res,
        )

    @classmethod
    @abc.abstractmethod
    def from_config(cls, config, observation_space, action_space, **kwargs):
        pass

    @property
    def policy_components(self):
        return [self.net, self.critic, self.action_distribution]

    def policy_parameters(self):
        for c in self.policy_components:
            yield from c.parameters()

    def all_policy_tensors(self):
        yield from self.policy_parameters()
        for c in self.policy_components:
            yield from c.buffers()

    def aux_loss_parameters(self):
        return {k: v.parameters() for k, v in self.aux_losses.items()}


class CriticHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
        nn.init.orthogonal_(self.fc.weight)
        self.fc.weight.data *= 1e-4
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x)


@baseline_registry.register_policy
class PointNavBaselinePolicy(NetPolicy):
    def __init__(
        self,
        aux_loss_config,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int = 512,
        **kwargs,
    ):
        super().__init__(
            aux_loss_config,
            PointNavBaselineNet(  # type: ignore
                observation_space=observation_space,
                hidden_size=hidden_size,
                **kwargs,
            ),
            action_space,
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
            aux_loss_config=config.RL.auxiliary_losses,
            observation_space=observation_space,
            action_space=action_space,
            hidden_size=config.RL.PPO.hidden_size,
        )


class Net(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        rnn_build_seq_info=None,
    ):
        pass

    @property
    @abc.abstractmethod
    def output_size(self):
        pass

    @property
    @abc.abstractmethod
    def num_recurrent_layers(self):
        pass

    @property
    @abc.abstractmethod
    def is_blind(self):
        pass


class PointNavBaselineNet(Net):
    r"""Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        hidden_size: int,
    ):
        super().__init__()

        if (
            IntegratedPointGoalGPSAndCompassSensor.cls_uuid
            in observation_space.spaces
        ):
            self._n_input_goal = observation_space.spaces[
                IntegratedPointGoalGPSAndCompassSensor.cls_uuid
            ].shape[0]
        elif PointGoalSensor.cls_uuid in observation_space.spaces:
            self._n_input_goal = observation_space.spaces[
                PointGoalSensor.cls_uuid
            ].shape[0]
        elif ImageGoalSensor.cls_uuid in observation_space.spaces:
            goal_observation_space = spaces.Dict(
                {"rgb": observation_space.spaces[ImageGoalSensor.cls_uuid]}
            )
            self.goal_visual_encoder = SimpleCNN(
                goal_observation_space, hidden_size
            )
            self._n_input_goal = hidden_size

        self._hidden_size = hidden_size

        self.visual_encoder = SimpleCNN(observation_space, hidden_size)

        self.state_encoder = build_rnn_state_encoder(
            self.rnn_input_size,
            self._hidden_size,
        )

        self.train()

    @property
    def rnn_input_size(self):
        return (
            (0 if self.is_blind else self._hidden_size) + self._n_input_goal,
        )

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def forward(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        rnn_build_seq_info=None,
    ):
        aux_loss_state = {}
        if IntegratedPointGoalGPSAndCompassSensor.cls_uuid in observations:
            target_encoding = observations[
                IntegratedPointGoalGPSAndCompassSensor.cls_uuid
            ]
        elif PointGoalSensor.cls_uuid in observations:
            target_encoding = observations[PointGoalSensor.cls_uuid]
        elif ImageGoalSensor.cls_uuid in observations:
            image_goal = observations[ImageGoalSensor.cls_uuid]
            target_encoding = self.goal_visual_encoder({"rgb": image_goal})

        x = [target_encoding]

        if not self.is_blind:
            perception_embed = self.visual_encoder(observations)
            aux_loss_state["perception_embed"] = perception_embed
            x = [perception_embed] + x

        x_out = torch.cat(x, dim=1)
        aux_loss_state["rnn_input"] = x_out
        x_out, rnn_hidden_states = self.state_encoder(
            x_out, rnn_hidden_states, masks, rnn_build_seq_info
        )
        aux_loss_state["rnn_output"] = x_out

        return x_out, rnn_hidden_states, aux_loss_state
