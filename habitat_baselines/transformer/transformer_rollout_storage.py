#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from typing import Iterator, Optional, Tuple

import numpy as np
import torch

from habitat_baselines.common.tensor_dict import TensorDict


class RolloutStorage:
    r"""Class for storing rollout information for RL trainers."""

    def __init__(
        self,
        numsteps,
        num_envs,
        observation_space,
        action_space,
        action_shape: Optional[Tuple[int]] = None,
        is_double_buffered: bool = False,
        discrete_actions: bool = True,
    ):
        self.buffers = TensorDict()
        self.buffers["observations"] = TensorDict()

        for sensor in observation_space.spaces:
            self.buffers["observations"][sensor] = torch.from_numpy(
                np.zeros(
                    (
                        numsteps + 1,
                        num_envs,
                        *observation_space.spaces[sensor].shape,
                    ),
                    dtype=observation_space.spaces[sensor].dtype,
                )
            )

        self.buffers["rewards"] = torch.zeros(numsteps + 1, num_envs, 1)
        self.buffers["cumulative_rewards"] = torch.zeros(numsteps + 1, num_envs, 1)
        self.buffers["returns"] = torch.zeros(numsteps + 1, num_envs, 1)

        if action_shape is None:
            if action_space.__class__.__name__ == "ActionSpace":
                action_shape = (1,)
            else:
                action_shape = action_space.shape

        self.buffers["actions"] = torch.zeros(
            numsteps + 1, num_envs, *action_shape
        )
        if (
            discrete_actions
            and action_space.__class__.__name__ == "ActionSpace"
        ):
            assert isinstance(self.buffers["actions"], torch.Tensor)
            self.buffers["actions"] = self.buffers["actions"].long()

        self.buffers["masks"] = torch.zeros(
            numsteps + 1, num_envs, 1, dtype=torch.bool
        )

        self.buffers["timesteps"] = torch.ones(
            numsteps + 1, num_envs, 1, dtype=torch.int64
        )

        self.is_double_buffered = is_double_buffered
        self._nbuffers = 2 if is_double_buffered else 1
        self._num_envs = num_envs

        assert (self._num_envs % self._nbuffers) == 0

        self.numsteps = numsteps
        self.current_rollout_step_idxs = [0 for _ in range(self._nbuffers)]

    @property
    def current_rollout_step_idx(self) -> int:
        assert all(
            s == self.current_rollout_step_idxs[0]
            for s in self.current_rollout_step_idxs
        )
        return self.current_rollout_step_idxs[0]

    def to(self, device):
        self.buffers.map_in_place(lambda v: v.to(device))

    def insert(
        self,
        next_observations=None,
        actions=None,
        rewards=None,
        cumulative_rewards=None,
        next_masks=None,
        timesteps=None,
        buffer_index: int = 0,
    ):
        if not self.is_double_buffered:
            assert buffer_index == 0

        next_step = dict(
            observations=next_observations,
            masks=next_masks,
            cumulative_rewards=cumulative_rewards,
        )

        current_step = dict(
            actions=actions,
            timesteps=timesteps,
            rewards=rewards,
        )

        next_step = {k: v for k, v in next_step.items() if v is not None}
        current_step = {k: v for k, v in current_step.items() if v is not None}

        env_slice = slice(
            int(buffer_index * self._num_envs / self._nbuffers),
            int((buffer_index + 1) * self._num_envs / self._nbuffers),
        )

        if len(next_step) > 0:
            self.buffers.set(
                (self.current_rollout_step_idxs[buffer_index] + 1, env_slice),
                next_step,
                strict=False,
            )

        if len(current_step) > 0:
            self.buffers.set(
                (self.current_rollout_step_idxs[buffer_index], env_slice),
                current_step,
                strict=False,
            )

    def advance_rollout(self, buffer_index: int = 0):
        self.current_rollout_step_idxs[buffer_index] += 1

    def after_update(self):
        for i in range(self._num_envs):
            self.buffers[0, i] = self.buffers[self.done_idxs[i][-1], i]
        self.current_rollout_step_idxs = [
            0 for _ in self.current_rollout_step_idxs
        ]

    def compute_returns(self, gamma=1.0):
        for step in reversed(range(self.current_rollout_step_idx)):
            self.buffers["returns"][step] = (
                gamma
                * self.buffers["returns"][step + 1]
                * self.buffers["masks"][step + 1]
                + self.buffers["rewards"][step]
            )
        self.done_idxs = []
        for i in range(self._num_envs):
            self.done_idxs.append(torch.nonzero(self.buffers["masks"][1:,i].squeeze() == False).squeeze(-1) + 1)

    def recurrent_generator(
        self, num_mini_batch, block_size
    ) -> Iterator[TensorDict]:
        num_environments = self._num_envs
        
        indices = torch.cat([torch.stack((torch.randperm(self.done_idxs[i][-1]), torch.ones(self.done_idxs[i][-1], dtype=torch.int32) * i)) for i in range(num_environments)], dim=-1)
        indices = indices.chunk((indices.shape[-1] // num_mini_batch) + 1, dim=-1)

        for inds in indices:
            batch = []
            for i,j in zip(inds[0], inds[1]):
                done_idx = self.done_idxs[j][torch.searchsorted(self.done_idxs[j].cpu(), i)]
                done_idx = torch.min(done_idx, i + block_size)
                start_idx = done_idx - block_size
                batch.append(self.buffers[start_idx : done_idx, j])

            yield batch

