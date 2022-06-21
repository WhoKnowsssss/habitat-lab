import math

import gym
import numpy as np
import torch
import torch.distributions
import torch.nn as nn
import torch.nn.functional as F

from habitat.core.spaces import ActionSpace, EmptySpace
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.models.rnn_state_encoder import _invert_permutation
from habitat_baselines.utils.common import (
    get_num_actions,
    iterate_action_space_recursively,
)


def masked_mean(t, valids, fill_value: float = 0.0):
    invalids = torch.logical_not(valids)
    t = torch.masked_fill(t, invalids, fill_value)

    return t.mean() / torch.clamp(
        torch.count_nonzero(valids) / valids.numel(), min=1.0
    )


def masked_index_select(
    t, indexer, valids, dim: int = 0, fill_value: float = 0.0
):
    indexer_size = indexer.size()
    not_valids = torch.logical_not(valids)

    indexer = torch.masked_fill(indexer, not_valids, 0).flatten()
    output = t.index_select(dim, indexer)

    mask_size = [1 for _ in range(output.dim())]
    mask_size[dim] = not_valids.numel()
    output.masked_fill_(not_valids.view(mask_size), fill_value)

    return output.unflatten(dim, indexer_size)


class BoxActionEmbedding(nn.Module):
    def __init__(self, action_space: gym.spaces.Box, dim_per_action: int = 32):
        super().__init__()

        self._ff_bands = dim_per_action // 2
        self.n_actions = action_space.shape[0]
        self.register_buffer(
            "_action_low",
            torch.as_tensor(action_space.low, dtype=torch.float32),
        )
        self.register_buffer(
            "_action_high",
            torch.as_tensor(action_space.high, dtype=torch.float32),
        )

        self.output_size = self._ff_bands * 2 * self.n_actions

    def forward(self, action, masks=None):
        action = action.float()
        if masks is not None:
            action = torch.where(masks, action, action.new_zeros(()))

        action = action.clamp(self._action_low, self._action_high)
        action -= self._action_low
        action *= (self._action_high - self._action_low) * 2
        action -= 1

        freqs = torch.logspace(
            start=0,
            end=self._ff_bands - 1,
            steps=self._ff_bands,
            base=2.0,
            device=action.device,
            dtype=action.dtype,
        ).mul_(math.pi)
        action = (action.unsqueeze(-1) * freqs).flatten(-2)

        return torch.cat((action.sin(), action.cos()), dim=-1)


class DiscreteActionEmbedding(nn.Module):
    def __init__(self, action_space: gym.spaces.Discrete, dim_per_action: int):
        super().__init__()
        self.n_actions = 1
        self.output_size = dim_per_action

        self.embedding = nn.Embedding(action_space.n + 1, dim_per_action)

    def forward(self, action, masks=None):
        action = action.long() + 1
        if masks is not None:
            action = torch.where(masks, action, action.new_zeros(()))

        return self.embedding(action.squeeze(-1))


class ActionEmbedding(nn.Module):
    def __init__(self, action_space: ActionSpace, dim_per_action: int = 32):
        super().__init__()

        self.embedding_modules = nn.ModuleList()
        self.embedding_slices = []

        all_spaces_empty = all(
            isinstance(space, EmptySpace)
            for space in iterate_action_space_recursively(action_space)
        )

        if all_spaces_empty:
            self.embedding_modules.append(
                DiscreteActionEmbedding(action_space, dim_per_action)
            )
            self.embedding_slices.append(slice(0, 1))
            self.output_size = self.embedding_modules[-1].output_size
        else:

            ptr = 0
            self.output_size = 0
            for space in iterate_action_space_recursively(action_space):
                if isinstance(space, gym.spaces.Box):
                    e = BoxActionEmbedding(space, dim_per_action)
                elif isinstance(space, gym.spaces.Discrete):
                    e = DiscreteActionEmbedding(space, dim_per_action)
                else:
                    raise RuntimeError(str(space))

                self.embedding_modules.append(e)
                self.embedding_slices.append(slice(ptr, ptr + e.n_actions))

                self.output_size += e.output_size
                ptr += e.n_actions

    def forward(self, action, masks=None):
        output = []
        for _slice, emb_mod in zip(
            self.embedding_slices, self.embedding_modules
        ):
            output.append(emb_mod(action[..., _slice], masks))

        return torch.cat(output, -1)


class ActionConditionedForwardModelingLoss(nn.Module):
    def __init__(
        self,
        action_space: gym.spaces.Dict,
        hidden_size: int,
        k: int = 20,
        time_subsample: int = 6,
        future_subsample: int = 2,
    ):
        super().__init__()

        self._action_embed = ActionEmbedding(action_space)

        self._future_predictor = nn.LSTM(
            self._action_embed.output_size, hidden_size
        )

        self.k = k
        self.time_subsample = time_subsample
        self.future_subsample = future_subsample
        self._hidden_size = hidden_size

        self.layer_init()

    def layer_init(self):
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() > 1:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)

    def _build_inds(self, rnn_build_seq_info):
        num_seqs_at_step = rnn_build_seq_info["num_seqs_at_step"]
        sequence_lengths = rnn_build_seq_info["cpu_sequence_lengths"]
        device = num_seqs_at_step.device

        shortest_seq = sequence_lengths.min()
        longest_seq = sequence_lengths.max()
        n_seqs = sequence_lengths.size(0)
        shortest_seq, longest_seq = map(
            lambda t: t.item(), (shortest_seq, longest_seq)
        )

        start_times = []
        seq_offsets = []
        max_valid = []
        for i in range(n_seqs):
            if self.time_subsample >= (sequence_lengths[i] - 1):
                start_times.append(
                    torch.arange(
                        1,
                        sequence_lengths[i],
                        device=device,
                        dtype=torch.int64,
                    )
                )
            else:
                start_times.append(
                    torch.randperm(
                        sequence_lengths[i], device=device, dtype=torch.int64
                    )[0 : self.time_subsample]
                )

            seq_offsets.append(torch.full_like(start_times[-1], i))
            max_valid.append(
                torch.full_like(start_times[-1], sequence_lengths[i] - 1)
            )

        start_times = torch.cat(start_times, dim=0)
        seq_offsets = torch.cat(seq_offsets, dim=0)
        max_valid = torch.cat(max_valid, dim=0)

        all_times = torch.arange(
            self.k, dtype=torch.int64, device=device
        ).view(-1, 1) + start_times.view(1, -1)

        action_valids = all_times < max_valid.view(1, -1)
        target_valids = (all_times + 1) < max_valid.view(1, -1)
        all_times[torch.logical_not(action_valids)] = 0

        time_start_inds = torch.cumsum(num_seqs_at_step, 0) - num_seqs_at_step
        action_inds = time_start_inds[all_times] + seq_offsets.view(1, -1)
        target_inds = time_start_inds[all_times + 1] + seq_offsets.view(1, -1)

        select_inds = rnn_build_seq_info["select_inds"]

        action_inds, target_inds = map(
            lambda t: select_inds.index_select(0, t.flatten()).view_as(t),
            (action_inds, target_inds),
        )

        return action_inds, target_inds, action_valids, target_valids

    def forward(self, aux_loss_state, batch):
        action = self._action_embed(batch["actions"])

        (
            action_inds,
            target_inds,
            action_valids,
            target_valids,
        ) = self._build_inds(batch["rnn_build_seq_info"])

        hidden_states = masked_index_select(
            aux_loss_state["rnn_output"], action_inds[0], action_valids[0]
        ).unsqueeze(0)
        action = masked_index_select(action, action_inds, action_valids)

        future_preds, _ = self._future_predictor(
            action, (hidden_states, hidden_states)
        )

        k = action.size(0)
        num_samples = self.future_subsample
        if num_samples < k:
            future_inds = torch.multinomial(
                action.new_full((), 1.0 / k).expand(action.size(1), k),
                num_samples=num_samples,
                replacement=False,
            )
        else:
            future_inds = (
                torch.arange(k, device=action.device, dtype=torch.int64)
                .view(k, 1)
                .expand(k, action.size(1))
            )

        future_inds = (
            future_inds
            + torch.arange(
                0,
                future_inds.size(0) * k,
                k,
                device=future_inds.device,
                dtype=future_inds.dtype,
            ).view(-1, 1)
        ).flatten()

        return (
            future_preds,
            action_inds,
            target_inds,
            future_inds,
            action_valids,
            target_valids,
        )


@baseline_registry.register_auxiliary_loss(name="cpca")
class CPCA(ActionConditionedForwardModelingLoss):
    def __init__(
        self,
        action_space: gym.spaces.Box,
        input_size: int,
        hidden_size: int,
        k: int = 20,
        time_subsample: int = 6,
        future_subsample: int = 2,
        num_negatives: int = 20,
    ):
        super().__init__(
            action_space, hidden_size, k, time_subsample, future_subsample
        )

        self._predictor_first_layers = nn.ModuleList(
            [
                nn.Linear(hidden_size, hidden_size, bias=True),
                nn.Linear(input_size, hidden_size, bias=False),
            ]
        )
        self._predictor = nn.Sequential(
            nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, 1),
        )

        self.num_negatives = num_negatives

        self.layer_init()

    def forward(self, aux_loss_state, batch):
        (
            future_preds,
            action_inds,
            target_inds,
            future_inds,
            action_valids,
            target_valids,
        ) = super().forward(aux_loss_state, batch)
        device = future_preds.device

        targets = aux_loss_state["perception_embed"]

        future_preds = self._predictor_first_layers[0](
            future_preds.flatten(0, 1)[future_inds]
        )
        positive_inds = target_inds.flatten()[future_inds]
        action_valids = action_valids.flatten()[future_inds]
        target_valids = target_valids.flatten()[future_inds]

        positive_targets = masked_index_select(
            targets, positive_inds, target_valids
        )
        positive_logits = self._predictor(
            future_preds + self._predictor_first_layers[1](positive_targets)
        )
        if "is_coeffs" in batch:
            is_coeffs = masked_index_select(
                batch["is_coeffs"].clamp(max=1.0), positive_inds, target_valids
            )
        else:
            is_coeffs = targets.new_full((), 1.0)
        positive_loss = masked_mean(
            F.binary_cross_entropy_with_logits(
                positive_logits,
                positive_logits.new_full((), 1.0).expand_as(positive_logits),
                reduction="none",
            )
            * is_coeffs,
            target_valids.view(-1, 1),
        )

        negative_inds_probs = targets.new_full(
            (positive_inds.size(0), targets.size(0)), 1.0
        )
        negative_inds_probs[
            torch.arange(
                positive_inds.size(0), device=device, dtype=torch.int64
            ),
            positive_inds,
        ] = 0.0
        negative_inds_probs = negative_inds_probs / negative_inds_probs.sum(
            -1, keepdim=True
        )

        negatives_inds = torch.multinomial(
            negative_inds_probs,
            num_samples=self.num_negatives,
            replacement=self.num_negatives > negative_inds_probs.size(-1),
        )

        negative_targets = targets.index_select(
            0, negatives_inds.flatten()
        ).unflatten(0, negatives_inds.size())
        negative_logits = self._predictor(
            future_preds.unsqueeze(1)
            + self._predictor_first_layers[1](negative_targets)
        )

        negative_loss = F.binary_cross_entropy_with_logits(
            negative_logits,
            negative_logits.new_zeros(()).expand_as(negative_logits),
            reduction="none",
        ) * is_coeffs.view(-1, 1, 1)
        negative_loss = masked_mean(
            negative_loss, target_valids.view(-1, 1, 1)
        )

        loss = 0.1 * (positive_loss + negative_loss)

        return dict(loss=loss)
