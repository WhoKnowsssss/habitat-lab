#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import functools
from typing import Tuple, Union

import numba
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence


@functools.wraps(numba.njit)
def _nb_njit_and_build_return_dict(*args):
    if len(args) == 0 or (len(args) == 1 and callable(args[0])):
        nb_jit = numba.njit
        fn = None if len(args) == 0 else args[0]
    else:
        nb_jit = numba.njit(*args)
        fn = None

    def _build(fn):
        jit_fn = nb_jit(fn)

        @functools.wraps(fn)
        def _fn(*args, **kwargs):
            keys, values = jit_fn(*args, **kwargs)

            return {k: v for k, v in zip(keys, values)}

        return _fn

    if fn is not None:
        return _build(fn)
    else:
        return _build


def _invert_permutation(permutation: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(permutation)
    output.scatter_(
        0,
        permutation,
        torch.arange(0, permutation.numel(), device=permutation.device),
    )
    return output


@numba.njit
def _np_invert_permutation(permutation: np.ndarray) -> np.ndarray:
    perm_shape = permutation.shape
    return np.argsort(permutation.ravel()).reshape(perm_shape)


@numba.extending.overload(np.unique)
def _np_unique(
    arr: np.ndarray, return_counts: bool = False
) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:

    if return_counts:

        def _impl(
            arr: np.ndarray, return_counts: bool = False
        ) -> Tuple[np.ndarray, np.ndarray]:
            assert return_counts
            arr = np.sort(arr.ravel())
            unq = list(arr[:1])
            counts = [1]
            prev = arr[0]
            for i in range(1, arr.size):
                if arr[i] == prev:
                    counts[-1] += 1
                else:
                    unq.append(arr[i])
                    counts.append(1)
                    prev = arr[i]

            unq = np.array(unq, dtype=arr.dtype)
            counts = np.array(counts, dtype=np.int64)

            return unq, counts

    else:

        def _impl(arr: np.ndarray, return_counts: bool = False) -> np.ndarray:
            assert not return_counts
            arr = np.sort(arr.ravel())
            unq = list(arr[:1])
            unq.extend(x for i, x in enumerate(arr[1:]) if arr[i] != x)

            return np.array(unq, dtype=arr.dtype)

    return _impl


@_nb_njit_and_build_return_dict
def _build_pack_info_from_episode_ids(episode_ids, actor_inds, step_ids):
    unsorted_episode_ids = episode_ids
    # Sort in increasing order of (episode ID, step_ids).  This will
    # put things into an order such that each episode is a contiguous
    # block. This makes a ton of the following logic MUCH easier
    sort_keys = episode_ids * (step_ids.max() + 1) + step_ids
    assert np.unique(sort_keys).size == sort_keys.size
    episode_id_sorting = np.argsort(
        episode_ids * (step_ids.max() + 1) + step_ids
    )
    episode_ids = episode_ids[episode_id_sorting]

    unique_episode_ids, sequence_lengths = np.unique(
        episode_ids, return_counts=True
    )
    # Exclusive cumsum
    sequence_starts = np.cumsum(sequence_lengths) - sequence_lengths

    sorted_indices = np.argsort(-sequence_lengths)
    lengths = sequence_lengths[sorted_indices]
    #  print(lengths)

    unique_episode_ids = unique_episode_ids[sorted_indices]
    sequence_starts = sequence_starts[sorted_indices]

    max_length = int(lengths[0])

    #  for i, eid in enumerate(unique_episode_ids):
    #  assert sequence_starts[i] == (episode_ids == eid).nonzero()[0].min()

    select_inds = np.empty((episode_ids.size,), dtype=np.int64)

    # num_seqs_at_step is *always* on the CPU
    num_seqs_at_step = np.empty((max_length,), dtype=np.int64)

    offset = 0
    prev_len = 0
    num_valid_for_length = lengths.shape[0]
    #  print(lengths)

    for next_len in np.unique(lengths):
        num_valid_for_length = (
            (lengths[0:num_valid_for_length] > prev_len).nonzero()[0].size
        )

        num_seqs_at_step[prev_len:next_len] = num_valid_for_length

        new_inds = (
            sequence_starts[0:num_valid_for_length].reshape(
                1, num_valid_for_length
            )
            + np.arange(prev_len, next_len).reshape(next_len - prev_len, 1)
        ).reshape(-1)

        select_inds[offset : offset + new_inds.size] = new_inds

        offset += new_inds.size

        prev_len = int(next_len)

    assert offset == select_inds.size

    select_inds = episode_id_sorting[select_inds]
    sequence_starts = select_inds[0 : num_seqs_at_step[0]]

    rnn_state_batch_inds = np.empty_like(sequence_starts)

    unique_actor_inds = np.unique(actor_inds)

    episode_actor_inds = actor_inds[sequence_starts]
    episode_ids_for_starts = unsorted_episode_ids[sequence_starts]
    actor_eps_masks = []
    last_sequence_in_batch_mask = np.zeros_like(episode_actor_inds == 0)
    first_sequence_in_batch_mask = np.zeros_like(last_sequence_in_batch_mask)
    for actor_id in unique_actor_inds:
        actor_eps = episode_actor_inds == actor_id
        actor_eps_ids = episode_ids_for_starts[actor_eps]
        actor_eps_masks.append(actor_eps)

        last_sequence_in_batch_mask[actor_eps] = (
            actor_eps_ids == actor_eps_ids.max()
        )
        first_sequence_in_batch_mask[actor_eps] = (
            actor_eps_ids == actor_eps_ids.min()
        )

    first_sequence_in_batch_mask_cumsum = np.cumsum(
        first_sequence_in_batch_mask.astype(np.int64)
    )
    for actor_eps in actor_eps_masks:
        first_ep_ind = int(
            (actor_eps & first_sequence_in_batch_mask).nonzero()[0].item()
        )
        rnn_state_batch_inds[actor_eps] = (
            first_sequence_in_batch_mask_cumsum[first_ep_ind] - 1
        )

    res = {
        "select_inds": select_inds,
        "num_seqs_at_step": num_seqs_at_step,
        "sequence_starts": sequence_starts,
        "sequence_lengths": lengths,
        "rnn_state_batch_inds": rnn_state_batch_inds,
        "last_sequence_in_batch_mask": last_sequence_in_batch_mask,
        "first_sequence_in_batch_mask": first_sequence_in_batch_mask,
        "last_sequence_in_batch_inds": np.nonzero(last_sequence_in_batch_mask)[
            0
        ],
        "first_episode_in_batch_inds": np.nonzero(
            first_sequence_in_batch_mask
        )[0],
    }

    return list(res.keys()), res.values()


@_nb_njit_and_build_return_dict
def _build_pack_info_from_dones(dones: np.ndarray):
    r"""Create the indexing info needed to make the PackedSequence
    based on the dones.

    PackedSequences are PyTorch's way of supporting a single RNN forward
    call where each input in the batch can have an arbitrary sequence length

    They work as follows: Given the sequences [c], [x, y, z], [a, b],
    we generate data [x, a, c, y, b, z] and num_seqs_at_step [3, 2, 1].  The
    data is a flattened out version of the input sequences (the ordering in
    data is determined by sequence length).  num_seqs_at_step tells you that
    for each index, how many sequences have a length of (index + 1) or greater.

    This method will generate the new index ordering such that you can
    construct the data for a PackedSequence from a (T*N, ...) tensor
    via x.index_select(0, select_inds)
    """
    T, N = dones.shape

    rollout_boundaries = dones.copy()
    # Force a rollout boundary for t=0.  We will use the
    # original dones for masking later, so this is fine
    # and simplifies logic considerably
    rollout_boundaries[0] = True
    rollout_boundaries = rollout_boundaries.nonzero()

    # The rollout_boundaries[:, 0]*N will make the sequence_starts index into
    # the T*N flattened tensors
    sequence_starts = rollout_boundaries[0] * N + rollout_boundaries[1]

    # We need to create a transposed start indexing so we can compute episode lengths
    # As if we make the starts index into a N*T tensor, then starts[1] - starts[0]
    # will compute the length of the 0th episode
    sequence_starts_transposed = (
        rollout_boundaries[1] * T + rollout_boundaries[0]
    )
    # Need to sort so the above logic is correct
    sorted_indices = np.argsort(sequence_starts_transposed)
    sequence_starts_transposed = sequence_starts_transposed[sorted_indices]

    # Calculate length of episode rollouts
    rollout_lengths = np.zeros_like(sequence_starts_transposed)
    rollout_lengths[:-1] = (
        sequence_starts_transposed[1:] - sequence_starts_transposed[:-1]
    )
    rollout_lengths[-1] = N * T - sequence_starts_transposed[-1]
    # Undo the sort above
    rollout_lengths = rollout_lengths[_np_invert_permutation(sorted_indices)]

    # Resort in descending order of episode length
    sorted_indices = np.argsort(-rollout_lengths)
    lengths = rollout_lengths[sorted_indices]

    sequence_starts = sequence_starts[sorted_indices]
    select_inds = np.empty((T * N), dtype=np.int64)

    max_length = int(lengths[0].item())
    # num_seqs_at_step is *always* on the CPU
    num_seqs_at_step = np.empty((max_length,), dtype=np.int64)

    offset = 0
    prev_len = 0
    num_valid_for_length = lengths.size

    for next_len in np.unique(lengths):
        num_valid_for_length = (
            (lengths[0:num_valid_for_length] > prev_len).nonzero()[0].size
        )

        num_seqs_at_step[prev_len:next_len] = num_valid_for_length

        # Creates this array
        # [step * N + start for step in range(prev_len, next_len)
        #                   for start in sequence_starts[0:num_valid_for_length]
        # * N because each step is seperated by N elements
        new_inds = (
            np.arange(prev_len, next_len).reshape(next_len - prev_len, 1) * N
            + sequence_starts[0:num_valid_for_length].reshape(
                1, num_valid_for_length
            )
        ).reshape(-1)

        select_inds[offset : offset + new_inds.size] = new_inds

        offset += new_inds.size

        prev_len = next_len

    # Make sure we have an index for all elements
    assert offset == T * N

    # This is used in conjunction with sequence_starts to get
    # the RNN hidden states
    rnn_state_batch_inds = sequence_starts % N
    # This indicates that a given episode is the last one
    # in that rollout.  In other words, there are N places
    # where this is True, and for each n, True indicates
    # that this episode is the last contiguous block of experience,
    # This is needed for getting the correct hidden states after
    # the RNN forward pass
    last_sequence_in_batch_mask = (
        (sequence_starts + (lengths - 1) * N) // N
    ) == (T - 1)
    first_sequence_in_batch_mask = (sequence_starts // N) == 0

    first_sequence_in_batch_inds = np.nonzero(first_sequence_in_batch_mask)[0]
    last_sequence_in_batch_inds = np.nonzero(last_sequence_in_batch_mask)[0]

    res = {
        "select_inds": select_inds,
        "num_seqs_at_step": num_seqs_at_step,
        "sequence_starts": sequence_starts,
        "sequence_lengths": lengths,
        "rnn_state_batch_inds": rnn_state_batch_inds,
        "last_sequence_in_batch_mask": last_sequence_in_batch_mask,
        "first_sequence_in_batch_mask": first_sequence_in_batch_mask,
        "last_sequence_in_batch_inds": last_sequence_in_batch_inds,
        "first_episode_in_batch_inds": first_sequence_in_batch_inds,
    }

    return list(res.keys()), res.values()


def build_rnn_inputs(
    x: torch.Tensor,
    rnn_states: torch.Tensor,
    not_dones,
    rnn_build_seq_info,
) -> Tuple[
    PackedSequence,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    r"""Create a PackedSequence input for an RNN such that each
    set of steps that are part of the same episode are all part of
    a batch in the PackedSequence.

    Use the returned select_inds and build_rnn_out_from_seq to invert this.

    :param x: A (T * N, -1) tensor of the data to build the PackedSequence out of
    :param rnn_states: A (-1, N, -1) tensor of the rnn_hidden_states

    :return: tuple(x_seq, rnn_states, select_inds, rnn_state_batch_inds, last_sequence_in_batch_mask)
        WHERE
        x_seq is the PackedSequence version of x to pass to the RNN

        rnn_states are the corresponding rnn state

        select_inds can be passed to build_rnn_out_from_seq to retrieve the
            RNN output

        rnn_state_batch_inds indicates which of the rollouts in the batch a hidden
            state came from/is for

        last_sequence_in_batch_mask indicates if an episode is the last in that batch.
            There will be exactly N places where this is True

    """

    select_inds = rnn_build_seq_info["select_inds"]
    num_seqs_at_step = rnn_build_seq_info["cpu_num_seqs_at_step"]

    x_seq = PackedSequence(
        x.index_select(0, select_inds), num_seqs_at_step, None, None
    )

    rnn_state_batch_inds = rnn_build_seq_info["rnn_state_batch_inds"]
    sequence_starts = rnn_build_seq_info["sequence_starts"]

    # Just select the rnn_states by batch index, the masking bellow will set things
    # to zero in the correct locations
    rnn_states = rnn_states.index_select(1, rnn_state_batch_inds)
    # Now zero things out in the correct locations
    rnn_states = torch.where(
        not_dones.view(1, -1, 1).index_select(1, sequence_starts),
        rnn_states,
        rnn_states.new_zeros(()),
    )

    return (
        x_seq,
        rnn_states,
    )


def build_rnn_out_from_seq(
    x_seq: PackedSequence,
    hidden_states,
    rnn_build_seq_info,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Construct the output of the rnn from a packed sequence returned by
        forward propping an RNN on the packed sequence returned by :ref:`build_rnn_inputs`.

    :param x_seq: The packed sequence output from the rnn
    :param hidden_statess: The hidden states output from the rnn
    :param select_inds: Returned from :ref:`build_rnn_inputs`
    :param rnn_state_batch_inds: Returned from :ref:`build_rnn_inputs`
    :param last_sequence_in_batch_mask: Returned from :ref:`build_rnn_inputs`
    :param N: The number of simulator instances in the batch of experience.
    """
    select_inds = rnn_build_seq_info["select_inds"]
    x = x_seq.data.index_select(0, _invert_permutation(select_inds))

    last_sequence_in_batch_inds = rnn_build_seq_info[
        "last_sequence_in_batch_inds"
    ]
    rnn_state_batch_inds = rnn_build_seq_info["rnn_state_batch_inds"]
    output_hidden_states = hidden_states.index_select(
        1,
        last_sequence_in_batch_inds[
            _invert_permutation(
                rnn_state_batch_inds[last_sequence_in_batch_inds]
            )
        ],
    )

    return x, output_hidden_states


class RNNStateEncoder(nn.Module):
    r"""RNN encoder for use with RL and possibly IL.

    The main functionality this provides over just using PyTorch's RNN interface directly
    is that it takes an addition masks input that resets the hidden state between two adjacent
    timesteps to handle episodes ending in the middle of a rollout.
    """

    def layer_init(self):
        for name, param in self.rnn.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)

    def pack_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states

    def unpack_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states.contiguous()

    def single_forward(
        self, x, hidden_states, masks
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Forward for a non-sequence input"""

        hidden_states = torch.where(
            masks.view(1, -1, 1), hidden_states, hidden_states.new_zeros(())
        )

        x, hidden_states = self.rnn(
            x.unsqueeze(0), self.unpack_hidden(hidden_states)
        )
        hidden_states = self.pack_hidden(hidden_states)

        x = x.squeeze(0)
        return x, hidden_states

    def seq_forward(
        self, x, hidden_states, masks, rnn_build_seq_info
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Forward for a sequence of length T

        Args:
            x: (T, N, -1) Tensor that has been flattened to (T * N, -1)
            hidden_states: The starting hidden state.
            masks: The masks to be applied to hidden state at every timestep.
                A (T, N) tensor flatten to (T * N)
        """

        (
            x_seq,
            hidden_states,
        ) = build_rnn_inputs(x, hidden_states, masks, rnn_build_seq_info)

        x_seq, hidden_states = self.rnn(
            x_seq, self.unpack_hidden(hidden_states)
        )
        hidden_states = self.pack_hidden(hidden_states)

        x, hidden_states = build_rnn_out_from_seq(
            x_seq,
            hidden_states,
            rnn_build_seq_info,
        )

        return x, hidden_states

    def forward(
        self, x, hidden_states, masks, rnn_build_seq_info=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states = hidden_states.permute(1, 0, 2)
        if x.size(0) == hidden_states.size(1):
            assert rnn_build_seq_info is None
            x, hidden_states = self.single_forward(x, hidden_states, masks)
        else:
            x, hidden_states = self.seq_forward(
                x, hidden_states, masks, rnn_build_seq_info
            )

        hidden_states = hidden_states.permute(1, 0, 2)

        return x, hidden_states


class LSTMStateEncoder(RNNStateEncoder):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
    ):
        super().__init__()

        self.num_recurrent_layers = num_layers * 2

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )

        self.layer_init()

    def pack_hidden(
        self, hidden_states: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        return torch.cat(hidden_states, 0)

    def unpack_hidden(
        self, hidden_states
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        lstm_states = torch.chunk(hidden_states.contiguous(), 2, 0)
        return (lstm_states[0], lstm_states[1])


class GRUStateEncoder(RNNStateEncoder):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
    ):
        super().__init__()

        self.num_recurrent_layers = num_layers

        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )

        self.layer_init()


def build_rnn_state_encoder(
    input_size: int,
    hidden_size: int,
    rnn_type: str = "GRU",
    num_layers: int = 1,
):
    r"""Factory for :ref:`RNNStateEncoder`.  Returns one with either a GRU or LSTM based on
        the specified RNN type.

    :param input_size: The input size of the RNN
    :param hidden_size: The hidden dimension of the RNN
    :param rnn_types: The type of the RNN cell.  Can either be GRU or LSTM
    :param num_layers: The number of RNN layers.
    """
    rnn_type = rnn_type.lower()
    if rnn_type == "gru":
        return GRUStateEncoder(input_size, hidden_size, num_layers)
    elif rnn_type == "lstm":
        return LSTMStateEncoder(input_size, hidden_size, num_layers)
    else:
        raise RuntimeError(f"Did not recognize rnn type '{rnn_type}'")
