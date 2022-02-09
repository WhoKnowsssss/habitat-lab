import os, pickle, time
from this import d
from typing import Any, ClassVar, Dict, List, Tuple, Union, Optional
import itertools as its
from collections import deque

import numpy as np
import torch

from torch.utils.data import (
    Dataset, 
    IterableDataset, 
    RandomSampler,
    SequentialSampler,
    DistributedSampler,
    get_worker_info
)

from habitat import Config, logger

class StateActionReturnDataset(Dataset):

    def __init__(self, data, block_size, actions, done_idxs, rtgs, timesteps):        
        self.block_size = block_size
        self.vocab_size = actions.shape[-1]
        self.data = data
        self.actions = actions
        self.done_idxs = done_idxs
        self.rtgs = rtgs
        self.timesteps = timesteps
    
    def __len__(self):
        assert len(self.data) > self.block_size, "No enough transitions in this dataset"
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        block_size = self.block_size // 3
        idx = idx + block_size
        done_idx = idx + block_size
        for i in self.done_idxs:
            if i > idx: # first done_idx greater than idx
                done_idx = min(int(i), done_idx)
                break
        idx = done_idx - block_size
        states = self.data[idx:done_idx]
        # states = states / 255.
        assert (idx >= 0), "Error on indexing"
        assert (len(states) == 30), "Error on states length"
        actions = torch.tensor(self.actions[idx:done_idx], dtype=torch.float32).unsqueeze(1) # (block_size, 1)
        rtgs = torch.tensor(self.rtgs[idx:done_idx], dtype=torch.float32).unsqueeze(1)
        timesteps = torch.tensor(self.timesteps[idx:idx+1], dtype=torch.int64).unsqueeze(1)

        return states, actions, rtgs, timesteps

    @classmethod
    def from_config(
        cls,
        config: Config,
        context_length: int=30,
        rng: np.random.Generator=None,
        verbose: bool=False
    ):
        obss = []
        actions = []
        returns = [0]
        done_idxs = []
        stepwise_returns = []

        path = config.trajectory_dir

        filenames = os.listdir(path)
        
        filenames.remove("model.pth")

        if verbose:
            logger.info(
                "Trajectory Files: {}".format(
                    filenames
                )
            )

        transitions_per_buffer = np.zeros(len(filenames), dtype=int)
        num_trajectories = 0
        while len(obss) < config.steps_per_load:
            buffer_num = rng.choice(np.arange(len(filenames)), 1)[0]
            i = transitions_per_buffer[buffer_num]
            if verbose:
                logger.info(
                    "Loading from buffer {} which has {} already loaded".format(
                        buffer_num, i
                    )
                )
            file = os.path.join(path, filenames[buffer_num])
            if os.path.exists(file):
                dataset_raw = torch.load(file, map_location=torch.device('cpu'))
                done = False
                curr_num_transitions = len(obss)
                trajectories_to_load = config.trajs_per_file
                buffer_index = rng.integers(0, len(dataset_raw["actions"]))
                while not done:
                    # states, ac, ret, next_states, next_action, next_reward, terminal, indices = frb.sample_transition_batch(batch_size=1, indices=[i])
                    states, ac, ret, terminal = {k:dataset_raw["obs"][buffer_index][k] for k in dataset_raw["obs"][0].keys()}, dataset_raw["actions"][buffer_index].numpy(), [dataset_raw["rewards"][buffer_index].numpy()], [not dataset_raw["masks"][buffer_index].numpy()]
                    # states = states.transpose((0, 3, 1, 2))[0] # (1, 84, 84, 4) --> (4, 84, 84)
                    obss += [states]
                    actions += [ac] if ac.shape[0] > 1 else [ac[0]]
                    stepwise_returns += [ret[0]]
                    if terminal[0] == 1:
                        done_idxs += [len(obss)]
                        curr_num_transitions = done_idxs[-1]
                        returns += [0]
                        if trajectories_to_load == 0:
                            done = True
                        else:
                            trajectories_to_load -= 1
                            buffer_index = rng.integers(0, len(dataset_raw["actions"]))
                    returns[-1] += ret[0]
                    i += 1
                    buffer_index += 1
                    
                    if i >= 110000:
                        obss = obss[:curr_num_transitions]
                        actions = actions[:curr_num_transitions]
                        stepwise_returns = stepwise_returns[:curr_num_transitions]
                        returns[-1] = 0
                        i = curr_num_transitions
                        done = True
                num_trajectories += (config.trajs_per_file - trajectories_to_load)
                transitions_per_buffer[buffer_num] = i
            if verbose:
                logger.info(
                    "This buffer has {} loaded transitions and there are now {} transitions total divided into {} trajectories. ".format(
                        i, len(obss), num_trajectories
                    )
                )

        actions = np.array(actions)
        returns = np.array(returns, dtype=np.float32)
        stepwise_returns = np.array(stepwise_returns)
        done_idxs = np.array(done_idxs)

        # -- create reward-to-go dataset
        start_index = 0
        rtg = np.zeros_like(stepwise_returns)
        for i in done_idxs:
            i = int(i)
            curr_traj_returns = stepwise_returns[start_index:i]
            for j in range(i-1, start_index-1, -1): # start from i-1
                rtg_j = curr_traj_returns[j-start_index:i-start_index]
                rtg[j] = sum(rtg_j)
            start_index = i
        # print('max rtg is %d' % max(rtg))

        # -- create timestep dataset
        start_index = 0
        timesteps = np.zeros(len(actions)+1, dtype=int)
        for i in done_idxs:
            i = int(i)
            timesteps[start_index:i+1] = np.arange(i+1 - start_index)
            start_index = i+1
        # print('max timestep is %d' % max(timesteps))

        if verbose:
            logger.info(
                "In this load, max rtg is {}, max timestep is {}. ".format(
                    rtg.max().round(2), timesteps.max()
                )
            )

        return cls(
            obss, 
            context_length,
            actions, 
            done_idxs, 
            rtg, 
            timesteps
        )

class RollingDataset(IterableDataset):

    class DatasetIterator:
        dataset: StateActionReturnDataset

        def __init__(self, 
            config: Config, 
            context_length: int, 
            sampler_params: Tuple, 
            dataset_context: Dict
        ):
            self.config = config
            self.context_length = context_length
            self.dataset_context = dataset_context
            self.steps_to_reload = config.steps_to_reload
            num_replicas, rank, self.seed = sampler_params
            assert (num_replicas is None == rank is None), "Local or Distributed Training? "
            if num_replicas is None:
                self._is_distributed = False
            else:
                self.num_replicas = num_replicas
                self.rank = rank
                self.steps_to_reload = self.steps_to_reload // num_replicas
                self._is_distributed = True

            self.dataset_context['num_iterated'] = 0
            self.dataset_context['num_init'] = 0
            self.num_iterated_epoch = 0
            
        def init_dataset(self):

            assert hasattr(self, 'seed_epoch'), "Set epoch before Dataloader loads"
            rng = np.random.default_rng(self.seed + self.seed_epoch)

            self.dataset = StateActionReturnDataset.from_config(self.config, self.context_length, rng, (self.id == 0))
                  
            self.dataset_context['num_init'] += 1

            if self._is_distributed:
                self.sampler = DistributedSampler(self.dataset, num_replicas=self.num_replicas, rank=self.rank, seed=self.seed, drop_last=True)
            else:
                self.sampler = SequentialSampler(self.dataset) # RandomSampler

        def __iter__(self):
            worker_info = get_worker_info()
            self.num_workers = worker_info.num_workers - 1 if worker_info is not None else 0
            self.id = worker_info.id if worker_info is not None else 0

            if worker_info is not None: 
                self.id = worker_info.id

            self.init_dataset()

            self.sampler_iterator = iter(self.sampler)

            if worker_info is not None: 
                if self.dataset_context['num_init'] == worker_info.num_workers:
                    self.dataset_context['num_init'] = -1
                next(its.islice(self.sampler_iterator, worker_info.id, worker_info.id), None)
            else:
                self.dataset_context['num_init'] = -1

            return self

        def __next__(self):
            
            self.num_iterated_epoch += 1

            try:
                idx = next(self.sampler_iterator)
            except StopIteration:
                
                self.dataset_context['num_iterated'] += self.num_iterated_epoch
                if self.dataset_context['num_iterated'] >= self.steps_to_reload:
                    self.dataset_context['num_iterated'] = 0
                    self.dataset_context['num_init'] = 0
                
                raise StopIteration

            item = self.dataset.__getitem__(idx)
            next(its.islice(self.sampler_iterator, self.num_workers, self.num_workers), None)
            return item

        def set_epoch(self, epoch):
            if self._is_distributed:
                self.sampler.set_epoch(epoch)
            self.seed_epoch = epoch

        
    def __init__(self, config: Config, context_length: int, sampler_params: Tuple, dataset_context: dict):
        self.iterator = self.DatasetIterator(config, context_length, sampler_params, dataset_context)

    def __iter__(self):
        return iter(self.iterator)

    def set_epoch(self, epoch):
        self.iterator.set_epoch(epoch)