import os, pickle, time
from threading import Thread
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
from habitat_baselines.transformer.dataset_utils import producer

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
        buffer: Tuple,
        context_length: int=30,
    ):

        obss, actions, done_idxs, rtg, timesteps = buffer
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
            dataset_context: Dict,
            world_rank: bool,
        ):
            self.config = config
            self.context_length = context_length
            self.dataset_context = dataset_context
            self.steps_to_reload = config.steps_to_reload
            self.world_rank = world_rank
            num_replicas, rank, self.seed = sampler_params
            assert ((num_replicas is None) == (rank is None)), "Local or Distributed Training? "
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
            self.queue = deque()
            self.producer = None
            
            
        def init_dataset(self):

            assert hasattr(self, 'seed_epoch'), "Set epoch before Dataloader loads"
            
            while len(self.queue) == 0:
                time.sleep(1)
            self.dataset = StateActionReturnDataset.from_config(self.queue.popleft(), self.context_length)
                  
            self.dataset_context['num_init'] += 1

            if self._is_distributed:
                self.sampler = DistributedSampler(self.dataset, num_replicas=self.num_replicas, rank=self.rank, seed=(self.seed + self.seed_epoch), drop_last=True)
            else:
                self.sampler = SequentialSampler(self.dataset) # RandomSampler

        def __iter__(self):
            worker_info = get_worker_info()
            self.num_workers = worker_info.num_workers - 1 if worker_info is not None else 0
            self.id = worker_info.id if worker_info is not None else 0

            rng = np.random.default_rng(self.seed)
            if self.producer is None:
                self.producer = Thread(target=producer, args=(self.config, rng, self.queue, False))
                self.producer.start()

            if self.dataset_context['num_init'] != -1:
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
                try:
                    self.sampler.set_epoch(epoch)
                except:
                    pass
            self.seed_epoch = epoch

        
    def __init__(self, config: Config, context_length: int, sampler_params: Tuple, dataset_context: dict, world_rank: bool):
        self.iterator = self.DatasetIterator(config, context_length, sampler_params, dataset_context, world_rank)

    def __iter__(self):
        return iter(self.iterator)

    def set_epoch(self, epoch):
        self.iterator.set_epoch(epoch)