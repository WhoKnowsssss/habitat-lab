import os, pickle

import numpy as np
import torch

from torch.utils.data import Dataset

from habitat.config import Config

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
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        block_size = self.block_size // 3
        done_idx = idx + block_size
        for i in self.done_idxs:
            if i > idx: # first done_idx greater than idx
                done_idx = min(int(i), done_idx)
                break
        idx = done_idx - block_size
        states = torch.tensor(np.array(self.data[idx:done_idx]), dtype=torch.float32).reshape(block_size, -1) # (block_size, 4*84*84)
        # states = states / 255.
        actions = torch.tensor(self.actions[idx:done_idx], dtype=torch.float32).unsqueeze(1) # (block_size, 1)
        rtgs = torch.tensor(self.rtgs[idx:done_idx], dtype=torch.float32).unsqueeze(1)
        timesteps = torch.tensor(self.timesteps[idx:idx+1], dtype=torch.int64).unsqueeze(1)

        return states, actions, rtgs, timesteps

    @classmethod
    def from_config(
        cls,
        config: Config,
    ):
        obss = []
        actions = []
        returns = [0]
        done_idxs = []
        stepwise_returns = []

        path, _, filenames = next(os.walk(config.trajectory_dir))
        transitions_per_buffer = np.zeros(len(filenames), dtype=int)
        num_trajectories = 0
        while len(obss) < config.steps_per_training:
            buffer_num = np.random.choice(np.arange(len(filenames)), 1)[0]
            i = transitions_per_buffer[buffer_num]
            print('loading from buffer %d which has %d already loaded' % (buffer_num, i))

            file = os.join(path, filenames[buffer_num])
            if os.path.exists(file):
                dataset_raw = pickle.load(open(file, "rb"))
                done = False
                curr_num_transitions = len(obss)
                trajectories_to_load = config.steps_per_file
                while not done:
                    buffer_index = np.random.randint(len(dataset_raw))
                    # states, ac, ret, next_states, next_action, next_reward, terminal, indices = frb.sample_transition_batch(batch_size=1, indices=[i])
                    states, ac, ret, terminal = np.concatenate([dataset_raw[buffer_index]["p_obs"][k].reshape(-1) for k in dataset_raw[buffer_index]["p_obs"].keys()]), dataset_raw[buffer_index]["act"], [dataset_raw[buffer_index]["rwd"]], [dataset_raw[buffer_index]["dne"]]
                    # states = states.transpose((0, 3, 1, 2))[0] # (1, 84, 84, 4) --> (4, 84, 84)
                    obss += [states]
                    actions += [ac] if ac.shape[0] > 1 else [ac[0]]
                    stepwise_returns += [ret[0]]
                    if terminal[0]:
                        done_idxs += [len(obss)]
                        returns += [0]
                        if trajectories_to_load == 0:
                            done = True
                        else:
                            trajectories_to_load -= 1
                    returns[-1] += ret[0]
                    i += 1
                    if i >= 100000:
                        obss = obss[:curr_num_transitions]
                        actions = actions[:curr_num_transitions]
                        stepwise_returns = stepwise_returns[:curr_num_transitions]
                        returns[-1] = 0
                        i = transitions_per_buffer[buffer_num]
                        done = True
                num_trajectories += (config.steps_per_file - trajectories_to_load)
                transitions_per_buffer[buffer_num] = i
            print('this buffer has %d loaded transitions and there are now %d transitions total divided into %d trajectories' % (i, len(obss), num_trajectories))

        actions = np.array(actions)
        returns = np.array(returns)
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
        print('max rtg is %d' % max(rtg))

        # -- create timestep dataset
        start_index = 0
        timesteps = np.zeros(len(actions)+1, dtype=int)
        for i in done_idxs:
            i = int(i)
            timesteps[start_index:i+1] = np.arange(i+1 - start_index)
            start_index = i+1
        print('max timestep is %d' % max(timesteps))

        return cls(
            obss, 
            actions, 
            returns, 
            done_idxs, 
            rtg, 
            timesteps
        )