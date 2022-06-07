import os, pickle, time
from this import d
from typing import Any, ClassVar, Dict, List, Tuple, Union, Optional
import itertools as its
from collections import deque

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate
import numba


from habitat import Config, logger 
from habitat_baselines.common.tensor_dict import TensorDict
        

def read_dataset(
    config: Config, 
    verbose: bool,
    rng: np.random.Generator
):        
    obss = []
    actions = []
    done_idxs = []
    stepwise_returns = []

    path = config.trajectory_dir

    filenames = os.listdir(path)
    filenames = ['{}.pt'.format(i*10) for i in range(1,2)]

    if verbose:
        logger.info(
            "Trajectory Files: {}".format(
                filenames
            )
        )

    transitions_per_buffer = np.zeros(len(filenames), dtype=int)
    num_trajectories = 0
    previous_done = 0
    while len(obss) < config.files_per_load:
        buffer_num = rng.choice(np.arange(len(filenames)), 1, replace=False)[0]
        i = transitions_per_buffer[buffer_num]
        if verbose:
            logger.info(
                "Loading from buffer {}".format(
                    buffer_num, i
                )
            )
        file = os.path.join(path, filenames[buffer_num])
        if os.path.exists(file):
            dataset_raw = torch.load(file, map_location=torch.device('cpu'))

            if len(dataset_raw["obs"]) == len(dataset_raw["actions"]):
                temp_obs = np.array(dataset_raw["obs"])
            else:
                temp_obs = np.array(dataset_raw["obs"][:-1])
            temp_actions = torch.stack(dataset_raw["actions"]).numpy()
            temp_stepwise_returns = torch.cat(dataset_raw["rewards"]).numpy()
            temp_dones = torch.cat(dataset_raw["masks"]).numpy()
            temp_infos = np.array([v["success"] for v in dataset_raw["infos"]])

            
            #==================== Only Arm Action Phase ===================
            # stepwise_idx = np.argwhere(np.all(temp_actions[:,8:10] == 0, axis=-1) & temp_dones == True).squeeze()

            # temp_obs = np.delete(temp_obs, stepwise_idx, 0)
            # temp_actions = np.delete(temp_actions, stepwise_idx, 0)
            # temp_stepwise_returns = np.delete(temp_stepwise_returns, stepwise_idx, 0)
            # temp_dones = np.delete(temp_dones, stepwise_idx, 0)

            
            #==================== Only Successful Episodes ===================
            # stepwise_idx = np.argwhere(temp_infos == False).squeeze()

            # temp_obs = np.delete(temp_obs, stepwise_idx, 0)
            # temp_actions = np.delete(temp_actions, stepwise_idx, 0)
            # temp_stepwise_returns = np.delete(temp_stepwise_returns, stepwise_idx, 0)
            # temp_dones = np.delete(temp_dones, stepwise_idx, 0)
            
            temp_done_idxs = np.argwhere(temp_dones == False).reshape(-1) + 1
            if len(temp_done_idxs) == 0:
                print("EVEN NO SUCCESS IN THIS FILE!!!!!!")
                continue
            
            #==================== Only Episodes that are larger than 30 ===================
            temp_start_idxs = np.roll(temp_done_idxs, -1)
            temp_start_idxs[0] = 0

            idx = np.nonzero(temp_done_idxs[:] - temp_start_idxs[:] < 30)[0]
            if len(idx) > 0:

                stepwise_idx = np.concatenate([np.arange(temp_start_idxs[i] , temp_done_idxs[i]) for i in idx])

                temp_obs = np.delete(temp_obs, stepwise_idx, 0)
                temp_actions = np.delete(temp_actions, stepwise_idx, 0)
                temp_stepwise_returns = np.delete(temp_stepwise_returns, stepwise_idx, 0)
                temp_dones = np.delete(temp_dones, stepwise_idx, 0)

            temp_done_idxs = np.argwhere(temp_dones == False).reshape(-1) + 1

            
            #==================== Categorize Gripper Action ===================
            temp_actions = np.clip(temp_actions, -1, 1)
            temp_pick_action = temp_actions[:,7]
            temp_pick_action = np.nonzero((temp_pick_action[1:] - temp_pick_action[:-1]) < -0.1)[0] + 1
            temp_place_idx = np.searchsorted(temp_pick_action, temp_done_idxs, side='right') - 1
            temp_pick_action = temp_pick_action[temp_place_idx].reshape(-1)
            stepwise_idx = np.concatenate([np.arange(temp_pick_action[i] , temp_done_idxs[i]) for i in range(temp_pick_action.shape[0])])
            temp_actions[stepwise_idx, 7] = 0
            
            l = temp_done_idxs[1:] - temp_done_idxs[:-1]
            # debug
            assert all(l <= 1000), f"Length too long: file:  {file}  dn:  {temp_done_idxs}"
            # assert all(l >= 30), f"Length too short: file:  {file}  dn:  {temp_done_idxs}"
            # print(f"file:  {file}  dn:  {temp_done_idxs}")

            obss += [temp_obs]
            actions += [temp_actions]
            done_idxs += [temp_done_idxs + previous_done]
            previous_done += len(temp_actions)
            stepwise_returns += [temp_stepwise_returns]
    
    actions = np.concatenate(actions)
    obss = np.concatenate(obss).tolist()
    stepwise_returns = np.concatenate(stepwise_returns)
    done_idxs = np.concatenate(done_idxs)

    rtg, timesteps = _timesteps_rtg(done_idxs, stepwise_returns)

    if verbose:
        logger.info(
            "In this load, max rtg is {}, max timestep is {}. ".format(
                rtg.max().round(2), timesteps.max()
            )
        )

    obss = TensorDict.from_tree(default_collate(obss))
    actions = torch.from_numpy(actions).to(torch.float32)
    rtg = torch.from_numpy(rtg).to(torch.float32)
    timesteps = torch.from_numpy(timesteps).to(torch.int64)
    return obss, actions, done_idxs, rtg, timesteps

@numba.jit(nopython=True, parallel=True)
def _timesteps_rtg(done_idxs, stepwise_returns):
    rtg = np.zeros_like(stepwise_returns)
    timesteps = np.zeros(len(stepwise_returns), dtype=np.int64)
    start_index = np.concatenate((np.array([0], dtype=np.int64), done_idxs[:-1]))
    for i in numba.prange(len(done_idxs)):
        start = start_index[i]
        done = done_idxs[i]
        curr_traj_returns = stepwise_returns[start:done]

        for j in numba.prange(done - start):
            rtg[j+start] = np.sum(curr_traj_returns[j:])
        
        timesteps[start:done] = np.arange(done - start)
    return rtg, timesteps


def producer(
    config: Config, 
    rng: np.random.Generator,
    deque: deque,
    verbose: bool,
):
    while True:
        if len(deque) < 1:
            import time
            s = time.time_ns()//1000000
            deque.append(read_dataset(config, verbose, rng))
            print("dataset loaded, ", time.time_ns()//1000000 - s)
            time.sleep(1)
        else:
            time.sleep(1)