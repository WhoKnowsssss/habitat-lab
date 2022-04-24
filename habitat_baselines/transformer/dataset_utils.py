import os, pickle, time
from this import d
from typing import Any, ClassVar, Dict, List, Tuple, Union, Optional
import itertools as its
from collections import deque

import numpy as np
import torch
import numba


from habitat import Config, logger 
        

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

            # stepwise_idx = np.argwhere(np.all(temp_actions[:,8:10] == 0, axis=-1) & temp_dones == True).squeeze()

            # temp_obs = np.delete(temp_obs, stepwise_idx, 0)
            # temp_actions = np.delete(temp_actions, stepwise_idx, 0)
            # temp_stepwise_returns = np.delete(temp_stepwise_returns, stepwise_idx, 0)
            # temp_dones = np.delete(temp_dones, stepwise_idx, 0)

            temp_done_idxs = np.argwhere(temp_dones == False).squeeze() + 1

            idx = np.nonzero(temp_done_idxs[1:] - temp_done_idxs[:-1] < 30)[0]
            if len(idx) > 0:

                stepwise_idx = np.concatenate([np.arange(temp_done_idxs[:-1][i] , temp_done_idxs[1:][i]) for i in idx])

                temp_obs = np.delete(temp_obs, stepwise_idx, 0)
                temp_actions = np.delete(temp_actions, stepwise_idx, 0)
                temp_stepwise_returns = np.delete(temp_stepwise_returns, stepwise_idx, 0)
                temp_dones = np.delete(temp_dones, stepwise_idx, 0)

                

            temp_done_idxs = np.argwhere(temp_dones == False).squeeze() + 1

            # idx = np.nonzero(temp_stepwise_returns[temp_done_idxs-1] < 90)[0]
            # if len(idx) > 0:
            #     stepwise_idx = np.concatenate([np.arange(temp_done_idxs[i-1] if i > 0 else 0, temp_done_idxs[i]) for i in idx])

            #     temp_obs = np.delete(temp_obs, stepwise_idx, 0)
            #     temp_actions = np.delete(temp_actions, stepwise_idx, 0)
            #     temp_stepwise_returns = np.delete(temp_stepwise_returns, stepwise_idx, 0)
            #     temp_dones = np.delete(temp_dones, stepwise_idx, 0)

            # temp_done_idxs = np.argwhere(temp_dones == False).squeeze() + 1

            l = temp_done_idxs[1:] - temp_done_idxs[:-1]
            # debug
            assert all(l <= 500), f"Length too long: file:  {file}  dn:  {temp_done_idxs}"
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
        if len(deque) < 2:
            deque.append(read_dataset(config, verbose, rng))
            time.sleep(3)
        else:
            time.sleep(10)