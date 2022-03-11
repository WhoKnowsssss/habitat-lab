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
    

    path = config.trajectory_dir

    filenames = os.listdir(path)

    obss = []
    actions = None
    done_idxs = np.array([], dtype=np.int64)
    stepwise_returns = np.array([])

    while len(obss) < config.steps_per_load:
        buffer_num = rng.choice(np.arange(len(filenames)), 1)[0]

        file = os.path.join(path, filenames[buffer_num])
        dataset_raw = torch.load(file, map_location=torch.device('cpu'))

        obss += dataset_raw["obs"]
        actions = np.concatenate([actions, torch.stack(dataset_raw["actions"]).numpy()]) if actions is not None else torch.stack(dataset_raw["actions"]).numpy()
        stepwise_returns = np.concatenate([stepwise_returns, torch.cat(dataset_raw["rewards"]).numpy()])
        done_idxs = np.concatenate([done_idxs, np.argwhere(torch.cat(dataset_raw["masks"]).numpy() == False).squeeze()])

    # logger.info(
    #     "Success rate: {}.".format(

    #         sum((np.array(rlist) > 100)) / len(rlist)
    #     )
    # )
        
    # print('max rtg is %d' % max(rtg))

    # -- create timestep dataset
    # start_index = 0
    # timesteps = np.zeros(len(actions)+1, dtype=int)
    # for i in done_idxs:
    #     i = int(i)
    #     timesteps[start_index:i+1] = np.arange(i+1 - start_index)
    #     start_index = i+1
    # print('max timestep is %d' % max(timesteps))

    rtg, timesteps = _timesteps_rtg(done_idxs, stepwise_returns)

    return obss, actions, done_idxs, rtg, timesteps

@numba.jit(nopython=True, parallel=True)
def _timesteps_rtg(done_idxs, stepwise_returns):
    return_list = np.zeros_like(stepwise_returns)
    rtg = np.zeros_like(stepwise_returns)
    timesteps = np.zeros(len(stepwise_returns)+1, dtype=np.int64)
    start_index = np.concatenate((np.array([0], dtype=np.int64), done_idxs[:-1]))
    for i in numba.prange(len(done_idxs)):
        start = start_index[i]
        done = done_idxs[i]
        curr_traj_returns = stepwise_returns[start:done]

        for j in numba.prange(start-1, done-1): # start from i-1
            rtg[j] = np.sum(curr_traj_returns[j-start:i-done])
        
        return_list[i] = np.sum(curr_traj_returns)
        timesteps[start+1:done+1] = np.arange(done - start)
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
            time.sleep(2)
        else:
            time.sleep(10)