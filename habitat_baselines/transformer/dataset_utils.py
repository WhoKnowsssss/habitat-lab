import os, pickle, time
from this import d
from typing import Any, ClassVar, Dict, List, Tuple, Union, Optional
import itertools as its
from collections import deque

import numpy as np
import torch


from habitat import Config, logger 
        
def read_dataset(
    config: Config, 
    verbose: bool,
    rng: np.random.Generator
):        
    obss = []
    actions = []
    returns = [0]
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
                buffer_index += 1
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

        # debug
        l = np.array(done_idxs[1:]) - np.array(done_idxs[:-1])
        assert all(l <= 200), f"file:  {file}  dn:  {done_idxs}"

    actions = np.array(actions)
    returns = np.array(returns, dtype=np.float32)
    stepwise_returns = np.array(stepwise_returns)
    done_idxs = np.array(done_idxs)

    # -- create reward-to-go dataset
    start_index = 0
    rlist = []
    rtg = np.zeros_like(stepwise_returns)
    for i in done_idxs:
        i = int(i)
        curr_traj_returns = stepwise_returns[start_index:i]
        for j in range(i-1, start_index-1, -1): # start from i-1
            rtg_j = curr_traj_returns[j-start_index:i-start_index]
            rtg[j] = sum(rtg_j)
        
        rlist.append(sum(curr_traj_returns))
        start_index = i

    # logger.info(
    #     "Success rate: {}.".format(

    #         sum((np.array(rlist) > 100)) / len(rlist)
    #     )
    # )
        
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


    return obss, actions, done_idxs, rtg, timesteps

def producer(
    config: Config, 
    rng: np.random.Generator,
    deque: deque,
    verbose: bool,
):
    while True:
        if len(deque) < 1:
            deque.append(read_dataset(config, verbose, rng))
            time.sleep(2)
        else:
            time.sleep(10)