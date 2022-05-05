#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from habitat.core.dataset import Episode
from habitat.core.registry import registry
from habitat.tasks.rearrange.sub_tasks.pick_task import RearrangePickTaskV1


@registry.register_task(name="RearrangePlaceTask-v0")
class RearrangePlaceTaskV1(RearrangePickTaskV1):
    def _get_targ_pos(self, sim):
        return sim.get_targets()[1]

    def _should_prevent_grip(self, action_args):
        # Never allow regrasping
        return (
            not self._sim.grasp_mgr.is_grasped
            and action_args.get("grip_action", None) is not None
            and action_args["grip_action"] >= 0
        )

    def get_receptacle_info(self, episode, sel_idx):
        return episode.goal_receptacles[sel_idx]

    def reset(self, episode: Episode):
        sim = self._sim
        super().reset(episode, fetch_observations=False)

        abs_obj_idx = sim.scene_obj_ids[self.abs_targ_idx]

        sim.grasp_mgr.snap_to_obj(abs_obj_idx, force=True)
        sim.internal_step(-1)

        self.was_prev_holding = self.targ_idx

        sim.internal_step(-1)
        return self._get_observations(episode)
