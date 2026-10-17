"""Wall-time safety budgets for slow Gazebo simulation."""

import unittest

from mushroom_panda_gazebo.day15_state_deadlines import (
    StateDeadlineExceeded,
    StateDeadlineTracker,
    trajectory_wall_timeout_sec,
)
from mushroom_panda_gazebo.pick_state import PickState


class SimTimingTest(unittest.TestCase):
    def test_pregrasp_action_and_state_survive_observed_slow_sim(self):
        # DAY19 VM advanced about 3.2 simulated seconds in 7.68 wall seconds.
        self.assertEqual(trajectory_wall_timeout_sec(8.0, 4.0, 2.0), 34.0)
        tracker = StateDeadlineTracker(motion_wall_time_scale=4.0)
        tracker.enter(PickState.MOVE_PREGRASP, 0)
        self.assertEqual(tracker.check(PickState.MOVE_PREGRASP, 20_000_000_000).timeout_sec, 48.0)
        with self.assertRaises(StateDeadlineExceeded):
            tracker.check(PickState.MOVE_PREGRASP, 49_000_000_000)

    def test_non_motion_gate_keeps_original_deadline(self):
        tracker = StateDeadlineTracker(motion_wall_time_scale=4.0)
        tracker.enter(PickState.WAIT_D435_TARGET, 0)
        self.assertEqual(
            tracker.check(PickState.WAIT_D435_TARGET, 0).timeout_sec,
            35.0,
        )

    def test_scale_must_remain_bounded(self):
        with self.assertRaises(ValueError):
            StateDeadlineTracker(motion_wall_time_scale=11.0)
        with self.assertRaises(ValueError):
            trajectory_wall_timeout_sec(8.0, 0.0, 2.0)


if __name__ == "__main__":
    unittest.main()
