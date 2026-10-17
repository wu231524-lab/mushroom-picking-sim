"""Regression checks for the asynchronous D405 startup gate."""

import unittest

from mushroom_panda_gazebo.day15_runtime_evidence import (
    d405_observation_start_ready,
)
from mushroom_panda_gazebo.day15_state_deadlines import STATE_TIMEOUT_SEC
from mushroom_panda_gazebo.pick_state import PickState


class D405StartTest(unittest.TestCase):
    def setUp(self):
        self.alignment = {
            "valid": True,
            "session_generation": 1,
            "target_tracking": {"lock_id": "d405-lock-1"},
        }

    def test_waits_for_new_valid_alignment_after_detector_switch(self):
        ready = d405_observation_start_ready
        self.assertFalse(ready(self.alignment, 0.1, 3, 3))
        self.assertFalse(ready(self.alignment, 2.1, 4, 3))
        self.assertFalse(ready({**self.alignment, "valid": False}, 0.1, 4, 3))
        self.assertFalse(ready({"valid": True}, 0.1, 4, 3))
        self.assertTrue(ready(self.alignment, 0.1, 4, 3))

    def test_deadlines_cover_d405_startup_and_observation(self):
        self.assertEqual(STATE_TIMEOUT_SEC[PickState.START_D405_CHAIN], 90.0)
        self.assertEqual(STATE_TIMEOUT_SEC[PickState.WAIT_D405_SPHERE], 70.0)


if __name__ == "__main__":
    unittest.main()
