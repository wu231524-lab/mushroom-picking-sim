"""Regression tests for pairing a slow D405 proposal with its source frame."""

import unittest

from mushroom_panda_gazebo.day15_runtime_evidence import (
    EvidenceRejected,
    matching_d405_alignment,
    validate_d405_chain,
)
from mushroom_panda_gazebo.target_handoff import GlobalTargetLock


def stamp(seconds):
    whole = int(seconds)
    return {"sec": whole, "nanosec": round((seconds - whole) * 1e9)}


class D405HandoffPairTest(unittest.TestCase):
    def setUp(self):
        self.lock = GlobalTargetLock(
            run_id="test", handoff_id="handoff-test", entity_name="mushroom_3",
            lock_generation=1, stamp_ns=263_000_000_000,
            compensated_world=(0.3, -0.12, 0.325),
            expected_cap_center_world=(0.3, -0.12, 0.325),
        )
        self.alignment = {
            "valid": True, "target_frame": "fer_hand_tcp",
            "detection_stamp": stamp(267.183),
            "upstream_sync": {"valid": True, "delta_sec": 0.02},
            "target_d405": {
                "depth": 0.2, "sphere_radius_m": 0.07,
                "sphere_fit_rms_m": 0.0005, "sphere_inlier_count": 2000,
                "center_distance_px": 20.0,
            },
            "target_tracking": {
                "enabled": True, "lock_id": "lock-1",
                "tracking_distance_m": 0.001,
            },
        }
        self.proposal = {
            "valid": True, "observation_active": False,
            "source_frame": "fer_hand_tcp", "single_step_only": True,
            "execute_allowed": False, "proposal_id": "proposal-1",
            "generated_stamp": stamp(269.9),
            "stamp_first": stamp(263.739), "stamp_last": stamp(267.183),
            "observation_count": 2, "required_observations": 2,
            "upstream_sync_evidence": [
                {"valid": True, "delta_sec": 0.02},
                {"valid": True, "delta_sec": 0.02},
            ],
            "maximum_spread": 0.0005, "max_step_m": 0.002,
            "deadband_m": 0.002, "reason": "stable_dry_run_proposal",
            "mean_alignment_error": {
                "x": 0.003, "y": 0.0, "planar": 0.003,
            },
            "proposed_delta_tcp": {"x": 0.002, "y": 0.0, "z": 0.0},
        }

    def test_uses_exact_historical_observation_not_latest(self):
        latest = {**self.alignment, "detection_stamp": stamp(270.2)}
        history = [(1, self.alignment), (2, latest)]
        self.assertIs(
            matching_d405_alignment(history, self.proposal, 0),
            self.alignment,
        )
        with self.assertRaisesRegex(EvidenceRejected, "not_in_history"):
            matching_d405_alignment(history, self.proposal, 1)

    def test_slow_two_frame_observation_has_bounded_handoff_age(self):
        report = validate_d405_chain(
            self.lock, self.alignment, self.proposal,
            {"x": 0.3, "y": -0.12, "z": 0.325}, 270_100_000_000,
        )
        self.assertAlmostEqual(report["d405_age_sec"], 2.917)
        with self.assertRaisesRegex(EvidenceRejected, "d405_stale"):
            validate_d405_chain(
                self.lock, self.alignment, self.proposal,
                {"x": 0.3, "y": -0.12, "z": 0.325}, 271_300_000_000,
            )


if __name__ == "__main__":
    unittest.main()
