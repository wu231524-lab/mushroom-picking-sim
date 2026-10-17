"""DAY13-bounded XY alignment exit and solver envelope."""

import unittest

from mushroom_panda_gazebo.day15_execution_plan import STAGE_BY_STATE
from mushroom_panda_gazebo.day15_runtime_evidence import (
    EvidenceRejected,
    validate_day13_xy_budget_gate,
    validate_robot_settled,
)
from mushroom_panda_gazebo.day15_xy_kinematics import XYKinematicLimits
from mushroom_panda_gazebo.day15_verified_trajectories import POST_XY_JOINTS
from mushroom_panda_gazebo.day14_motion_profiles import PROFILES
from mushroom_panda_gazebo.pick_state import PickState


class XYBudgetGateTest(unittest.TestCase):
    def setUp(self):
        self.xy = {"planar_error_m": 0.003877730494, "proposal_id": "fresh-xy"}
        self.vacuum = {
            "entity_name": "mushroom_3",
            "model_state_age_sec": 0.1,
            "planar_error_m": 0.002126,
            "axis_error_rad": 0.000142340321,
            "surface_gap_m": 0.154998650099,
        }

    def test_day13_geometry_allows_z_after_five_steps(self):
        report = validate_day13_xy_budget_gate(self.xy, self.vacuum)
        self.assertTrue(report["valid"])
        self.assertEqual(report["vacuum_planar_error_m"], 0.002126)

    def test_bad_independent_geometry_is_rejected(self):
        for change in (
            {"planar_error_m": 0.0031},
            {"entity_name": "mushroom_2"},
            {"model_state_age_sec": 0.6},
            {"axis_error_rad": 0.18},
            {"surface_gap_m": 0.0},
            {"surface_gap_m": 0.149},
            {"surface_gap_m": 0.161},
        ):
            with self.subTest(change=change):
                with self.assertRaises(EvidenceRejected):
                    validate_day13_xy_budget_gate(
                        self.xy, {**self.vacuum, **change}
                    )

    def test_visual_residual_remains_bounded(self):
        with self.assertRaisesRegex(EvidenceRejected, "xy_sphere_planar_exceeded"):
            validate_day13_xy_budget_gate(
                {**self.xy, "planar_error_m": 0.0051}, self.vacuum
            )

    def test_early_visual_deadband_still_requires_independent_geometry(self):
        aligned = {**self.xy, "planar_error_m": 0.001}
        self.assertTrue(validate_day13_xy_budget_gate(aligned, self.vacuum)["valid"])
        with self.assertRaisesRegex(EvidenceRejected, "xy_vacuum_planar_exceeded"):
            validate_day13_xy_budget_gate(
                aligned, {**self.vacuum, "planar_error_m": 0.0031}
            )

    def test_verified_z_route_start_rejects_joint_drift(self):
        observed = (
            -0.009497620947783503, -0.05296663001017421,
            0.011898133062761218, -1.5582455775820687,
            0.0006648873190240323, 1.5054210285590042,
            0.7878303651553278,
        )
        settled = validate_robot_settled(
            observed, (0.0,) * 7, POST_XY_JOINTS, 0.02
        )
        self.assertLess(settled["maximum_joint_error_rad"], 0.00002)
        drifted = (observed[0] + 0.0011,) + observed[1:]
        with self.assertRaisesRegex(EvidenceRejected, "robot_joint_target_not_reached"):
            validate_robot_settled(
                drifted, (0.0,) * 7, POST_XY_JOINTS, 0.02
            )

    def test_xy_motion_stays_inside_day13_envelope(self):
        profile = PROFILES[PickState.XY_ALIGN]
        self.assertEqual(XYKinematicLimits().max_joint_delta_rad, 0.007)
        self.assertEqual(profile.max_joint_delta_rad, 0.007)
        self.assertEqual(STAGE_BY_STATE[PickState.XY_ALIGN].maximum_commands, 5)
        self.assertEqual(profile.max_cumulative_m, 0.010)


if __name__ == "__main__":
    unittest.main()
