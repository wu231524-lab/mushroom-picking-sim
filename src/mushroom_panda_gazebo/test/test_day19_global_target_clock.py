"""Regression guard for the physical run's target receive-age clock."""

import ast
from pathlib import Path
import unittest


class GlobalTargetClockTest(unittest.TestCase):
    def test_physical_target_receive_age_uses_steady_clock(self):
        source = (
            Path(__file__).parents[1]
            / "mushroom_panda_gazebo"
            / "day15_autonomous_pick.py"
        ).read_text(encoding="utf-8")
        module = ast.parse(source)
        controller = next(
            item for item in module.body
            if isinstance(item, ast.ClassDef)
            and item.name == "Day15AutonomousPick"
        )
        method = next(
            item for item in controller.body
            if isinstance(item, ast.FunctionDef)
            and item.name == "accept_global_target_evidence"
        )
        validator = next(
            item for item in ast.walk(method)
            if isinstance(item, ast.Call)
            and isinstance(item.func, ast.Name)
            and item.func.id == "validate_fresh_global_target"
        )
        receive_age = ast.unparse(validator.args[2])
        self.assertIn("self.steady_clock.now().nanoseconds", receive_age)
        self.assertNotIn("self.get_clock()", receive_age)


if __name__ == "__main__":
    unittest.main()
