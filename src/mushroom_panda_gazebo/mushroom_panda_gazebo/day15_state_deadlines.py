"""Steady-clock state deadlines for one DAY15 autonomous run."""

from dataclasses import dataclass
import math
from typing import Mapping, Optional

from .pick_state import MOTION_STATES, PICK_SEQUENCE, TERMINAL_STATES, PickState


class StateDeadlineExceeded(TimeoutError):
    """Raised when the active state exceeds its complete runtime budget."""


def trajectory_wall_timeout_sec(
    duration_sec: float,
    motion_wall_time_scale: float,
    margin_sec: float,
) -> float:
    """Bound a sim-time trajectory by a slower but finite wall-time budget."""
    values = (duration_sec, motion_wall_time_scale, margin_sec)
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        raise ValueError("trajectory_wall_timeout_invalid")
    if not 1.0 <= motion_wall_time_scale <= 10.0:
        raise ValueError("motion_wall_time_scale_out_of_range")
    return duration_sec * motion_wall_time_scale + margin_sec


STATE_TIMEOUT_SEC: Mapping[PickState, float] = {
    PickState.STARTUP_AUDIT: 15.0,
    PickState.WAIT_D435_TARGET: 35.0,
    PickState.SELECT_GLOBAL_TARGET: 3.0,
    PickState.VALIDATE_GLOBAL_TARGET: 3.0,
    PickState.MOVE_PREGRASP: 12.0,
    PickState.WAIT_ROBOT_SETTLED: 5.0,
    # YOLO service callbacks and the first valid close-range frame can take
    # tens of wall seconds on the 4-vCPU DAY19 Gazebo VM.
    PickState.START_D405_CHAIN: 90.0,
    # The proposal node itself allows up to 60 wall seconds to gather its
    # required observations; leave bounded delivery margin for ROS callbacks.
    PickState.WAIT_D405_SPHERE: 70.0,
    PickState.VERIFY_TARGET_HANDOFF: 3.0,
    PickState.XY_ALIGN: 120.0,
    PickState.VERIFY_XY_GATE: 5.0,
    PickState.Z_APPROACH: 60.0,
    PickState.VERIFY_VACUUM_CONTACT: 5.0,
    PickState.ATTACH: 5.0,
    PickState.VERIFY_ATTACHED: 5.0,
    PickState.LIFT: 60.0,
    PickState.TRANSPORT: 105.0,
    PickState.PLACE_DESCENT: 60.0,
    PickState.VERIFY_PLACE_POSE: 5.0,
    PickState.DETACH: 5.0,
    PickState.VERIFY_DETACHED: 5.0,
    PickState.VERTICAL_RETREAT: 60.0,
    PickState.HORIZONTAL_RETURN: 105.0,
    PickState.RECOVER_PREGRASP: 10.0,
    PickState.RETURN_HOME: 12.0,
    PickState.FINAL_AUDIT: 10.0,
}


@dataclass(frozen=True)
class StateDeadlineReport:
    """Current deadline evidence for reports and tests."""

    state: str
    elapsed_sec: float
    timeout_sec: Optional[float]
    remaining_sec: Optional[float]


class StateDeadlineTracker:
    """Track state transitions against caller-provided steady nanoseconds."""

    def __init__(self, motion_wall_time_scale: float = 1.0):
        expected = set(PICK_SEQUENCE).difference({PickState.DONE})
        if set(STATE_TIMEOUT_SEC) != expected:
            raise ValueError("state_deadline_coverage_mismatch")
        if (
            not math.isfinite(motion_wall_time_scale)
            or not 1.0 <= motion_wall_time_scale <= 10.0
        ):
            raise ValueError("motion_wall_time_scale_out_of_range")
        self.motion_wall_time_scale = motion_wall_time_scale
        self.state = PickState.IDLE
        self.entered_ns: Optional[int] = None

    def enter(self, state: PickState, now_ns: int) -> None:
        state = PickState(state)
        if state == self.state and self.entered_ns is not None:
            return
        if int(now_ns) < 0:
            raise ValueError("state_deadline_time_invalid")
        self.state = state
        self.entered_ns = int(now_ns)

    def check(self, state: PickState, now_ns: int) -> StateDeadlineReport:
        state = PickState(state)
        if state != self.state or self.entered_ns is None:
            raise ValueError("state_deadline_not_synchronized")
        elapsed = (int(now_ns) - self.entered_ns) / 1_000_000_000.0
        if elapsed < 0.0:
            raise ValueError("state_deadline_clock_reversed")
        if state == PickState.IDLE or state in TERMINAL_STATES:
            return StateDeadlineReport(state.value, elapsed, None, None)
        timeout = STATE_TIMEOUT_SEC[state]
        if state in MOTION_STATES:
            timeout *= self.motion_wall_time_scale
        remaining = max(0.0, timeout - elapsed)
        if elapsed > timeout:
            raise StateDeadlineExceeded(
                f"state_timeout:{state.value}:{elapsed:.3f}>{timeout:.3f}"
            )
        return StateDeadlineReport(
            state.value,
            elapsed,
            timeout,
            remaining,
        )
