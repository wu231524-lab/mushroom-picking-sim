"""Static DAY13 motion envelopes used only for DAY14 preview validation.

No trajectory type, ROS action client, service client, or runtime command
exists in this module.  The values are conservative envelopes copied from the
verified DAY13 runners; loading a profile never grants execution authority.
"""

from dataclasses import asdict, dataclass
import math
from typing import Dict, Iterable, Mapping, Optional

from .pick_state import PickState


HOME_JOINTS = (
    -0.0000000018, -0.7850000008, 0.0000000003, -2.3560000004,
    -0.0000000007, 1.5709999985, 0.7850000049,
)

PREGRASP_JOINTS = (
    -0.003141, -0.029737, 0.018854, -1.530849,
    0.000564, 1.501278, 0.801142,
)


@dataclass(frozen=True)
class MotionProfile:
    state: PickState
    direction: str
    max_trajectories: int
    max_segment_m: Optional[float]
    max_cumulative_m: Optional[float]
    max_joint_delta_rad: float
    trajectory_duration_sec: float
    requires_attached: bool = False
    source: str = "DAY13 verified runner envelope"


PROFILES: Dict[PickState, MotionProfile] = {
    PickState.MOVE_PREGRASP: MotionProfile(
        PickState.MOVE_PREGRASP, "joint_space_home_to_pregrasp", 1,
        None, None, 0.830, 8.0,
    ),
    PickState.XY_ALIGN: MotionProfile(
        PickState.XY_ALIGN, "tcp_xy_only", 5,
        0.002, 0.010, 0.007, 5.0,
    ),
    PickState.Z_APPROACH: MotionProfile(
        PickState.Z_APPROACH, "world_negative_z", 8,
        0.020, 0.152, 0.080, 5.0,
    ),
    PickState.LIFT: MotionProfile(
        PickState.LIFT, "world_positive_z", 10,
        0.020, 0.192, 0.080, 5.0, True,
    ),
    PickState.TRANSPORT: MotionProfile(
        PickState.TRANSPORT, "world_horizontal_to_place", 14,
        0.020, 0.270, 0.050, 5.0, True,
    ),
    PickState.PLACE_DESCENT: MotionProfile(
        PickState.PLACE_DESCENT, "world_negative_z", 8,
        0.020, 0.152, 0.065, 5.0, True,
    ),
    PickState.VERTICAL_RETREAT: MotionProfile(
        PickState.VERTICAL_RETREAT, "world_positive_z", 8,
        0.020, 0.152, 0.065, 5.0,
    ),
    PickState.HORIZONTAL_RETURN: MotionProfile(
        PickState.HORIZONTAL_RETURN, "world_horizontal_to_pick", 14,
        0.020, 0.270, 0.050, 5.0,
    ),
    PickState.RECOVER_PREGRASP: MotionProfile(
        PickState.RECOVER_PREGRASP, "joint_space_to_pregrasp", 1,
        None, None, 0.035, 5.0,
    ),
    PickState.RETURN_HOME: MotionProfile(
        PickState.RETURN_HOME, "joint_space_pregrasp_to_home", 1,
        None, None, 0.830, 8.0,
    ),
}


class PreviewRejected(ValueError):
    """Raised when static preview metadata exceeds a DAY13 envelope."""


def load_static_profiles() -> Mapping[PickState, MotionProfile]:
    """Return an immutable-by-convention view of dry-run motion metadata."""
    return dict(PROFILES)


def profile_manifest() -> list:
    manifest = []
    for state in PROFILES:
        item = asdict(PROFILES[state])
        item["state"] = state.value
        manifest.append(item)
    return manifest


def validate_static_preview(
    state: PickState,
    segment_lengths_m: Iterable[float],
    joint_deltas_rad: Iterable[float],
    attached: bool,
) -> dict:
    """Validate numeric preview evidence without constructing a trajectory."""
    if state not in PROFILES:
        raise PreviewRejected("motion_profile_missing")

    profile = PROFILES[state]
    segments = tuple(float(value) for value in segment_lengths_m)
    joints = tuple(float(value) for value in joint_deltas_rad)

    if not segments or len(segments) != len(joints):
        raise PreviewRejected("preview_segment_count_invalid")
    if len(segments) > profile.max_trajectories:
        raise PreviewRejected("preview_trajectory_count_exceeded")
    if not all(math.isfinite(value) and value >= 0.0 for value in segments):
        raise PreviewRejected("preview_segment_invalid")
    if not all(math.isfinite(value) and value >= 0.0 for value in joints):
        raise PreviewRejected("preview_joint_delta_invalid")
    if profile.requires_attached and not attached:
        raise PreviewRejected("preview_requires_attached")
    if profile.max_segment_m is not None:
        if max(segments) > profile.max_segment_m + 1.0e-12:
            raise PreviewRejected("preview_segment_limit_exceeded")
    cumulative = sum(segments)
    if profile.max_cumulative_m is not None:
        if cumulative > profile.max_cumulative_m + 1.0e-12:
            raise PreviewRejected("preview_cumulative_limit_exceeded")
    if max(joints) > profile.max_joint_delta_rad + 1.0e-12:
        raise PreviewRejected("preview_joint_delta_limit_exceeded")

    return {
        "valid": True,
        "state": state.value,
        "preview_only": True,
        "trajectory_count": len(segments),
        "maximum_segment_m": max(segments),
        "cumulative_m": cumulative,
        "maximum_joint_delta_rad": max(joints),
        "execution_authorized": False,
    }
