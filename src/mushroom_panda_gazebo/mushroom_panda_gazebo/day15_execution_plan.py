"""Single-mushroom autonomous execution plan built from DAY13 results.

This module contains no ROS imports and sends no commands.  It states which
runtime operation belongs to each physical state so the full path can be
tested before a live adapter is allowed to create command interfaces.
"""

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Mapping, Optional

from .day14_motion_profiles import PROFILES
from .pick_state import PHYSICAL_OPERATION_STATES, PickState


class CommandKind(str, Enum):
    TRAJECTORY = "trajectory"
    ATTACH = "attach"
    DETACH = "detach"


@dataclass(frozen=True)
class ExecutionStage:
    state: PickState
    command_kind: CommandKind
    maximum_commands: int
    day13_source: str
    completion_gate: str
    motion_profile_state: Optional[PickState] = None


EXECUTION_STAGES = (
    ExecutionStage(
        PickState.MOVE_PREGRASP,
        CommandKind.TRAJECTORY,
        1,
        "DAY13 HOME to PREGRASP fixed trajectory",
        "pregrasp_joint_and_tcp_reached",
        PickState.MOVE_PREGRASP,
    ),
    ExecutionStage(
        PickState.XY_ALIGN,
        CommandKind.TRAJECTORY,
        5,
        "DAY13 ROI-sphere stop-observe XY loop",
        "sphere_planar_error_within_gate",
        PickState.XY_ALIGN,
    ),
    ExecutionStage(
        PickState.Z_APPROACH,
        CommandKind.TRAJECTORY,
        8,
        "DAY13 segmented vacuum Z approach",
        "vacuum_contact_valid",
        PickState.Z_APPROACH,
    ),
    ExecutionStage(
        PickState.ATTACH,
        CommandKind.ATTACH,
        1,
        "DAY13 contact-gated SetBool attach",
        "mushroom_3_confirmed_attached",
    ),
    ExecutionStage(
        PickState.LIFT,
        CommandKind.TRAJECTORY,
        8,
        "DAY13 segmented vertical lift",
        "lift_pose_and_rigid_following_valid",
        PickState.LIFT,
    ),
    ExecutionStage(
        PickState.TRANSPORT,
        CommandKind.TRAJECTORY,
        14,
        "DAY13 high horizontal transport",
        "transport_pose_and_rigid_following_valid",
        PickState.TRANSPORT,
    ),
    ExecutionStage(
        PickState.PLACE_DESCENT,
        CommandKind.TRAJECTORY,
        8,
        "DAY13 segmented place descent",
        "place_pose_and_support_height_valid",
        PickState.PLACE_DESCENT,
    ),
    ExecutionStage(
        PickState.DETACH,
        CommandKind.DETACH,
        1,
        "DAY13 SetBool detach at verified place pose",
        "mushroom_3_confirmed_detached",
    ),
    ExecutionStage(
        PickState.VERTICAL_RETREAT,
        CommandKind.TRAJECTORY,
        8,
        "DAY13 post-place vertical retreat",
        "retreat_height_reached",
        PickState.VERTICAL_RETREAT,
    ),
    ExecutionStage(
        PickState.HORIZONTAL_RETURN,
        CommandKind.TRAJECTORY,
        14,
        "DAY13 post-place horizontal return",
        "pick_high_pose_reached",
        PickState.HORIZONTAL_RETURN,
    ),
    ExecutionStage(
        PickState.RECOVER_PREGRASP,
        CommandKind.TRAJECTORY,
        1,
        "DAY13 post-XY pose to standard PREGRASP",
        "standard_pregrasp_recovered",
        PickState.RECOVER_PREGRASP,
    ),
    ExecutionStage(
        PickState.RETURN_HOME,
        CommandKind.TRAJECTORY,
        1,
        "DAY13 PREGRASP to HOME fixed trajectory",
        "home_joint_and_tcp_reached",
        PickState.RETURN_HOME,
    ),
)


STAGE_BY_STATE: Mapping[PickState, ExecutionStage] = {
    stage.state: stage for stage in EXECUTION_STAGES
}


def validate_execution_plan() -> dict:
    """Check complete physical-state coverage and DAY13 profile agreement."""
    states = [stage.state for stage in EXECUTION_STAGES]
    if len(states) != len(set(states)):
        raise ValueError("duplicate_execution_stage")
    if set(states) != set(PHYSICAL_OPERATION_STATES):
        raise ValueError("physical_state_coverage_mismatch")

    for stage in EXECUTION_STAGES:
        if stage.maximum_commands < 1:
            raise ValueError("stage_command_budget_invalid")
        if stage.command_kind == CommandKind.TRAJECTORY:
            if stage.motion_profile_state != stage.state:
                raise ValueError("motion_profile_state_mismatch")
            profile = PROFILES.get(stage.state)
            if profile is None:
                raise ValueError("motion_profile_missing")
            if stage.maximum_commands > profile.max_trajectories:
                raise ValueError("command_budget_exceeds_day13_profile")
        elif stage.motion_profile_state is not None:
            raise ValueError("non_motion_stage_has_motion_profile")

    trajectory_budget = sum(
        stage.maximum_commands
        for stage in EXECUTION_STAGES
        if stage.command_kind == CommandKind.TRAJECTORY
    )
    attachment_budget = sum(
        stage.maximum_commands
        for stage in EXECUTION_STAGES
        if stage.command_kind in {CommandKind.ATTACH, CommandKind.DETACH}
    )
    return {
        "valid": True,
        "single_target": "mushroom_3",
        "trajectory_command_budget": trajectory_budget,
        "attachment_command_budget": attachment_budget,
        "gripper_command_budget": 0,
        "automatic_retry_budget": 0,
        "stages": [
            {
                **asdict(stage),
                "state": stage.state.value,
                "command_kind": stage.command_kind.value,
                "motion_profile_state": (
                    stage.motion_profile_state.value
                    if stage.motion_profile_state is not None
                    else None
                ),
            }
            for stage in EXECUTION_STAGES
        ],
    }
