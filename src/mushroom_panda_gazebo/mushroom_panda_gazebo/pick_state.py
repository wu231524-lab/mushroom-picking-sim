"""DAY14 unified picker state names and safety classifications.

This module is deliberately ROS-free so transition logic can be tested without
a running simulator.  The two dry-run terminal states are outside the future
live physical sequence and make the reason for stopping unambiguous.
"""

from enum import Enum


class PickState(str, Enum):
    IDLE = "IDLE"
    STARTUP_AUDIT = "STARTUP_AUDIT"
    WAIT_D435_TARGET = "WAIT_D435_TARGET"
    SELECT_GLOBAL_TARGET = "SELECT_GLOBAL_TARGET"
    VALIDATE_GLOBAL_TARGET = "VALIDATE_GLOBAL_TARGET"
    MOVE_PREGRASP = "MOVE_PREGRASP"
    WAIT_ROBOT_SETTLED = "WAIT_ROBOT_SETTLED"
    START_D405_CHAIN = "START_D405_CHAIN"
    WAIT_D405_SPHERE = "WAIT_D405_SPHERE"
    VERIFY_TARGET_HANDOFF = "VERIFY_TARGET_HANDOFF"
    XY_ALIGN = "XY_ALIGN"
    VERIFY_XY_GATE = "VERIFY_XY_GATE"
    Z_APPROACH = "Z_APPROACH"
    VERIFY_VACUUM_CONTACT = "VERIFY_VACUUM_CONTACT"
    ATTACH = "ATTACH"
    VERIFY_ATTACHED = "VERIFY_ATTACHED"
    LIFT = "LIFT"
    TRANSPORT = "TRANSPORT"
    PLACE_DESCENT = "PLACE_DESCENT"
    VERIFY_PLACE_POSE = "VERIFY_PLACE_POSE"
    DETACH = "DETACH"
    VERIFY_DETACHED = "VERIFY_DETACHED"
    VERTICAL_RETREAT = "VERTICAL_RETREAT"
    HORIZONTAL_RETURN = "HORIZONTAL_RETURN"
    RECOVER_PREGRASP = "RECOVER_PREGRASP"
    RETURN_HOME = "RETURN_HOME"
    FINAL_AUDIT = "FINAL_AUDIT"
    DONE = "DONE"
    DRY_RUN_COMPLETE = "DRY_RUN_COMPLETE"
    SAFE_STOP = "SAFE_STOP"


PHYSICAL_OPERATION_STATES = frozenset({
    PickState.MOVE_PREGRASP,
    PickState.XY_ALIGN,
    PickState.Z_APPROACH,
    PickState.ATTACH,
    PickState.LIFT,
    PickState.TRANSPORT,
    PickState.PLACE_DESCENT,
    PickState.DETACH,
    PickState.VERTICAL_RETREAT,
    PickState.HORIZONTAL_RETURN,
    PickState.RECOVER_PREGRASP,
    PickState.RETURN_HOME,
})

MOTION_STATES = PHYSICAL_OPERATION_STATES.difference({
    PickState.ATTACH,
    PickState.DETACH,
})

TERMINAL_STATES = frozenset({
    PickState.DONE,
    PickState.DRY_RUN_COMPLETE,
    PickState.SAFE_STOP,
})


PICK_SEQUENCE = (
    PickState.STARTUP_AUDIT,
    PickState.WAIT_D435_TARGET,
    PickState.SELECT_GLOBAL_TARGET,
    PickState.VALIDATE_GLOBAL_TARGET,
    PickState.MOVE_PREGRASP,
    PickState.WAIT_ROBOT_SETTLED,
    PickState.START_D405_CHAIN,
    PickState.WAIT_D405_SPHERE,
    PickState.VERIFY_TARGET_HANDOFF,
    PickState.XY_ALIGN,
    PickState.VERIFY_XY_GATE,
    PickState.Z_APPROACH,
    PickState.VERIFY_VACUUM_CONTACT,
    PickState.ATTACH,
    PickState.VERIFY_ATTACHED,
    PickState.LIFT,
    PickState.TRANSPORT,
    PickState.PLACE_DESCENT,
    PickState.VERIFY_PLACE_POSE,
    PickState.DETACH,
    PickState.VERIFY_DETACHED,
    PickState.VERTICAL_RETREAT,
    PickState.HORIZONTAL_RETURN,
    PickState.RECOVER_PREGRASP,
    PickState.RETURN_HOME,
    PickState.FINAL_AUDIT,
    PickState.DONE,
)
