"""ROS-free command sequencing for one DAY15 autonomous pick cycle."""

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Dict, Optional

from .day15_execution_plan import (
    CommandKind,
    EXECUTION_STAGES,
    STAGE_BY_STATE,
)
from .day15_verified_trajectories import FIXED_ROUTES, JointVector
from .pick_state import PickState


class CommandRejected(RuntimeError):
    """Raised when a runtime command violates the single-run contract."""


@dataclass(frozen=True)
class RuntimeCommand:
    command_id: str
    run_id: str
    state: PickState
    command_kind: CommandKind
    state_ordinal: int
    target_joints: Optional[JointVector] = None
    trajectory_duration_sec: Optional[float] = None
    attachment_value: Optional[bool] = None
    dynamic_xy: bool = False


class CommandLedger:
    """Allow one active command and latch the first runtime failure."""

    def __init__(self, run_id: str, execution_authorized: bool = False):
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("run_id_invalid")
        self.run_id = run_id
        self.execution_authorized = bool(execution_authorized)
        self.active_state: Optional[PickState] = None
        self.active_command: Optional[RuntimeCommand] = None
        self.counts: Counter = Counter()
        self.completed_command_ids = set()
        self.failure_reason: Optional[str] = None
        self.closed = False
        self.events = []

    def _reject_if_unavailable(self) -> None:
        if not self.execution_authorized:
            raise CommandRejected("execution_not_authorized")
        if self.failure_reason is not None:
            raise CommandRejected("runtime_failure_latched")
        if self.closed:
            raise CommandRejected("single_run_already_closed")

    def enter_state(self, state: PickState) -> None:
        self._reject_if_unavailable()
        state = PickState(state)
        if state not in STAGE_BY_STATE:
            raise CommandRejected("state_has_no_physical_command")
        if self.active_command is not None:
            raise CommandRejected("command_still_active")
        if self.active_state is not None:
            raise CommandRejected("previous_state_not_closed")
        self.active_state = state
        self.events.append({"event": "state_entered", "state": state.value})

    def _issue(
        self,
        command_kind: CommandKind,
        target_joints: Optional[JointVector] = None,
        trajectory_duration_sec: Optional[float] = None,
        attachment_value: Optional[bool] = None,
        dynamic_xy: bool = False,
    ) -> RuntimeCommand:
        self._reject_if_unavailable()
        if self.active_state is None:
            raise CommandRejected("physical_state_not_entered")
        if self.active_command is not None:
            raise CommandRejected("command_already_active")

        stage = STAGE_BY_STATE[self.active_state]
        if command_kind != stage.command_kind:
            raise CommandRejected("command_kind_mismatch")
        ordinal = self.counts[self.active_state] + 1
        if ordinal > stage.maximum_commands:
            raise CommandRejected("state_command_budget_exceeded")

        command = RuntimeCommand(
            command_id=(
                f"{self.run_id}:{self.active_state.value}:{ordinal}"
            ),
            run_id=self.run_id,
            state=self.active_state,
            command_kind=command_kind,
            state_ordinal=ordinal,
            target_joints=target_joints,
            trajectory_duration_sec=trajectory_duration_sec,
            attachment_value=attachment_value,
            dynamic_xy=dynamic_xy,
        )
        self.active_command = command
        self.counts[self.active_state] += 1
        self.events.append({
            "event": "command_issued",
            **self.serialize(command),
        })
        return command

    def issue_next_fixed_trajectory(self) -> RuntimeCommand:
        if self.active_command is not None:
            raise CommandRejected("command_already_active")
        if self.active_state not in FIXED_ROUTES:
            raise CommandRejected("fixed_route_not_available")
        route = FIXED_ROUTES[self.active_state]
        waypoint_index = self.counts[self.active_state]
        if waypoint_index >= len(route.waypoints):
            raise CommandRejected("fixed_route_exhausted")
        return self._issue(
            CommandKind.TRAJECTORY,
            target_joints=route.waypoints[waypoint_index],
            trajectory_duration_sec=route.duration_sec,
        )

    def issue_dynamic_xy(
        self,
        target_joints: JointVector,
        trajectory_duration_sec: float = 5.0,
    ) -> RuntimeCommand:
        if self.active_state != PickState.XY_ALIGN:
            raise CommandRejected("dynamic_xy_wrong_state")
        if len(target_joints) != 7:
            raise CommandRejected("dynamic_xy_joint_dimension_invalid")
        return self._issue(
            CommandKind.TRAJECTORY,
            target_joints=tuple(float(value) for value in target_joints),
            trajectory_duration_sec=float(trajectory_duration_sec),
            dynamic_xy=True,
        )

    def issue_attachment(self) -> RuntimeCommand:
        if self.active_state == PickState.ATTACH:
            value = True
            kind = CommandKind.ATTACH
        elif self.active_state == PickState.DETACH:
            value = False
            kind = CommandKind.DETACH
        else:
            raise CommandRejected("attachment_wrong_state")
        return self._issue(kind, attachment_value=value)

    def complete(self, command_id: str, succeeded: bool, reason: str) -> None:
        self._reject_if_unavailable()
        if self.active_command is None:
            raise CommandRejected("no_active_command")
        if command_id != self.active_command.command_id:
            raise CommandRejected("command_result_id_mismatch")
        if command_id in self.completed_command_ids:
            raise CommandRejected("command_result_already_consumed")

        command = self.active_command
        self.completed_command_ids.add(command_id)
        self.active_command = None
        self.events.append({
            "event": "command_completed" if succeeded else "command_failed",
            "command_id": command_id,
            "state": command.state.value,
            "reason": str(reason),
        })
        if not succeeded:
            self.failure_reason = str(reason) or "command_failed"

    def close_state(self) -> None:
        self._reject_if_unavailable()
        if self.active_state is None:
            raise CommandRejected("physical_state_not_entered")
        if self.active_command is not None:
            raise CommandRejected("command_still_active")
        state = self.active_state
        self.events.append({"event": "state_completed", "state": state.value})
        self.active_state = None

    def fail(self, reason: str) -> None:
        if self.failure_reason is None:
            self.failure_reason = str(reason) or "runtime_failed"
            self.events.append({
                "event": "runtime_failed",
                "reason": self.failure_reason,
            })

    def finish_run(self) -> None:
        self._reject_if_unavailable()
        if self.active_state is not None or self.active_command is not None:
            raise CommandRejected("runtime_not_idle")
        self.closed = True
        self.events.append({"event": "single_run_closed"})

    @staticmethod
    def serialize(command: RuntimeCommand) -> dict:
        result = asdict(command)
        result["state"] = command.state.value
        result["command_kind"] = command.command_kind.value
        return result

    def report(self) -> dict:
        totals: Dict[str, int] = {
            state.value: self.counts[state]
            for state in self.counts
        }
        return {
            "run_id": self.run_id,
            "execution_authorized": self.execution_authorized,
            "active_state": (
                self.active_state.value if self.active_state else None
            ),
            "active_command": (
                self.serialize(self.active_command)
                if self.active_command is not None
                else None
            ),
            "completed_command_count": len(self.completed_command_ids),
            "command_counts_by_state": totals,
            "failure_reason": self.failure_reason,
            "closed": self.closed,
            "events": list(self.events),
        }


def simulate_complete_command_cycle(
    run_id: str = "day15-no-motion-replay",
) -> dict:
    """Exercise all command sequencing in memory without ROS or I/O."""
    ledger = CommandLedger(run_id, execution_authorized=True)
    for stage in EXECUTION_STAGES:
        ledger.enter_state(stage.state)
        for _ in range(stage.maximum_commands):
            if stage.state == PickState.XY_ALIGN:
                command = ledger.issue_dynamic_xy((0.0,) * 7)
            elif stage.command_kind == CommandKind.TRAJECTORY:
                command = ledger.issue_next_fixed_trajectory()
            else:
                command = ledger.issue_attachment()
            ledger.complete(
                command.command_id,
                succeeded=True,
                reason="simulated_result_only",
            )
        ledger.close_state()
    ledger.finish_run()
    report = ledger.report()
    report.update({
        "simulation_only": True,
        "physical_commands_sent": 0,
        "trajectory_commands_sent": 0,
        "attachment_commands_sent": 0,
    })
    return report


def validate_attachment_completion(
    command: RuntimeCommand,
    set_succeeded: bool,
    set_message: str,
    status_attached: bool,
    status_message: str,
) -> tuple:
    """Validate SetBool plus the follower's Trigger status semantics."""
    if not isinstance(command, RuntimeCommand):
        raise TypeError("runtime_command_required")
    expected = command.attachment_value
    if command.command_kind == CommandKind.ATTACH and expected is not True:
        raise ValueError("attach_command_value_invalid")
    if command.command_kind == CommandKind.DETACH and expected is not False:
        raise ValueError("detach_command_value_invalid")
    if command.command_kind not in {CommandKind.ATTACH, CommandKind.DETACH}:
        raise ValueError("attachment_command_kind_required")
    if not set_succeeded:
        return False, f"attachment_request_rejected:{set_message}"

    normalized = str(status_message).strip().lower()
    if expected:
        if (
            status_attached
            and "attached" in normalized
            and "detached" not in normalized
        ):
            return True, "attachment_confirmed"
        return False, f"attachment_not_confirmed:{status_message}"
    if not status_attached and "detached" in normalized:
        return True, "detachment_confirmed"
    return False, f"detachment_not_confirmed:{status_message}"
