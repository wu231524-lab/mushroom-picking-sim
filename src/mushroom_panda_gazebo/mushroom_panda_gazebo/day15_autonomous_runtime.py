"""ROS-free single-run coordinator for the DAY15 autonomous pick cycle."""

from typing import Mapping, Optional

from .day15_command_runtime import CommandLedger, RuntimeCommand
from .day15_execution_plan import CommandKind, STAGE_BY_STATE
from .day15_verified_trajectories import JointVector
from .pick_state import PHYSICAL_OPERATION_STATES, PickState
from .pick_state_machine_core import (
    DryRunMode,
    PickStateMachineCore,
    TransitionRejected,
)


class AutonomousSingleRunRuntime:
    """Coordinate logical gates and exactly one ledger-issued command."""

    def __init__(self, run_id: str):
        self.machine = PickStateMachineCore(
            DryRunMode.AUTONOMOUS_SINGLE_RUN,
            run_id_factory=lambda: run_id,
            execution_enabled=True,
        )
        self.ledger = CommandLedger(run_id, execution_authorized=True)
        self.safe_stop_count = 0

    @property
    def state(self) -> PickState:
        return self.machine.state

    @property
    def terminal(self) -> bool:
        return self.machine.terminal

    def start(
        self,
        reason: str = "autonomous_single_run_started",
    ) -> PickState:
        return self.machine.start(reason)

    def pass_current_gate(
        self,
        reason: str,
        evidence: Optional[Mapping] = None,
    ) -> PickState:
        if self.state in PHYSICAL_OPERATION_STATES:
            raise TransitionRejected("physical_state_requires_command")
        state = self.machine.succeed(reason, evidence)
        if state == PickState.DONE:
            self.ledger.finish_run()
        return state

    def issue_current_command(
        self,
        dynamic_xy_target: Optional[JointVector] = None,
        dynamic_xy_duration_sec: float = 5.0,
    ) -> RuntimeCommand:
        state = self.state
        if state not in PHYSICAL_OPERATION_STATES:
            raise TransitionRejected("current_state_is_not_physical")
        if self.ledger.active_state is None:
            self.ledger.enter_state(state)
        elif self.ledger.active_state != state:
            raise TransitionRejected("ledger_state_mismatch")

        stage = STAGE_BY_STATE[state]
        if state == PickState.XY_ALIGN:
            if dynamic_xy_target is None:
                raise TransitionRejected("dynamic_xy_target_required")
            return self.ledger.issue_dynamic_xy(
                dynamic_xy_target,
                dynamic_xy_duration_sec,
            )
        if stage.command_kind == CommandKind.TRAJECTORY:
            return self.ledger.issue_next_fixed_trajectory()
        return self.ledger.issue_attachment()

    def complete_command(
        self,
        command_id: str,
        succeeded: bool,
        reason: str,
        evidence: Optional[Mapping] = None,
    ) -> PickState:
        command = self.ledger.active_command
        if command is None:
            raise TransitionRejected("runtime_has_no_active_command")
        if command.command_id != command_id:
            raise TransitionRejected("runtime_command_id_mismatch")
        self.ledger.complete(command_id, succeeded, reason)
        if not succeeded:
            return self.safe_stop(reason, evidence)

        if command.state == PickState.XY_ALIGN:
            return self.state
        stage = STAGE_BY_STATE[command.state]
        if self.ledger.counts[command.state] < stage.maximum_commands:
            return self.state
        self.ledger.close_state()
        return self.machine.succeed(reason, evidence)

    def finish_xy_alignment(
        self,
        reason: str,
        evidence: Optional[Mapping] = None,
    ) -> PickState:
        if self.state != PickState.XY_ALIGN:
            raise TransitionRejected("xy_alignment_not_active")
        if self.ledger.active_command is not None:
            raise TransitionRejected("xy_command_still_active")
        if self.ledger.active_state is None:
            self.ledger.enter_state(PickState.XY_ALIGN)
        self.ledger.close_state()
        return self.machine.succeed(reason, evidence)

    def timeout_active_command(self, reason: str) -> PickState:
        command = self.ledger.active_command
        if command is None:
            return self.safe_stop(reason)
        self.ledger.complete(command.command_id, False, reason)
        return self.safe_stop(reason, {"command_id": command.command_id})

    def safe_stop(
        self,
        reason: str,
        evidence: Optional[Mapping] = None,
    ) -> PickState:
        self.ledger.fail(reason)
        if self.machine.state != PickState.SAFE_STOP:
            self.safe_stop_count += 1
            self.machine.fail(reason, evidence)
        return self.machine.state

    def report(self, simulation_only: bool = False) -> dict:
        logical = self.machine.report()
        ledger = self.ledger.report()
        issued = sum(self.ledger.counts.values())
        return {
            **logical,
            "simulation_only": bool(simulation_only),
            "physical_commands_sent": 0,
            "trajectory_commands_sent": 0,
            "attachment_commands_sent": 0,
            "issued_command_count": issued,
            "safe_stop_count": self.safe_stop_count,
            "command_report": ledger,
        }


def simulate_autonomous_single_run(
    fail_at_command: Optional[int] = None,
    timeout_at_command: Optional[int] = None,
    xy_command_count: int = 5,
) -> dict:
    """Drive the integrated coordinator in memory without ROS or I/O."""
    if not 0 <= xy_command_count <= 5:
        raise ValueError("xy_command_count_invalid")
    runtime = AutonomousSingleRunRuntime("day15-autonomous-no-motion")
    runtime.start("no_motion_integrated_replay_started")
    issued = 0

    while not runtime.terminal:
        state = runtime.state
        if state not in PHYSICAL_OPERATION_STATES:
            runtime.pass_current_gate(
                "simulated_gate_passed",
                {"simulation_only": True},
            )
            continue

        if (
            state == PickState.XY_ALIGN
            and runtime.ledger.counts[state] >= xy_command_count
        ):
            runtime.finish_xy_alignment(
                "simulated_xy_gate_passed",
                {"simulation_only": True},
            )
            continue

        target = (0.0,) * 7 if state == PickState.XY_ALIGN else None
        command = runtime.issue_current_command(target)
        issued += 1
        if issued == timeout_at_command:
            runtime.timeout_active_command("simulated_command_timeout")
        else:
            succeeded = issued != fail_at_command
            runtime.complete_command(
                command.command_id,
                succeeded,
                (
                    "simulated_command_completed"
                    if succeeded
                    else "simulated_command_failed"
                ),
                {"simulation_only": True},
            )

    report = runtime.report(simulation_only=True)
    report["execution_authorized"] = False
    report["logical_event_count"] = len(report["events"])
    report["planned_command_count"] = report["issued_command_count"]
    report["planned_command_counts_by_state"] = report[
        "command_report"
    ]["command_counts_by_state"]
    return report
