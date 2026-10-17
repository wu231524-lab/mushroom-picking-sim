"""ROS-independent state-machine core and non-blocking report snapshot."""

from dataclasses import asdict, dataclass
from enum import Enum
import json
from typing import Any, Callable, Dict, Mapping, Optional
from uuid import uuid4

from .pick_state import (
    PHYSICAL_OPERATION_STATES,
    PICK_SEQUENCE,
    TERMINAL_STATES,
    PickState,
)


class DryRunMode(str, Enum):
    LIVE_GATE = "live_gate"
    REPLAY_FULL_PATH = "replay_full_path"
    AUTONOMOUS_SINGLE_RUN = "autonomous_single_run"


@dataclass(frozen=True)
class StateEvent:
    sequence: int
    run_id: str
    previous_state: str
    state: str
    outcome: str
    reason: str
    evidence: Mapping[str, Any]


class TransitionRejected(RuntimeError):
    pass


class ReportSnapshot:
    """Keep the latest serialized report available without touching the core.

    A report service may be queried much faster than state transitions occur.
    Serializing the mutable state-machine history in every service callback
    makes that diagnostic traffic compete with the state timer.  Updating one
    immutable string at transition time lets readers return immediately.
    """

    def __init__(self, report: Mapping[str, Any]):
        self._message = ""
        self.update(report)

    def update(self, report: Mapping[str, Any]) -> None:
        self._message = json.dumps(report, ensure_ascii=False, default=str)

    def message(self) -> str:
        return self._message


class PickStateMachineCore:
    """A fail-closed transition engine with no external side effects."""

    def __init__(
        self,
        mode: DryRunMode = DryRunMode.LIVE_GATE,
        run_id_factory: Callable[[], str] = lambda: f"day14-{uuid4().hex}",
        execution_enabled: bool = False,
    ):
        self.mode = DryRunMode(mode)
        self.execution_enabled = bool(execution_enabled)
        if (
            self.mode == DryRunMode.AUTONOMOUS_SINGLE_RUN
            and not self.execution_enabled
        ):
            reason = "autonomous_single_run_requires_execution_enabled"
            raise ValueError(reason)
        if (
            self.mode != DryRunMode.AUTONOMOUS_SINGLE_RUN
            and self.execution_enabled
        ):
            raise ValueError("dry_run_mode_rejects_execution_enabled")
        self._run_id_factory = run_id_factory
        self.state = PickState.IDLE
        self.run_id: Optional[str] = None
        self.events = []
        self.failure_reason: Optional[str] = None
        self.would_execute_state: Optional[PickState] = None
        self._index = -1

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    def _record(
        self,
        previous: PickState,
        state: PickState,
        outcome: str,
        reason: str,
        evidence: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.events.append(StateEvent(
            sequence=len(self.events) + 1,
            run_id=self.run_id or "unstarted",
            previous_state=previous.value,
            state=state.value,
            outcome=outcome,
            reason=reason,
            evidence=dict(evidence or {}),
        ))

    def start(self, reason: str = "dry_run_requested") -> PickState:
        if self.state != PickState.IDLE:
            raise TransitionRejected("run_already_started")
        self.run_id = self._run_id_factory()
        self._index = 0
        previous = self.state
        self.state = PICK_SEQUENCE[self._index]
        self._record(previous, self.state, "entered", reason)
        return self.state

    def succeed(
        self,
        reason: str,
        evidence: Optional[Mapping[str, Any]] = None,
    ) -> PickState:
        if self.terminal or self.state == PickState.IDLE:
            raise TransitionRejected("terminal_or_idle_transition_rejected")
        if self._index < 0 or PICK_SEQUENCE[self._index] != self.state:
            raise TransitionRejected("state_sequence_corrupt")

        previous = self.state
        self._record(previous, previous, "passed", reason, evidence)
        next_index = self._index + 1
        if next_index >= len(PICK_SEQUENCE):
            raise TransitionRejected("sequence_exhausted")
        next_state = PICK_SEQUENCE[next_index]

        if (
            self.mode == DryRunMode.LIVE_GATE
            and next_state in PHYSICAL_OPERATION_STATES
        ):
            self.would_execute_state = next_state
            self.state = PickState.DRY_RUN_COMPLETE
            self._record(
                previous,
                self.state,
                "blocked_by_dry_run",
                f"would_enter_{next_state.value.lower()}",
                {"would_execute_state": next_state.value,
                    "execution_authorized": False},
            )
            return self.state

        self._index = next_index
        self.state = next_state
        self._record(previous, self.state, "entered", reason, evidence)
        return self.state

    def fail(
        self,
        reason: str,
        evidence: Optional[Mapping[str, Any]] = None,
    ) -> PickState:
        if self.state == PickState.SAFE_STOP:
            return self.state
        if self.terminal:
            raise TransitionRejected("terminal_transition_rejected")
        previous = self.state
        self.failure_reason = reason
        self.state = PickState.SAFE_STOP
        self._record(previous, self.state, "failed", reason, evidence)
        return self.state

    def run_replay(
        self,
        evidence_by_state: Optional[Dict[str, Mapping[str, Any]]] = None,
    ) -> dict:
        """Traverse the full path while marking every step simulated-only."""
        if self.mode != DryRunMode.REPLAY_FULL_PATH:
            raise TransitionRejected("full_replay_requires_replay_mode")
        evidence_by_state = evidence_by_state or {}
        if self.state == PickState.IDLE:
            self.start("full_path_replay_requested")
        while not self.terminal:
            evidence = dict(evidence_by_state.get(self.state.value, {}))
            evidence.update(
                {"simulated_only": True, "execution_authorized": False})
            self.succeed("simulated_gate_passed", evidence)
        return self.report()

    def report(self) -> dict:
        schema = (
            "mushroom_pick_day15_autonomous/v1"
            if self.mode == DryRunMode.AUTONOMOUS_SINGLE_RUN
            else "mushroom_pick_day14_dry_run/v1"
        )
        return {
            "schema": schema,
            "run_id": self.run_id,
            "mode": self.mode.value,
            "state": self.state.value,
            "terminal": self.terminal,
            "failure_reason": self.failure_reason,
            "would_execute_state": (
                self.would_execute_state.value
                if self.would_execute_state
                else None
            ),
            "execution_authorized": self.execution_enabled,
            "physical_commands_sent": 0,
            "trajectory_commands_sent": 0,
            "attachment_commands_sent": 0,
            "motion_executor_started": False,
            "events": [asdict(event) for event in self.events],
        }
