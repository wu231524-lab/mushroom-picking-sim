"""Integrated no-motion replay for the DAY15 autonomous cycle."""

import json

from .day15_autonomous_runtime import simulate_autonomous_single_run


SUMMARY_FIELDS = (
    "state",
    "terminal",
    "simulation_only",
    "execution_authorized",
    "motion_executor_started",
    "logical_event_count",
    "planned_command_count",
    "issued_command_count",
    "physical_commands_sent",
    "trajectory_commands_sent",
    "attachment_commands_sent",
    "safe_stop_count",
    "failure_reason",
    "run_id",
    "would_execute_state",
)


def build_autonomous_no_motion_report() -> dict:
    """Replay the state gates and all planned commands without external I/O."""
    report = simulate_autonomous_single_run()
    if report["state"] != "DONE":
        raise RuntimeError("integrated_replay_did_not_finish")
    if not report["command_report"]["closed"]:
        raise RuntimeError("command_replay_did_not_close")
    report.update({
        "schema": "mushroom_pick_day15_autonomous_no_motion/v1",
        "logical_report": {
            key: value
            for key, value in report.items()
            if key != "command_report"
        },
    })
    return report


def build_autonomous_no_motion_summary(report=None) -> dict:
    """Return a bounded ROS-service summary of the verified full replay."""
    if report is None:
        report = build_autonomous_no_motion_report()
    summary = {
        field: report.get(field)
        for field in SUMMARY_FIELDS
    }
    summary.update({
        "schema": "mushroom_pick_day15_no_motion_summary/v1",
        "dry_replay_verified": True,
        "full_report_utf8_bytes": len(json.dumps(
            report,
            ensure_ascii=False,
            default=str,
        ).encode("utf-8")),
    })
    return summary
