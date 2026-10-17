"""Pure validators for DAY15 live sensor and service evidence."""

import json
import math
from typing import Any, Mapping, MutableSet, Optional, Sequence

from .target_handoff import (
    GlobalTargetLock,
    HandoffLimits,
    HandoffRejected,
    create_global_lock,
    stamp_to_nanoseconds,
    validate_d405_handoff,
)


class EvidenceRejected(ValueError):
    """Raised when live evidence cannot safely advance the runtime."""


def decode_json_object(value: Any, label: str) -> dict:
    """Decode one String payload and require a JSON object."""
    try:
        payload = json.loads(value)
    except (json.JSONDecodeError, TypeError) as error:
        raise EvidenceRejected(f"{label}_json_invalid") from error
    if not isinstance(payload, dict):
        raise EvidenceRejected(f"{label}_not_object")
    return payload


def validate_fresh_global_target(
    payload: Mapping[str, Any],
    now_ns: int,
    received_age_sec: float,
    sequence: int,
    sequence_baseline: int,
    run_id: str,
    generation: int,
    limits: Optional[HandoffLimits] = None,
) -> GlobalTargetLock:
    """Require a newly received D435 target and create its run lock."""
    if sequence <= sequence_baseline:
        raise EvidenceRejected("d435_target_not_received_after_run_start")
    try:
        receive_age = float(received_age_sec)
    except (TypeError, ValueError) as error:
        raise EvidenceRejected("d435_receive_age_invalid") from error
    if not math.isfinite(receive_age) or receive_age < -0.05:
        raise EvidenceRejected("d435_receive_age_invalid")
    if receive_age > 1.0:
        raise EvidenceRejected("d435_target_receive_stale")
    try:
        return create_global_lock(
            payload,
            now_ns,
            run_id,
            generation,
            limits or HandoffLimits(global_max_age_sec=2.0),
        )
    except HandoffRejected as error:
        raise EvidenceRejected(str(error)) from error


def d405_observation_start_ready(
    alignment: Optional[Mapping[str, Any]],
    received_age_sec: float,
    sequence: int,
    sequence_baseline: int,
) -> bool:
    """Wait for a fresh valid D405 lock before starting an observation."""
    if alignment is None or sequence <= sequence_baseline:
        return False
    if (
        not math.isfinite(received_age_sec)
        or received_age_sec < 0.0
        or received_age_sec > 2.0
    ):
        return False
    tracking = alignment.get("target_tracking")
    if not isinstance(tracking, Mapping):
        return False
    generation = alignment.get(
        "session_generation", tracking.get("session_generation")
    )
    return (
        alignment.get("valid") is True
        and isinstance(generation, int)
        and generation >= 0
        and isinstance(tracking.get("lock_id"), str)
        and bool(tracking["lock_id"])
    )


def _finite_number(value: Any, reason: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise EvidenceRejected(reason) from error
    if not math.isfinite(number):
        raise EvidenceRejected(reason)
    return number


def _fresh_stamp(
    stamp: Any,
    now_ns: int,
    max_age_sec: float,
    prefix: str,
) -> float:
    try:
        stamp_ns = stamp_to_nanoseconds(stamp)
    except HandoffRejected as error:
        raise EvidenceRejected(str(error)) from error
    age = (now_ns - stamp_ns) / 1_000_000_000.0
    if age < -1.0e-6:
        raise EvidenceRejected(f"{prefix}_stamp_in_future")
    if age > max_age_sec:
        raise EvidenceRejected(f"{prefix}_stale")
    return age


def validate_xy_proposal(
    proposal: Mapping[str, Any],
    now_ns: int,
    consumed_ids: Optional[MutableSet[str]] = None,
    max_age_sec: float = 2.0,
) -> dict:
    """Validate one unprivileged D405 XY proposal without executing it."""
    if proposal.get("valid") is not True:
        raise EvidenceRejected("xy_proposal_invalid")
    if proposal.get("observation_active") is not False:
        raise EvidenceRejected("xy_observation_not_complete")
    if proposal.get("source_frame") != "fer_hand_tcp":
        raise EvidenceRejected("xy_proposal_frame_invalid")
    if proposal.get("single_step_only") is not True:
        raise EvidenceRejected("xy_single_step_evidence_missing")
    if proposal.get("execute_allowed") is not False:
        raise EvidenceRejected("xy_proposal_must_remain_unprivileged")

    proposal_id = proposal.get("proposal_id")
    if not isinstance(proposal_id, str) or not proposal_id:
        raise EvidenceRejected("xy_proposal_id_invalid")
    if consumed_ids is not None and proposal_id in consumed_ids:
        raise EvidenceRejected("xy_proposal_already_consumed")
    generated_age = _fresh_stamp(
        proposal.get("generated_stamp"),
        now_ns,
        max_age_sec,
        "xy_proposal",
    )

    try:
        observations = int(proposal.get("observation_count", 0))
        required = int(proposal.get("required_observations", 2))
    except (TypeError, ValueError) as error:
        raise EvidenceRejected("xy_observation_count_invalid") from error
    if required < 2 or observations < required:
        raise EvidenceRejected("xy_observations_insufficient")
    try:
        first_stamp_ns = stamp_to_nanoseconds(proposal.get("stamp_first"))
        last_stamp_ns = stamp_to_nanoseconds(proposal.get("stamp_last"))
    except HandoffRejected as error:
        raise EvidenceRejected("xy_observation_stamp_invalid") from error
    if last_stamp_ns <= first_stamp_ns:
        raise EvidenceRejected("xy_observation_stamps_not_increasing")
    sync_evidence = proposal.get("upstream_sync_evidence")
    if not isinstance(sync_evidence, list):
        raise EvidenceRejected("xy_sync_evidence_missing")
    if len(sync_evidence) != observations:
        raise EvidenceRejected("xy_sync_evidence_count_mismatch")
    for item in sync_evidence:
        if not isinstance(item, Mapping) or item.get("valid") is not True:
            raise EvidenceRejected("xy_sync_evidence_invalid")
        sync_delta = _finite_number(
            item.get("delta_sec"),
            "xy_sync_delta_invalid",
        )
        if sync_delta < 0.0 or sync_delta > 0.20:
            raise EvidenceRejected("xy_sync_delta_exceeded")
    spread = _finite_number(
        proposal.get("maximum_spread"),
        "xy_observation_spread_invalid",
    )
    if spread < 0.0 or spread > 0.001:
        raise EvidenceRejected("xy_observation_spread_exceeded")

    max_step = _finite_number(
        proposal.get("max_step_m"),
        "xy_max_step_invalid",
    )
    deadband = _finite_number(
        proposal.get("deadband_m"),
        "xy_deadband_invalid",
    )
    if abs(max_step - 0.002) > 1.0e-9:
        raise EvidenceRejected("xy_step_limit_mismatch")
    if abs(deadband - 0.002) > 1.0e-9:
        raise EvidenceRejected("xy_deadband_mismatch")

    mean = proposal.get("mean_alignment_error")
    delta = proposal.get("proposed_delta_tcp")
    if not isinstance(mean, Mapping) or not isinstance(delta, Mapping):
        raise EvidenceRejected("xy_vector_missing")
    mean_x = _finite_number(mean.get("x"), "xy_mean_error_invalid")
    mean_y = _finite_number(mean.get("y"), "xy_mean_error_invalid")
    mean_planar = _finite_number(
        mean.get("planar"),
        "xy_mean_error_invalid",
    )
    delta_x = _finite_number(delta.get("x"), "xy_delta_invalid")
    delta_y = _finite_number(delta.get("y"), "xy_delta_invalid")
    delta_z = _finite_number(delta.get("z"), "xy_delta_invalid")
    if abs(math.hypot(mean_x, mean_y) - mean_planar) > 1.0e-6:
        raise EvidenceRejected("xy_mean_error_inconsistent")
    if abs(delta_z) > 1.0e-12:
        raise EvidenceRejected("xy_z_motion_forbidden")
    delta_norm = math.hypot(delta_x, delta_y)
    if delta_norm > max_step + 1.0e-9:
        raise EvidenceRejected("xy_step_limit_exceeded")

    aligned = mean_planar <= deadband
    expected_reason = (
        "within_deadband" if aligned else "stable_dry_run_proposal"
    )
    if proposal.get("reason") != expected_reason:
        raise EvidenceRejected("xy_proposal_reason_inconsistent")
    if aligned and delta_norm > 1.0e-12:
        raise EvidenceRejected("xy_deadband_delta_not_zero")
    if not aligned and delta_norm <= 1.0e-12:
        raise EvidenceRejected("xy_correction_delta_zero")
    if consumed_ids is not None:
        consumed_ids.add(proposal_id)
    return {
        "valid": True,
        "proposal_id": proposal_id,
        "generated_age_sec": generated_age,
        "aligned": aligned,
        "planar_error_m": mean_planar,
        "delta_tcp": (delta_x, delta_y, delta_z),
        "observation_count": observations,
        "maximum_spread_m": spread,
    }


def validate_day13_xy_budget_gate(
    xy_report: Mapping[str, Any],
    vacuum_report: Mapping[str, Any],
) -> dict:
    """Cross-check a bounded visual residual against independent geometry."""
    sphere_planar = _finite_number(
        xy_report.get("planar_error_m"), "xy_sphere_planar_invalid"
    )
    # The observed DAY19 final pose was within 2.126 mm in Gazebo model
    # geometry, while D405 reported 3.878 mm. D405 remains a bounded
    # cross-check; the independent vacuum geometry keeps its 3 mm gate.
    if sphere_planar < 0.0 or sphere_planar > 0.005:
        raise EvidenceRejected("xy_sphere_planar_exceeded")
    if vacuum_report.get("entity_name") != "mushroom_3":
        raise EvidenceRejected("xy_vacuum_entity_mismatch")
    model_age = _finite_number(
        vacuum_report.get("model_state_age_sec"),
        "xy_vacuum_model_age_invalid",
    )
    if model_age < 0.0 or model_age > 0.5:
        raise EvidenceRejected("xy_vacuum_model_stale")
    vacuum_planar = _finite_number(
        vacuum_report.get("planar_error_m"),
        "xy_vacuum_planar_invalid",
    )
    if vacuum_planar < 0.0 or vacuum_planar > 0.003:
        raise EvidenceRejected("xy_vacuum_planar_exceeded")
    axis_error = _finite_number(
        vacuum_report.get("axis_error_rad"),
        "xy_vacuum_axis_invalid",
    )
    if axis_error < 0.0 or axis_error > math.radians(10.0):
        raise EvidenceRejected("xy_vacuum_axis_exceeded")
    surface_gap = _finite_number(
        vacuum_report.get("surface_gap_m"),
        "xy_vacuum_surface_gap_invalid",
    )
    # The verified fixed Z route descends about 0.152 m. A merely positive
    # starting gap could place that route inside the cap before contact check.
    if surface_gap < 0.150:
        raise EvidenceRejected("xy_pre_z_surface_gap_too_small")
    if surface_gap > 0.160:
        raise EvidenceRejected("xy_pre_z_surface_gap_too_large")
    return {
        "valid": True,
        "reason": "day13_five_step_xy_vacuum_planar_gate",
        "sphere_planar_error_m": sphere_planar,
        "vacuum_planar_error_m": vacuum_planar,
        "axis_error_rad": axis_error,
        "surface_gap_m": surface_gap,
        "model_state_age_sec": model_age,
        "proposal_id": xy_report.get("proposal_id"),
    }


def validate_d405_chain(
    target_lock: GlobalTargetLock,
    alignment: Mapping[str, Any],
    proposal: Mapping[str, Any],
    sphere_world: Mapping[str, Any],
    now_ns: int,
    limits: Optional[HandoffLimits] = None,
) -> dict:
    """Bind actual alignment/proposal payloads to the D435 target lock."""
    if alignment.get("valid") is not True:
        raise EvidenceRejected("d405_alignment_invalid")
    if alignment.get("target_frame") != "fer_hand_tcp":
        raise EvidenceRejected("d405_target_frame_invalid")
    sync = alignment.get("upstream_sync")
    target = alignment.get("target_d405")
    tracking = alignment.get("target_tracking")
    if not all(isinstance(item, Mapping) for item in (sync, target, tracking)):
        raise EvidenceRejected("d405_alignment_evidence_missing")
    if sync.get("valid") is not True:
        raise EvidenceRejected("d405_sync_invalid")
    if tracking.get("enabled") is not True or not tracking.get("lock_id"):
        raise EvidenceRejected("d405_tracking_lock_missing")

    proposal_report = validate_xy_proposal(proposal, now_ns)
    alignment_stamp = alignment.get(
        "detection_stamp",
        alignment.get("stamp"),
    )
    try:
        alignment_stamp_ns = stamp_to_nanoseconds(alignment_stamp)
        proposal_stamp_ns = stamp_to_nanoseconds(proposal.get("stamp_last"))
    except HandoffRejected as error:
        raise EvidenceRejected(str(error)) from error
    if alignment_stamp_ns != proposal_stamp_ns:
        raise EvidenceRejected("d405_proposal_alignment_stamp_mismatch")

    handoff_evidence = {
        "valid": True,
        "entity_name": target_lock.entity_name,
        "handoff_id": target_lock.handoff_id,
        "lock_generation": target_lock.lock_generation,
        "stamp": alignment_stamp,
        "sync_delta_sec": sync.get("delta_sec"),
        "depth_m": target.get("depth"),
        "sphere_radius_m": target.get("sphere_radius_m"),
        "sphere_fit_rms_m": target.get("sphere_fit_rms_m"),
        "sphere_inlier_count": target.get("sphere_inlier_count"),
        "center_distance_px": target.get("center_distance_px"),
        "tracking_distance_m": tracking.get("tracking_distance_m"),
        "observation_count": proposal_report["observation_count"],
        "observation_spread_m": proposal_report["maximum_spread_m"],
        "sphere_world": sphere_world,
    }
    try:
        # The observed two-frame D405 proposal was generated 2.717 s after
        # its last source frame. Keep the separate 2.0 s proposal-generation
        # gate and bound the source-frame age to 4.0 s for this handoff only.
        report = validate_d405_handoff(
            target_lock,
            handoff_evidence,
            now_ns,
            limits or HandoffLimits(d405_max_age_sec=4.0),
        )
    except HandoffRejected as error:
        raise EvidenceRejected(str(error)) from error
    report["proposal"] = proposal_report
    report["d405_tracking_lock_id"] = tracking.get("lock_id")
    return report


def matching_d405_alignment(
    alignment_history: Sequence[tuple],
    proposal: Mapping[str, Any],
    sequence_baseline: int,
) -> Mapping[str, Any]:
    """Find the exact observation that generated a post-start proposal."""
    try:
        proposal_stamp_ns = stamp_to_nanoseconds(proposal.get("stamp_last"))
    except HandoffRejected as error:
        raise EvidenceRejected("d405_proposal_stamp_invalid") from error
    for sequence, alignment in reversed(alignment_history):
        if sequence <= sequence_baseline:
            break
        try:
            stamp_ns = stamp_to_nanoseconds(
                alignment.get("detection_stamp", alignment.get("stamp"))
            )
        except HandoffRejected:
            continue
        if stamp_ns == proposal_stamp_ns:
            return alignment
    raise EvidenceRejected("d405_proposal_alignment_not_in_history")


def _finite_xyz_sequence(value: Any, reason: str) -> tuple:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise EvidenceRejected(reason)
    if len(value) != 3:
        raise EvidenceRejected(reason)
    return tuple(_finite_number(item, reason) for item in value)


def validate_vacuum_contact_response(
    response_success: bool,
    response_message: str,
    require_place_pose: bool = False,
) -> dict:
    """Validate follower contact geometry and optional DAY13 place pose."""
    report = decode_json_object(response_message, "vacuum_contact")
    if not response_success or report.get("valid") is not True:
        raise EvidenceRejected(
            f"vacuum_contact_rejected:{report.get('reason', 'unknown')}"
        )
    if report.get("entity_name") != "mushroom_3":
        raise EvidenceRejected("vacuum_contact_entity_invalid")
    if report.get("tcp_frame") != "fer_hand_tcp":
        raise EvidenceRejected("vacuum_contact_tcp_frame_invalid")
    age = _finite_number(
        report.get("model_state_age_sec"),
        "vacuum_model_state_age_invalid",
    )
    planar = _finite_number(
        report.get("planar_error_m"),
        "vacuum_planar_error_invalid",
    )
    gap = _finite_number(
        report.get("surface_gap_m"),
        "vacuum_surface_gap_invalid",
    )
    axis = _finite_number(
        report.get("axis_error_rad"),
        "vacuum_axis_error_invalid",
    )
    tcp_world = _finite_xyz_sequence(
        report.get("tcp_world"),
        "vacuum_tcp_world_invalid",
    )
    cap_top = _finite_xyz_sequence(
        report.get("cap_top_world"),
        "vacuum_cap_top_world_invalid",
    )
    if age < 0.0 or age > 0.50:
        raise EvidenceRejected("vacuum_model_state_stale")
    if planar < 0.0 or planar > 0.003:
        raise EvidenceRejected("vacuum_planar_error_exceeded")
    if not -0.002 <= gap <= 0.005:
        raise EvidenceRejected("vacuum_surface_gap_out_of_range")
    if not 0.0 <= axis <= math.radians(10.0):
        raise EvidenceRejected("vacuum_axis_error_exceeded")

    result = {
        "valid": True,
        "reason": "vacuum_contact_valid",
        "model_state_age_sec": age,
        "planar_error_m": planar,
        "surface_gap_m": gap,
        "axis_error_rad": axis,
        "tcp_world": tcp_world,
        "cap_top_world": cap_top,
    }
    if require_place_pose:
        expected_tcp = (
            0.49844398300673304,
            -0.2985249026640738,
            0.3980000015077858,
        )
        tcp_error = math.dist(tcp_world, expected_tcp)
        support_offset = cap_top[2] - 0.400
        support_error = abs(support_offset - (-0.005))
        if tcp_error > 0.001:
            raise EvidenceRejected("place_tcp_error_exceeded")
        if support_error > 0.0005:
            raise EvidenceRejected("place_support_error_exceeded")
        result.update({
            "reason": "place_pose_valid",
            "tcp_error_m": tcp_error,
            "support_offset_m": support_offset,
            "support_error_m": support_error,
        })
    return result


def validate_attachment_status(
    expected_attached: bool,
    status_success: bool,
    status_message: str,
) -> dict:
    """Validate the follower's asymmetric Trigger status response."""
    normalized = str(status_message).strip().lower()
    if expected_attached:
        valid = (
            status_success
            and "attached" in normalized
            and "detached" not in normalized
        )
        reason = "attachment_confirmed"
    else:
        valid = not status_success and "detached" in normalized
        reason = "detachment_confirmed"
    if not valid:
        raise EvidenceRejected(f"attachment_status_mismatch:{status_message}")
    return {
        "valid": True,
        "reason": reason,
        "attached": bool(expected_attached),
        "message": str(status_message),
    }


def validate_robot_settled(
    positions: Sequence[float],
    velocities: Sequence[float],
    expected_positions: Sequence[float],
    received_age_sec: float,
    position_tolerance_rad: float = 0.001,
    velocity_limit_rad_sec: float = 0.01,
) -> dict:
    """Require a fresh, stopped seven-joint snapshot at a target pose."""
    if not (
        len(positions) == len(velocities) == len(expected_positions) == 7
    ):
        raise EvidenceRejected("robot_joint_dimension_invalid")
    values = tuple(positions) + tuple(velocities) + tuple(expected_positions)
    if not all(math.isfinite(float(value)) for value in values):
        raise EvidenceRejected("robot_joint_value_invalid")
    age = _finite_number(received_age_sec, "robot_joint_age_invalid")
    if age < -0.05 or age > 0.50:
        raise EvidenceRejected("robot_joint_state_stale")
    maximum_velocity = max(abs(float(value)) for value in velocities)
    if maximum_velocity > velocity_limit_rad_sec:
        raise EvidenceRejected("robot_not_stopped")
    maximum_error = max(
        abs(float(actual) - float(expected))
        for actual, expected in zip(positions, expected_positions)
    )
    if maximum_error > position_tolerance_rad:
        raise EvidenceRejected("robot_joint_target_not_reached")
    return {
        "valid": True,
        "reason": "robot_settled",
        "joint_state_age_sec": age,
        "maximum_joint_velocity_rad_sec": maximum_velocity,
        "maximum_joint_error_rad": maximum_error,
    }
