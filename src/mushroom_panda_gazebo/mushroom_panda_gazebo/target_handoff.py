"""Pure validation for the DAY14 D435-to-D405 target handoff."""

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Optional, Tuple
from uuid import uuid4


class HandoffRejected(ValueError):
    """Raised when target identity, freshness, or geometry is untrusted."""


@dataclass(frozen=True)
class HandoffLimits:
    global_max_age_sec: float = 1.0
    d405_max_age_sec: float = 0.5
    max_sync_delta_sec: float = 0.20
    min_depth_m: float = 0.07
    max_depth_m: float = 0.50
    min_sphere_radius_m: float = 0.05
    max_sphere_radius_m: float = 0.09
    max_sphere_rms_m: float = 0.001
    min_sphere_inliers: int = 1000
    max_center_distance_px: float = 120.0
    max_tracking_jump_m: float = 0.010
    min_observations: int = 2
    max_observation_spread_m: float = 0.001
    max_cap_center_error_m: float = 0.020
    max_coarse_global_distance_m: float = 0.080


@dataclass(frozen=True)
class GlobalTargetLock:
    run_id: str
    handoff_id: str
    entity_name: str
    lock_generation: int
    stamp_ns: int
    compensated_world: Tuple[float, float, float]
    expected_cap_center_world: Tuple[float, float, float]


def stamp_to_nanoseconds(value: Any) -> int:
    if not isinstance(value, Mapping):
        raise HandoffRejected("stamp_missing")
    try:
        seconds = int(value["sec"])
        nanoseconds = int(value["nanosec"])
    except (KeyError, TypeError, ValueError) as error:
        raise HandoffRejected("stamp_invalid") from error
    if seconds < 0 or not 0 <= nanoseconds < 1_000_000_000:
        raise HandoffRejected("stamp_invalid")
    return seconds * 1_000_000_000 + nanoseconds


def _finite_xyz(value: Any, reason: str) -> Tuple[float, float, float]:
    if not isinstance(value, Mapping):
        raise HandoffRejected(reason)
    try:
        xyz = (float(value["x"]), float(value["y"]), float(value["z"]))
    except (KeyError, TypeError, ValueError) as error:
        raise HandoffRejected(reason) from error
    if not all(math.isfinite(item) for item in xyz):
        raise HandoffRejected(reason)
    return xyz


def _age_seconds(
    now_ns: int,
    stamp_ns: int,
    max_age_sec: float,
    prefix: str,
) -> float:
    age = (now_ns - stamp_ns) / 1_000_000_000.0
    if age < -1.0e-6:
        raise HandoffRejected(f"{prefix}_stamp_in_future")
    if age > max_age_sec:
        raise HandoffRejected(f"{prefix}_stale")
    return age


def create_global_lock(
    payload: Mapping[str, Any],
    now_ns: int,
    run_id: str,
    generation: int,
    limits: Optional[HandoffLimits] = None,
) -> GlobalTargetLock:
    """Validate the current selector payload and bind a run-scoped identity."""
    limits = limits or HandoffLimits()
    if not isinstance(run_id, str) or not run_id:
        raise HandoffRejected("run_id_invalid")
    if payload.get("valid") is not True:
        raise HandoffRejected("global_target_invalid")
    if payload.get("frame_id") != "world":
        raise HandoffRejected("global_target_frame_invalid")

    entity_name = payload.get("entity_name")
    if entity_name is None and payload.get("reason") == "mushroom3_selected":
        entity_name = "mushroom_3"
    if entity_name != "mushroom_3":
        raise HandoffRejected("global_target_identity_invalid")

    stamp_ns = stamp_to_nanoseconds(payload.get("stamp"))
    _age_seconds(now_ns, stamp_ns, limits.global_max_age_sec, "global_target")
    compensated = _finite_xyz(
        payload.get("compensated_world"), "global_target_world_invalid"
    )

    expected_cap = (0.300, -0.120, 0.325)
    try:
        lock_generation = int(generation)
    except (TypeError, ValueError) as error:
        raise HandoffRejected("lock_generation_invalid") from error
    if lock_generation < 1:
        raise HandoffRejected("lock_generation_invalid")

    return GlobalTargetLock(
        run_id=run_id,
        handoff_id=f"handoff-{uuid4().hex}",
        entity_name=entity_name,
        lock_generation=lock_generation,
        stamp_ns=stamp_ns,
        compensated_world=compensated,
        expected_cap_center_world=expected_cap,
    )


def validate_d405_handoff(
    target_lock: GlobalTargetLock,
    evidence: Mapping[str, Any],
    now_ns: int,
    limits: Optional[HandoffLimits] = None,
) -> dict:
    """Validate stable D405 sphere evidence against the entity lock."""
    limits = limits or HandoffLimits()
    if evidence.get("valid") is not True:
        raise HandoffRejected("d405_evidence_invalid")
    if evidence.get("entity_name") != target_lock.entity_name:
        raise HandoffRejected("d405_entity_mismatch")
    if evidence.get("handoff_id") != target_lock.handoff_id:
        raise HandoffRejected("d405_handoff_id_mismatch")
    try:
        evidence_generation = int(evidence.get("lock_generation", -1))
    except (TypeError, ValueError) as error:
        raise HandoffRejected("d405_lock_generation_invalid") from error
    if evidence_generation != target_lock.lock_generation:
        raise HandoffRejected("d405_lock_generation_mismatch")

    stamp_ns = stamp_to_nanoseconds(evidence.get("stamp"))
    age = _age_seconds(now_ns, stamp_ns, limits.d405_max_age_sec, "d405")
    try:
        sync_delta = float(evidence.get("sync_delta_sec", math.inf))
        depth = float(evidence.get("depth_m", math.nan))
        radius = float(evidence.get("sphere_radius_m", math.nan))
        rms = float(evidence.get("sphere_fit_rms_m", math.nan))
        inliers = int(evidence.get("sphere_inlier_count", -1))
        center_px = float(evidence.get("center_distance_px", math.inf))
        jump = float(evidence.get("tracking_distance_m", 0.0))
        observations = int(evidence.get("observation_count", 0))
        spread = float(evidence.get("observation_spread_m", math.inf))
    except (TypeError, ValueError) as error:
        raise HandoffRejected("d405_numeric_evidence_invalid") from error

    checks = (
        (math.isfinite(sync_delta) and sync_delta <= limits.max_sync_delta_sec,
         "d405_sync_delta_exceeded"),
        (
            math.isfinite(depth)
            and limits.min_depth_m <= depth <= limits.max_depth_m,
            "d405_depth_out_of_range"),
        (
            math.isfinite(radius)
            and limits.min_sphere_radius_m
            <= radius
            <= limits.max_sphere_radius_m,
            "d405_sphere_radius_out_of_range"),
        (math.isfinite(rms) and rms <= limits.max_sphere_rms_m,
         "d405_sphere_rms_exceeded"),
        (
            inliers >= limits.min_sphere_inliers,
            "d405_sphere_inliers_insufficient",
        ),
        (
            math.isfinite(center_px)
            and center_px <= limits.max_center_distance_px,
            "d405_center_distance_exceeded"),
        (math.isfinite(jump) and jump <= limits.max_tracking_jump_m,
         "d405_tracking_jump_exceeded"),
        (
            observations >= limits.min_observations,
            "d405_observations_insufficient",
        ),
        (math.isfinite(spread) and spread <= limits.max_observation_spread_m,
         "d405_observation_spread_exceeded"),
    )
    for passed, reason in checks:
        if not passed:
            raise HandoffRejected(reason)

    world = _finite_xyz(evidence.get("sphere_world"), "d405_world_invalid")
    cap_distance = math.dist(world, target_lock.expected_cap_center_world)
    coarse_distance = math.dist(world, target_lock.compensated_world)
    if cap_distance > limits.max_cap_center_error_m:
        raise HandoffRejected("d405_expected_cap_center_mismatch")
    if coarse_distance > limits.max_coarse_global_distance_m:
        raise HandoffRejected("d405_global_coarse_gate_exceeded")

    return {
        "valid": True,
        "reason": "target_handoff_valid",
        "target_lock": asdict(target_lock),
        "d405_age_sec": age,
        "cap_center_error_m": cap_distance,
        "coarse_global_distance_m": coarse_distance,
        "motion_authorized": False,
    }
