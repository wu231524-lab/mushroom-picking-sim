"""Pure DAY15 D405 TCP-XY to seven-joint target solver."""

from dataclasses import dataclass
import math
import xml.etree.ElementTree as ET

import numpy as np

from .day15_verified_trajectories import JOINT_NAMES, JointVector


class XYKinematicsRejected(ValueError):
    """Raised when an XY correction cannot satisfy the DAY13 envelope."""


@dataclass(frozen=True)
class XYKinematicLimits:
    """Verified limits copied from the DAY13 D405 step executor."""

    max_step_m: float = 0.002
    damping: float = 0.02
    max_joint_delta_rad: float = 0.007
    min_joint_limit_margin_rad: float = 0.02
    min_jacobian_singular_value: float = 0.05
    max_prediction_error_m: float = 0.00025
    max_predicted_z_m: float = 0.00010
    max_orientation_error_rad: float = 0.001
    min_direction_cosine: float = 0.98
    trajectory_duration_sec: float = 2.0


class FerKinematics:
    """Load the live expanded FER URDF and compute TCP Jacobians."""

    def __init__(self, urdf_path: str, base_link: str, tip_link: str):
        self.base_link = base_link
        self.tip_link = tip_link
        self.chain = self.load_chain(urdf_path, base_link, tip_link)
        movable = [
            item["name"] for item in self.chain if item["type"] != "fixed"
        ]
        if tuple(movable) != JOINT_NAMES:
            raise RuntimeError(f"unexpected_movable_chain:{movable}")
        self.lower_limits = np.array([
            item["lower"]
            for item in self.chain
            if item["type"] != "fixed"
        ])
        self.upper_limits = np.array([
            item["upper"]
            for item in self.chain
            if item["type"] != "fixed"
        ])

    @staticmethod
    def parse_vector(text, default="0 0 0"):
        return np.array([float(value) for value in (text or default).split()])

    @staticmethod
    def rpy_rotation(rpy):
        roll, pitch, yaw = rpy
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        rotation_x = np.array([
            [1.0, 0.0, 0.0],
            [0.0, cr, -sr],
            [0.0, sr, cr],
        ])
        rotation_y = np.array([
            [cp, 0.0, sp],
            [0.0, 1.0, 0.0],
            [-sp, 0.0, cp],
        ])
        rotation_z = np.array([
            [cy, -sy, 0.0],
            [sy, cy, 0.0],
            [0.0, 0.0, 1.0],
        ])
        return rotation_z @ rotation_y @ rotation_x

    @staticmethod
    def axis_rotation(axis, angle):
        axis = axis / np.linalg.norm(axis)
        x, y, z = axis
        skew = np.array([
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ])
        return (
            np.eye(3)
            + math.sin(angle) * skew
            + (1.0 - math.cos(angle)) * (skew @ skew)
        )

    @staticmethod
    def transform(rotation=None, translation=None):
        result = np.eye(4)
        if rotation is not None:
            result[:3, :3] = rotation
        if translation is not None:
            result[:3, 3] = translation
        return result

    @staticmethod
    def rotation_error_angle(first, second):
        relative = first.T @ second
        cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
        return math.acos(cosine)

    def load_chain(self, urdf_path, base_link, tip_link):
        root = ET.parse(urdf_path).getroot()
        child_to_joint = {}
        for joint in root.findall("joint"):
            child = joint.find("child")
            if child is not None:
                child_to_joint[child.attrib["link"]] = joint

        xml_chain = []
        current_link = tip_link
        while current_link != base_link:
            if current_link not in child_to_joint:
                raise RuntimeError(f"kinematic_chain_broken_at:{current_link}")
            joint = child_to_joint[current_link]
            xml_chain.append(joint)
            current_link = joint.find("parent").attrib["link"]
        xml_chain.reverse()

        result = []
        for joint in xml_chain:
            joint_type = joint.attrib["type"]
            origin = joint.find("origin")
            origin_xyz = self.parse_vector(
                origin.attrib.get("xyz") if origin is not None else None
            )
            origin_rpy = self.parse_vector(
                origin.attrib.get("rpy") if origin is not None else None
            )
            axis_element = joint.find("axis")
            axis = None
            if axis_element is not None:
                axis = self.parse_vector(axis_element.attrib.get("xyz"))
                axis = axis / np.linalg.norm(axis)
            lower = None
            upper = None
            if joint_type != "fixed":
                if joint_type != "revolute":
                    raise RuntimeError(f"unsupported_joint_type:{joint_type}")
                limit = joint.find("limit")
                if limit is None:
                    raise RuntimeError(
                        f"joint_limit_missing:{joint.attrib['name']}"
                    )
                lower = float(limit.attrib["lower"])
                upper = float(limit.attrib["upper"])
            result.append({
                "name": joint.attrib["name"],
                "type": joint_type,
                "origin": self.transform(
                    self.rpy_rotation(origin_rpy),
                    origin_xyz,
                ),
                "axis": axis,
                "lower": lower,
                "upper": upper,
            })
        return result

    def forward_and_jacobian(self, positions):
        tcp_transform = np.eye(4)
        joint_records = []
        movable_index = 0
        for joint in self.chain:
            tcp_transform = tcp_transform @ joint["origin"]
            if joint["type"] == "fixed":
                continue
            axis_local = joint["axis"]
            joint_position = tcp_transform[:3, 3].copy()
            axis_base = tcp_transform[:3, :3] @ axis_local
            joint_records.append((joint_position, axis_base))
            tcp_transform = tcp_transform @ self.transform(
                self.axis_rotation(axis_local, positions[movable_index])
            )
            movable_index += 1

        tcp_position = tcp_transform[:3, 3]
        jacobian = np.zeros((6, len(joint_records)))
        for index, (position, axis) in enumerate(joint_records):
            jacobian[:3, index] = np.cross(axis, tcp_position - position)
            jacobian[3:, index] = axis
        return tcp_transform, jacobian


def solve_xy_joint_target(
    kinematics: FerKinematics,
    current_joints: JointVector,
    delta_tcp,
    limits: XYKinematicLimits = XYKinematicLimits(),
) -> dict:
    """Solve and independently forward-check one D405 TCP correction."""
    current = np.asarray(current_joints, dtype=float)
    delta = np.asarray(delta_tcp, dtype=float)
    if current.shape != (7,) or not np.all(np.isfinite(current)):
        raise XYKinematicsRejected("xy_current_joints_invalid")
    if delta.shape != (3,) or not np.all(np.isfinite(delta)):
        raise XYKinematicsRejected("xy_delta_tcp_invalid")
    if abs(float(delta[2])) > 1.0e-12:
        raise XYKinematicsRejected("xy_z_motion_forbidden")
    planar_norm = float(np.linalg.norm(delta[:2]))
    if planar_norm <= 0.0 or planar_norm > limits.max_step_m + 1.0e-9:
        raise XYKinematicsRejected("xy_planar_step_invalid")

    current_transform, jacobian = kinematics.forward_and_jacobian(current)
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    if np.linalg.matrix_rank(jacobian, tol=1.0e-8) < 6:
        raise XYKinematicsRejected("xy_jacobian_rank_deficient")
    minimum_singular = float(singular_values[-1])
    if minimum_singular < limits.min_jacobian_singular_value:
        raise XYKinematicsRejected("xy_jacobian_too_close_to_singular")

    tcp_rotation = current_transform[:3, :3]
    desired_translation_base = tcp_rotation @ delta
    desired_twist = np.concatenate((desired_translation_base, np.zeros(3)))
    joint_delta = jacobian.T @ np.linalg.solve(
        jacobian @ jacobian.T + (limits.damping ** 2) * np.eye(6),
        desired_twist,
    )
    if not np.all(np.isfinite(joint_delta)):
        raise XYKinematicsRejected("xy_joint_delta_not_finite")
    maximum_joint_delta = float(np.max(np.abs(joint_delta)))
    if maximum_joint_delta > limits.max_joint_delta_rad:
        raise XYKinematicsRejected("xy_joint_delta_limit_exceeded")

    candidate = current + joint_delta
    lower = kinematics.lower_limits + limits.min_joint_limit_margin_rad
    upper = kinematics.upper_limits - limits.min_joint_limit_margin_rad
    if not np.all((candidate >= lower) & (candidate <= upper)):
        raise XYKinematicsRejected("xy_joint_limit_margin_violated")

    candidate_transform, _ = kinematics.forward_and_jacobian(candidate)
    actual_translation_base = (
        candidate_transform[:3, 3] - current_transform[:3, 3]
    )
    predicted_delta_tcp = tcp_rotation.T @ actual_translation_base
    prediction_error = float(np.linalg.norm(predicted_delta_tcp - delta))
    orientation_error = kinematics.rotation_error_angle(
        current_transform[:3, :3],
        candidate_transform[:3, :3],
    )
    if prediction_error > limits.max_prediction_error_m:
        raise XYKinematicsRejected("xy_fk_prediction_error_exceeded")
    if abs(float(predicted_delta_tcp[2])) > limits.max_predicted_z_m:
        raise XYKinematicsRejected("xy_predicted_z_motion_exceeded")
    if orientation_error > limits.max_orientation_error_rad:
        raise XYKinematicsRejected("xy_predicted_orientation_error_exceeded")

    denominator = (
        np.linalg.norm(delta[:2])
        * np.linalg.norm(predicted_delta_tcp[:2])
    )
    if denominator <= 0.0:
        raise XYKinematicsRejected("xy_predicted_planar_step_zero")
    direction_cosine = float(
        np.dot(delta[:2], predicted_delta_tcp[:2]) / denominator
    )
    if direction_cosine < limits.min_direction_cosine:
        raise XYKinematicsRejected("xy_predicted_direction_mismatch")

    return {
        "valid": True,
        "reason": "xy_joint_target_valid",
        "current_joints": tuple(float(value) for value in current),
        "target_joints": tuple(float(value) for value in candidate),
        "joint_delta": tuple(float(value) for value in joint_delta),
        "predicted_delta_tcp": tuple(
            float(value) for value in predicted_delta_tcp
        ),
        "maximum_joint_delta_rad": maximum_joint_delta,
        "minimum_singular_value": minimum_singular,
        "prediction_error_m": prediction_error,
        "orientation_error_rad": orientation_error,
        "direction_cosine": direction_cosine,
        "trajectory_duration_sec": limits.trajectory_duration_sec,
    }
