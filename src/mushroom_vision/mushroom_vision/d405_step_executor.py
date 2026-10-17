import json
import math
import xml.etree.ElementTree as ET

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectoryPoint


class ProposalRejected(Exception):
    pass


class FerKinematics:

    JOINT_NAMES = [
        "fer_joint1",
        "fer_joint2",
        "fer_joint3",
        "fer_joint4",
        "fer_joint5",
        "fer_joint6",
        "fer_joint7",
    ]

    def __init__(self, urdf_path, base_link, tip_link):
        self.base_link = base_link
        self.tip_link = tip_link
        self.chain = self.load_chain(urdf_path, base_link, tip_link)

        movable_names = [item["name"] for item in self.chain if item["type"] != "fixed"]

        if movable_names != self.JOINT_NAMES:
            raise RuntimeError("unexpected_movable_chain: " f"{movable_names}")

        self.lower_limits = np.array(
            [item["lower"] for item in self.chain if item["type"] != "fixed"]
        )
        self.upper_limits = np.array(
            [item["upper"] for item in self.chain if item["type"] != "fixed"]
        )

    @staticmethod
    def parse_vector(text, default="0 0 0"):
        return np.array([float(value) for value in (text or default).split()])

    @staticmethod
    def rpy_rotation(rpy):
        roll, pitch, yaw = rpy

        cr = math.cos(roll)
        sr = math.sin(roll)
        cp = math.cos(pitch)
        sp = math.sin(pitch)
        cy = math.cos(yaw)
        sy = math.sin(yaw)

        rotation_x = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, cr, -sr],
                [0.0, sr, cr],
            ]
        )
        rotation_y = np.array(
            [
                [cp, 0.0, sp],
                [0.0, 1.0, 0.0],
                [-sp, 0.0, cp],
            ]
        )
        rotation_z = np.array(
            [
                [cy, -sy, 0.0],
                [sy, cy, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        return rotation_z @ rotation_y @ rotation_x

    @staticmethod
    def axis_rotation(axis, angle):
        axis = axis / np.linalg.norm(axis)
        x, y, z = axis
        skew = np.array(
            [
                [0.0, -z, y],
                [z, 0.0, -x],
                [-y, x, 0.0],
            ]
        )

        return (
            np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)
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
        cosine = (np.trace(relative) - 1.0) / 2.0
        cosine = float(np.clip(cosine, -1.0, 1.0))
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
                raise RuntimeError("kinematic_chain_broken_at: " f"{current_link}")

            joint = child_to_joint[current_link]
            parent_link = joint.find("parent").attrib["link"]
            xml_chain.append(joint)
            current_link = parent_link

        xml_chain.reverse()
        result = []

        for joint in xml_chain:
            joint_type = joint.attrib["type"]
            origin = joint.find("origin")

            if origin is None:
                origin_xyz = np.zeros(3)
                origin_rpy = np.zeros(3)
            else:
                origin_xyz = self.parse_vector(origin.attrib.get("xyz"))
                origin_rpy = self.parse_vector(origin.attrib.get("rpy"))

            axis_element = joint.find("axis")

            if axis_element is None:
                axis = None
            else:
                axis = self.parse_vector(axis_element.attrib.get("xyz"))
                axis = axis / np.linalg.norm(axis)

            lower = None
            upper = None

            if joint_type != "fixed":
                if joint_type != "revolute":
                    raise RuntimeError("unsupported_joint_type: " f"{joint_type}")

                limit = joint.find("limit")

                if limit is None:
                    raise RuntimeError(
                        "joint_limit_missing: " f'{joint.attrib["name"]}'
                    )

                lower = float(limit.attrib["lower"])
                upper = float(limit.attrib["upper"])

            result.append(
                {
                    "name": joint.attrib["name"],
                    "type": joint_type,
                    "origin": self.transform(self.rpy_rotation(origin_rpy), origin_xyz),
                    "axis": axis,
                    "lower": lower,
                    "upper": upper,
                }
            )

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


class D405StepExecutor(Node):

    JOINT_NAMES = FerKinematics.JOINT_NAMES

    def __init__(self):
        super().__init__("d405_step_executor")

        self.declare_parameters(
            namespace="",
            parameters=[
                ("enable_execution", False),
                ("proposal_service", "/d405/mushroom_pick/" "get_last_proposal"),
                ("execute_service", "/d405/mushroom_pick/" "execute_last_step"),
                ("state_topic", "/d405/mushroom_pick/" "execution_state"),
                ("joint_state_topic", "/joint_states"),
                ("arm_action", "/fer_arm_controller/" "follow_joint_trajectory"),
                ("attached_service", "/mushroom_pick/get_attached"),
                ("expected_source_frame", "fer_hand_tcp"),
                ("urdf_path", "/tmp/mushroom_pick_demo_runtime.urdf"),
                ("base_link", "fer_link0"),
                ("tip_link", "fer_hand_tcp"),
                ("max_step_m", 0.005),
                ("first_test_mode", True),
                ("first_test_max_step_m", 0.002),
                ("max_command_age_sec", 2.0),
                ("max_sync_delta_sec", 0.20),
                ("trajectory_duration_sec", 2.0),
                ("damping", 0.02),
                ("max_joint_delta_rad", 0.010),
                ("first_test_max_joint_delta_rad", 0.005),
                ("min_joint_limit_margin_rad", 0.02),
                ("max_joint_state_age_sec", 0.50),
                ("max_joint_velocity_rad_sec", 0.01),
                ("min_jacobian_singular_value", 0.05),
                ("max_prediction_error_m", 0.00025),
                ("max_predicted_z_m", 0.00010),
                ("max_orientation_error_rad", 0.001),
                ("min_direction_cosine", 0.98),
                ("allow_z_motion", False),
            ],
        )

        self.enable_execution = bool(self.get_parameter("enable_execution").value)
        self.expected_source_frame = str(
            self.get_parameter("expected_source_frame").value
        )
        self.max_step_m = float(self.get_parameter("max_step_m").value)
        self.first_test_mode = bool(self.get_parameter("first_test_mode").value)
        self.first_test_max_step_m = float(
            self.get_parameter("first_test_max_step_m").value
        )
        self.max_command_age_sec = float(
            self.get_parameter("max_command_age_sec").value
        )
        self.max_sync_delta_sec = float(self.get_parameter("max_sync_delta_sec").value)
        self.trajectory_duration_sec = float(
            self.get_parameter("trajectory_duration_sec").value
        )
        self.damping = float(self.get_parameter("damping").value)
        self.max_joint_delta_rad = float(
            self.get_parameter("max_joint_delta_rad").value
        )
        self.first_test_max_joint_delta_rad = float(
            self.get_parameter("first_test_max_joint_delta_rad").value
        )
        self.min_joint_limit_margin_rad = float(
            self.get_parameter("min_joint_limit_margin_rad").value
        )
        self.max_joint_state_age_sec = float(
            self.get_parameter("max_joint_state_age_sec").value
        )
        self.max_joint_velocity_rad_sec = float(
            self.get_parameter("max_joint_velocity_rad_sec").value
        )
        self.min_jacobian_singular_value = float(
            self.get_parameter("min_jacobian_singular_value").value
        )
        self.max_prediction_error_m = float(
            self.get_parameter("max_prediction_error_m").value
        )
        self.max_predicted_z_m = float(self.get_parameter("max_predicted_z_m").value)
        self.max_orientation_error_rad = float(
            self.get_parameter("max_orientation_error_rad").value
        )
        self.min_direction_cosine = float(
            self.get_parameter("min_direction_cosine").value
        )
        self.allow_z_motion = bool(self.get_parameter("allow_z_motion").value)

        if self.trajectory_duration_sec < 2.0:
            raise RuntimeError("trajectory_duration_sec_below_2.0")

        urdf_path = str(self.get_parameter("urdf_path").value)
        base_link = str(self.get_parameter("base_link").value)
        tip_link = str(self.get_parameter("tip_link").value)
        self.kinematics = FerKinematics(urdf_path, base_link, tip_link)

        state_qos = QoSProfile(depth=1)
        state_qos.reliability = ReliabilityPolicy.RELIABLE
        state_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        self.state_publisher = self.create_publisher(
            String, str(self.get_parameter("state_topic").value), state_qos
        )
        self.joint_subscription = self.create_subscription(
            JointState,
            str(self.get_parameter("joint_state_topic").value),
            self.joint_state_callback,
            10,
        )
        self.proposal_client = self.create_client(
            Trigger, str(self.get_parameter("proposal_service").value)
        )
        self.execute_service = self.create_service(
            Trigger,
            str(self.get_parameter("execute_service").value),
            self.execute_callback,
        )

        self.arm_client = None
        self.attached_client = None

        if self.enable_execution:
            self.arm_client = ActionClient(
                self, FollowJointTrajectory, str(self.get_parameter("arm_action").value)
            )
            self.attached_client = self.create_client(
                Trigger, str(self.get_parameter("attached_service").value)
            )

        self.latest_joint_state = None
        self.latest_joint_state_received_ns = None
        self.busy = False
        self.current_proposal_id = None
        self.current_goal_handle = None
        self.consumed_proposal_ids = set()
        self.pending_execution = None

        initial_reason = (
            "ready_for_manual_single_step"
            if self.enable_execution
            else "execution_disabled_by_default"
        )
        self.publish_state("idle", initial_reason)

        self.get_logger().info(
            "D405 单步执行器已启动: " f"enable_execution={self.enable_execution}"
        )

        if not self.enable_execution:
            self.get_logger().info(
                "默认 dry-run 安全模式: " "未创建机械臂 Action client"
            )

    @staticmethod
    def stamp_to_nanoseconds(stamp):
        if not isinstance(stamp, dict):
            return None

        try:
            sec = int(stamp["sec"])
            nanosec = int(stamp["nanosec"])
        except (KeyError, TypeError, ValueError):
            return None

        if sec < 0 or nanosec < 0 or nanosec >= 1_000_000_000:
            return None

        return sec * 1_000_000_000 + nanosec

    def publish_state(self, state, reason, extra=None):
        output = {
            "state": state,
            "reason": reason,
            "enable_execution": self.enable_execution,
            "busy": self.busy,
            "proposal_id": self.current_proposal_id,
            "requires_new_observation": (state in ("succeeded", "failed")),
        }

        if isinstance(extra, dict):
            output.update(extra)

        message = String()
        message.data = json.dumps(output, ensure_ascii=False)
        self.state_publisher.publish(message)

    def joint_state_callback(self, message):
        self.latest_joint_state = message
        self.latest_joint_state_received_ns = self.get_clock().now().nanoseconds

    def reject(self, reason):
        self.busy = False
        self.pending_execution = None
        self.publish_state("rejected", reason)
        self.get_logger().warning(f"单步执行被拒绝: {reason}")

    def fail(self, reason):
        self.busy = False
        self.pending_execution = None
        self.publish_state("failed", reason)
        self.get_logger().error(f"单步执行失败: {reason}")

    def execute_callback(self, request, response):
        del request

        if not self.enable_execution:
            reason = "execution_disabled"
            self.publish_state("rejected", reason)
            response.success = False
            response.message = reason
            return response

        if self.busy:
            reason = "execution_already_in_progress"
            self.publish_state("rejected", reason)
            response.success = False
            response.message = reason
            return response

        proposal_node_count = sum(
            1
            for name, namespace in (
                self.get_node_names_and_namespaces()
            )
            if name == "d405_step_proposal"
            and namespace == "/"
        )

        if proposal_node_count != 1:
            reason = (
                "proposal_node_count_not_one: "
                f"{proposal_node_count}"
            )
            self.publish_state("rejected", reason)
            response.success = False
            response.message = reason
            return response

        if not self.proposal_client.service_is_ready():
            reason = "proposal_service_not_ready"
            self.publish_state("rejected", reason)
            response.success = False
            response.message = reason
            return response

        self.busy = True
        self.current_proposal_id = None
        self.publish_state("accepted", "querying_last_proposal")

        future = self.proposal_client.call_async(Trigger.Request())
        future.add_done_callback(self.proposal_response_callback)

        response.success = True
        response.message = "accepted; querying and validating " "last proposal"
        return response

    def validate_proposal(self, proposal):
        if not isinstance(proposal, dict):
            raise ProposalRejected("proposal_is_not_object")

        if proposal.get("valid") is not True:
            raise ProposalRejected("proposal_not_valid")

        if proposal.get("reason") != ("stable_dry_run_proposal"):
            raise ProposalRejected("proposal_reason_not_executable")

        if proposal.get("observation_active") is not False:
            raise ProposalRejected("observation_still_active")

        if proposal.get("source_frame") != (self.expected_source_frame):
            raise ProposalRejected("unexpected_source_frame")

        if proposal.get("execute_allowed") is not False:
            raise ProposalRejected("proposal_must_remain_unprivileged")

        if proposal.get("single_step_only") is not True:
            raise ProposalRejected("single_step_evidence_missing")

        active_max_step = (
            self.first_test_max_step_m
            if self.first_test_mode
            else self.max_step_m
        )

        try:
            proposal_max_step = float(
                proposal["max_step_m"]
            )
        except (KeyError, TypeError, ValueError):
            raise ProposalRejected(
                "proposal_max_step_invalid"
            )

        if (
            not math.isfinite(proposal_max_step)
            or proposal_max_step <= 0.0
        ):
            raise ProposalRejected(
                "proposal_max_step_invalid"
            )

        if abs(
            proposal_max_step - active_max_step
        ) > 1e-9:
            raise ProposalRejected(
                "proposal_step_limit_mismatch"
            )

        proposal_id = proposal.get("proposal_id")

        if not isinstance(proposal_id, str) or not proposal_id:
            raise ProposalRejected("proposal_id_missing")

        if proposal_id in self.consumed_proposal_ids:
            raise ProposalRejected("proposal_already_consumed")

        generated_ns = self.stamp_to_nanoseconds(proposal.get("generated_stamp"))

        if generated_ns is None:
            raise ProposalRejected("generated_stamp_invalid")

        age_sec = (self.get_clock().now().nanoseconds - generated_ns) / 1_000_000_000.0

        if age_sec < -0.05:
            raise ProposalRejected("proposal_stamp_in_future")

        if age_sec > self.max_command_age_sec:
            raise ProposalRejected("proposal_too_old")

        first_ns = self.stamp_to_nanoseconds(proposal.get("stamp_first"))
        last_ns = self.stamp_to_nanoseconds(proposal.get("stamp_last"))

        if first_ns is None or last_ns is None:
            raise ProposalRejected("observation_stamps_invalid")

        if last_ns <= first_ns:
            raise ProposalRejected("observation_stamps_not_increasing")

        observation_count = proposal.get("observation_count")
        required_count = proposal.get("required_observations")

        if (
            not isinstance(observation_count, int)
            or not isinstance(required_count, int)
            or observation_count < required_count
            or required_count < 2
        ):
            raise ProposalRejected("observation_count_invalid")

        sync_evidence = proposal.get("upstream_sync_evidence")

        if (
            not isinstance(sync_evidence, list)
            or len(sync_evidence) != observation_count
        ):
            raise ProposalRejected("sync_evidence_count_invalid")

        for evidence in sync_evidence:
            if not isinstance(evidence, dict) or evidence.get("valid") is not True:
                raise ProposalRejected("sync_evidence_not_valid")

            try:
                sync_delta = float(evidence["delta_sec"])
            except (KeyError, TypeError, ValueError):
                raise ProposalRejected("sync_delta_invalid")

            if (
                not math.isfinite(sync_delta)
                or sync_delta < 0.0
                or sync_delta > self.max_sync_delta_sec
            ):
                raise ProposalRejected("sync_delta_exceeded")

        delta_data = proposal.get("proposed_delta_tcp")

        if not isinstance(delta_data, dict):
            raise ProposalRejected("proposed_delta_missing")

        try:
            delta = np.array(
                [
                    float(delta_data["x"]),
                    float(delta_data["y"]),
                    float(delta_data["z"]),
                ]
            )
        except (KeyError, TypeError, ValueError):
            raise ProposalRejected("proposed_delta_invalid")

        if not np.all(np.isfinite(delta)):
            raise ProposalRejected("proposed_delta_not_finite")

        if not self.allow_z_motion and abs(delta[2]) > 1e-9:
            raise ProposalRejected("z_motion_forbidden")

        planar_norm = float(np.linalg.norm(delta[:2]))

        if planar_norm <= 0.0:
            raise ProposalRejected("zero_planar_step")

        if planar_norm > proposal_max_step + 1e-9:
            raise ProposalRejected(
                "proposed_delta_exceeds_declared_limit"
            )

        if planar_norm > active_max_step + 1e-9:
            raise ProposalRejected("step_limit_exceeded")

        return proposal_id, delta, age_sec

    def proposal_response_callback(self, future):
        try:
            response = future.result()
        except Exception as error:
            self.reject("proposal_service_call_failed: " f"{error}")
            return

        if not response.success:
            self.reject("proposal_service_returned_failure: " f"{response.message}")
            return

        try:
            proposal = json.loads(response.message)
        except json.JSONDecodeError:
            self.reject("proposal_json_invalid")
            return

        try:
            proposal_id, delta, age_sec = self.validate_proposal(proposal)
        except ProposalRejected as error:
            self.reject(str(error))
            return

        self.current_proposal_id = proposal_id
        self.pending_execution = {
            "proposal": proposal,
            "delta": delta,
            "age_sec": age_sec,
        }

        if self.attached_client is None or not self.attached_client.service_is_ready():
            self.reject("attached_status_service_not_ready")
            return

        attached_future = self.attached_client.call_async(Trigger.Request())
        attached_future.add_done_callback(self.attached_status_callback)

    def attached_status_callback(self, future):
        try:
            response = future.result()
        except Exception as error:
            self.reject("attached_status_call_failed: " f"{error}")
            return

        message = response.message.lower()

        if response.success or "detached" not in message:
            self.reject("mushroom_not_confirmed_detached")
            return

        try:
            execution = self.prepare_execution(self.pending_execution["delta"])
        except ProposalRejected as error:
            self.reject(str(error))
            return

        execution.update(self.pending_execution)
        self.pending_execution = execution
        self.send_goal()

    def current_joint_positions(self):
        if (
            self.latest_joint_state is None
            or self.latest_joint_state_received_ns is None
        ):
            raise ProposalRejected("joint_state_missing")

        age_sec = (
            self.get_clock().now().nanoseconds - self.latest_joint_state_received_ns
        ) / 1_000_000_000.0

        if age_sec > self.max_joint_state_age_sec:
            raise ProposalRejected("joint_state_too_old")

        positions_by_name = dict(
            zip(self.latest_joint_state.name, self.latest_joint_state.position)
        )
        velocities_by_name = dict(
            zip(self.latest_joint_state.name, self.latest_joint_state.velocity)
        )

        if any(name not in positions_by_name for name in self.JOINT_NAMES):
            raise ProposalRejected("joint_state_names_incomplete")

        positions = np.array([positions_by_name[name] for name in self.JOINT_NAMES])

        if not np.all(np.isfinite(positions)):
            raise ProposalRejected("joint_positions_not_finite")

        velocities = []

        for name in self.JOINT_NAMES:
            if name not in velocities_by_name:
                raise ProposalRejected("joint_velocity_missing")
            velocities.append(velocities_by_name[name])

        velocities = np.array(velocities)

        if not np.all(np.isfinite(velocities)):
            raise ProposalRejected("joint_velocities_not_finite")

        if np.max(np.abs(velocities)) > (self.max_joint_velocity_rad_sec):
            raise ProposalRejected("robot_not_stopped")

        return positions

    def prepare_execution(self, delta_tcp):
        current = self.current_joint_positions()
        current_transform, jacobian = self.kinematics.forward_and_jacobian(current)
        singular_values = np.linalg.svd(jacobian, compute_uv=False)

        if np.linalg.matrix_rank(jacobian, tol=1e-8) < 6:
            raise ProposalRejected("jacobian_rank_deficient")

        if singular_values[-1] < (self.min_jacobian_singular_value):
            raise ProposalRejected("jacobian_too_close_to_singular")

        tcp_rotation = current_transform[:3, :3]
        desired_translation_base = tcp_rotation @ delta_tcp
        desired_twist = np.concatenate(
            [
                desired_translation_base,
                np.zeros(3),
            ]
        )

        joint_delta = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + (self.damping**2) * np.eye(6), desired_twist
        )

        if not np.all(np.isfinite(joint_delta)):
            raise ProposalRejected("joint_delta_not_finite")

        active_joint_limit = (
            self.first_test_max_joint_delta_rad
            if self.first_test_mode
            else self.max_joint_delta_rad
        )
        maximum_joint_delta = float(np.max(np.abs(joint_delta)))

        if maximum_joint_delta > active_joint_limit:
            raise ProposalRejected("joint_delta_limit_exceeded")

        candidate = current + joint_delta
        lower = self.kinematics.lower_limits + self.min_joint_limit_margin_rad
        upper = self.kinematics.upper_limits - self.min_joint_limit_margin_rad

        if not np.all((candidate >= lower) & (candidate <= upper)):
            raise ProposalRejected("joint_limit_margin_violated")

        candidate_transform, _ = self.kinematics.forward_and_jacobian(candidate)
        actual_translation_base = candidate_transform[:3, 3] - current_transform[:3, 3]
        predicted_delta_tcp = tcp_rotation.T @ actual_translation_base
        prediction_error = float(np.linalg.norm(predicted_delta_tcp - delta_tcp))
        orientation_error = self.kinematics.rotation_error_angle(
            current_transform[:3, :3], candidate_transform[:3, :3]
        )

        if prediction_error > self.max_prediction_error_m:
            raise ProposalRejected("fk_prediction_error_exceeded")

        if abs(predicted_delta_tcp[2]) > (self.max_predicted_z_m):
            raise ProposalRejected("predicted_z_motion_exceeded")

        if orientation_error > (self.max_orientation_error_rad):
            raise ProposalRejected("predicted_orientation_error_exceeded")

        desired_planar = delta_tcp[:2]
        predicted_planar = predicted_delta_tcp[:2]
        denominator = np.linalg.norm(desired_planar) * np.linalg.norm(predicted_planar)

        if denominator <= 0.0:
            raise ProposalRejected("predicted_planar_step_zero")

        direction_cosine = float(np.dot(desired_planar, predicted_planar) / denominator)

        if direction_cosine < self.min_direction_cosine:
            raise ProposalRejected("predicted_direction_mismatch")

        return {
            "current": current,
            "candidate": candidate,
            "joint_delta": joint_delta,
            "predicted_delta_tcp": predicted_delta_tcp,
            "maximum_joint_delta": maximum_joint_delta,
            "minimum_singular_value": float(singular_values[-1]),
            "prediction_error": prediction_error,
            "orientation_error": orientation_error,
            "direction_cosine": direction_cosine,
        }

    def send_goal(self):
        if self.arm_client is None or not self.arm_client.server_is_ready():
            self.reject("arm_action_server_not_ready")
            return

        proposal_id = self.current_proposal_id
        self.consumed_proposal_ids.add(proposal_id)

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(self.JOINT_NAMES)

        point = JointTrajectoryPoint()
        point.positions = self.pending_execution["candidate"].tolist()
        point.velocities = [0.0] * len(self.JOINT_NAMES)

        duration_ns = int(round(self.trajectory_duration_sec * 1_000_000_000))
        point.time_from_start.sec = duration_ns // 1_000_000_000
        point.time_from_start.nanosec = duration_ns % 1_000_000_000
        goal.trajectory.points = [point]

        self.publish_state(
            "accepted",
            "all_safety_checks_passed; sending_goal",
            self.execution_summary(),
        )

        try:
            future = self.arm_client.send_goal_async(goal)
        except Exception as error:
            self.fail("send_goal_failed: " f"{error}")
            return

        future.add_done_callback(self.goal_response_callback)

    def execution_summary(self):
        if not isinstance(self.pending_execution, dict):
            return {}

        return {
            "proposed_delta_tcp": (self.pending_execution["delta"].tolist()),
            "joint_delta": (self.pending_execution["joint_delta"].tolist()),
            "candidate_joints": (self.pending_execution["candidate"].tolist()),
            "predicted_delta_tcp": (
                self.pending_execution["predicted_delta_tcp"].tolist()
            ),
            "maximum_joint_delta": (self.pending_execution["maximum_joint_delta"]),
            "minimum_singular_value": (
                self.pending_execution["minimum_singular_value"]
            ),
            "prediction_error": (self.pending_execution["prediction_error"]),
            "orientation_error": (self.pending_execution["orientation_error"]),
            "direction_cosine": (self.pending_execution["direction_cosine"]),
            "trajectory_duration_sec": (self.trajectory_duration_sec),
        }

    def goal_response_callback(self, future):
        try:
            goal_handle = future.result()
        except Exception as error:
            self.fail("goal_response_failed: " f"{error}")
            return

        if not goal_handle.accepted:
            self.fail("trajectory_goal_rejected")
            return

        self.current_goal_handle = goal_handle
        self.publish_state(
            "executing", "single_trajectory_goal_accepted", self.execution_summary()
        )
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.goal_result_callback)

    def goal_result_callback(self, future):
        try:
            wrapped_result = future.result()
            result = wrapped_result.result
        except Exception as error:
            self.fail("trajectory_result_failed: " f"{error}")
            return

        self.current_goal_handle = None

        if wrapped_result.status != (GoalStatus.STATUS_SUCCEEDED):
            self.fail("trajectory_status_not_succeeded: " f"{wrapped_result.status}")
            return

        if result.error_code != 0:
            self.fail(
                "trajectory_error_code: "
                f"{result.error_code}; "
                f"{result.error_string}"
            )
            return

        summary = self.execution_summary()
        self.busy = False
        self.publish_state(
            "succeeded", "single_step_completed; " "new_observation_required", summary
        )
        self.pending_execution = None
        self.get_logger().info(
            "单次轨迹完成；禁止自动第二步，" "必须重新观察并比较误差"
        )


def main(args=None):
    rclpy.init(args=args)
    node = D405StepExecutor()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
