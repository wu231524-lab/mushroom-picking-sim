"""Single ROS 2 entry point for one complete mushroom-pick run."""

import json
import math
import threading
from collections import deque
from uuid import uuid4

import rclpy
from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PointStamped
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.clock import Clock, ClockType
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import SetBool, Trigger
from tf2_geometry_msgs import do_transform_point
from tf2_ros import Buffer, TransformException, TransformListener
from trajectory_msgs.msg import JointTrajectoryPoint

from .day15_autonomous_runtime import AutonomousSingleRunRuntime
from .day15_command_runtime import (
    RuntimeCommand,
    validate_attachment_completion,
)
from .day15_execution_plan import CommandKind, STAGE_BY_STATE
from .day15_runtime_evidence import (
    EvidenceRejected,
    d405_observation_start_ready,
    matching_d405_alignment,
    decode_json_object,
    validate_day13_xy_budget_gate,
    validate_attachment_status,
    validate_d405_chain,
    validate_fresh_global_target,
    validate_robot_settled,
    validate_vacuum_contact_response,
    validate_xy_proposal,
)
from .day15_state_deadlines import (
    StateDeadlineExceeded,
    StateDeadlineTracker,
    trajectory_wall_timeout_sec,
)
from .day15_verified_trajectories import JOINT_NAMES
from .day15_verified_trajectories import (
    HOME_JOINTS,
    POST_XY_JOINTS,
    PREGRASP_JOINTS,
)
from .day15_xy_kinematics import (
    FerKinematics,
    XYKinematicLimits,
    XYKinematicsRejected,
    solve_xy_joint_target,
)
from .pick_state_machine_core import (
    DryRunMode,
    PickStateMachineCore,
    ReportSnapshot,
)
from .pick_state import PHYSICAL_OPERATION_STATES, PickState


class Day15AutonomousPick(Node):
    """Serve dry replay or one service-triggered autonomous runtime."""

    def __init__(self):
        super().__init__("day15_autonomous_pick")
        self.declare_parameter("execution_enabled", False)
        self.declare_parameter("joint_state_max_age_sec", 0.5)
        self.declare_parameter("max_joint_velocity_rad_sec", 0.01)
        self.declare_parameter("joint_goal_tolerance_rad", 0.001)
        self.declare_parameter("trajectory_timeout_margin_sec", 2.0)
        self.declare_parameter("motion_wall_time_scale", 4.0)
        self.declare_parameter("attachment_timeout_sec", 2.0)
        self.declare_parameter("watchdog_period_sec", 0.05)
        self.declare_parameter(
            "urdf_path",
            "/tmp/mushroom_pick_demo_runtime.urdf",
        )
        self.declare_parameter("base_link", "fer_link0")
        self.declare_parameter("tip_link", "fer_hand_tcp")
        self.execution_enabled = bool(
            self.get_parameter("execution_enabled").value
        )
        self.joint_state_max_age_sec = float(
            self.get_parameter("joint_state_max_age_sec").value
        )
        self.max_joint_velocity_rad_sec = float(
            self.get_parameter("max_joint_velocity_rad_sec").value
        )
        self.joint_goal_tolerance_rad = float(
            self.get_parameter("joint_goal_tolerance_rad").value
        )
        self.trajectory_timeout_margin_sec = float(
            self.get_parameter("trajectory_timeout_margin_sec").value
        )
        self.motion_wall_time_scale = float(
            self.get_parameter("motion_wall_time_scale").value
        )
        self.attachment_timeout_sec = float(
            self.get_parameter("attachment_timeout_sec").value
        )
        self.watchdog_period_sec = float(
            self.get_parameter("watchdog_period_sec").value
        )
        if min(
            self.joint_state_max_age_sec,
            self.max_joint_velocity_rad_sec,
            self.joint_goal_tolerance_rad,
            self.trajectory_timeout_margin_sec,
            self.attachment_timeout_sec,
            self.watchdog_period_sec,
        ) <= 0.0:
            raise ValueError("runtime_joint_limits_must_be_positive")

        state_qos = QoSProfile(depth=1)
        state_qos.reliability = ReliabilityPolicy.RELIABLE
        state_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.state_publisher = self.create_publisher(
            String,
            "/mushroom_pick/day15/state",
            state_qos,
        )
        self.report_group = ReentrantCallbackGroup()
        self.sensor_group = ReentrantCallbackGroup()
        self.runtime_lock = threading.Lock()
        self.transition_lock = threading.RLock()
        self.steady_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.state_deadline = StateDeadlineTracker(
            motion_wall_time_scale=self.motion_wall_time_scale,
        )
        self.latest_joint_state = None
        self.latest_joint_received_ns = None
        self.active_runtime_command = None
        self.active_completion_callback = None
        self.active_command_deadline_ns = None
        self.active_goal_handle = None
        self.runtime = None
        self.dry_run_machine = None
        self.latest_global_target = None
        self.latest_global_target_received_ns = None
        self.latest_global_target_sequence = 0
        self.run_target_sequence_baseline = 0
        self.lock_generation = 0
        self.target_lock = None
        self.latest_d405_alignment = None
        self.latest_d405_alignment_received_ns = None
        self.latest_d405_alignment_sequence = 0
        self.d405_alignment_history = deque(maxlen=256)
        self.last_d405_gate_rejection = None
        self.first_d405_candidate_rejection = None
        self.d405_start_alignment_baseline = 0
        self.latest_d405_proposal = None
        self.consumed_proposal_ids = set()
        self.d405_start_future = None
        self.contact_future = None
        self.status_gate_future = None
        self.pending_xy_solution = None
        self.pending_xy_budget_gate = None
        self.latest_d405_proposal_sequence = 0
        self.d405_proposal_sequence_baseline = 0
        self.d405_request_state = None
        self.perception_switch_future = None
        self.close_range_perception_ready = False
        self.trajectory_commands_sent = 0
        self.attachment_commands_sent = 0
        self.arm_client = None
        self.attachment_client = None
        self.attachment_status_client = None
        self.vacuum_contact_client = None
        self.d405_observation_client = None
        self.kinematics = None
        self.xy_kinematic_limits = XYKinematicLimits()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.attachment_status_client = self.create_client(
            Trigger,
            "/mushroom_pick/get_attached",
        )
        self.vacuum_contact_client = self.create_client(
            Trigger,
            "/mushroom_pick/check_vacuum_contact",
        )
        self.d405_observation_client = self.create_client(
            Trigger,
            "/d405/mushroom_pick/start_observation",
        )
        self.global_detector_enable_client = self.create_client(
            SetBool,
            "/mushroom_detector/set_enabled",
        )
        self.d405_detector_enable_client = self.create_client(
            SetBool,
            "/d405_mushroom_detector/set_enabled",
        )
        self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_callback,
            10,
            callback_group=self.sensor_group,
        )
        self.create_subscription(
            String,
            "/mushroom_pick/selected_target",
            self.global_target_callback,
            10,
            callback_group=self.sensor_group,
        )
        self.create_subscription(
            String,
            "/d405/mushroom_pick/sphere_alignment_error",
            self.d405_alignment_callback,
            10,
            callback_group=self.sensor_group,
        )
        self.create_subscription(
            String,
            "/d405/mushroom_pick/step_proposal",
            self.d405_proposal_callback,
            10,
            callback_group=self.sensor_group,
        )
        if self.execution_enabled:
            self.kinematics = FerKinematics(
                str(self.get_parameter("urdf_path").value),
                str(self.get_parameter("base_link").value),
                str(self.get_parameter("tip_link").value),
            )
            self.arm_client = ActionClient(
                self,
                FollowJointTrajectory,
                "/fer_arm_controller/follow_joint_trajectory",
            )
            self.attachment_client = self.create_client(
                SetBool,
                "/mushroom_pick/set_attached",
            )
        initial = {
            "schema": "mushroom_pick_day15_autonomous_no_motion/v1",
            "state": "IDLE",
            "terminal": False,
            "simulation_only": not self.execution_enabled,
            "execution_authorized": self.execution_enabled,
            "physical_commands_sent": 0,
            "trajectory_commands_sent": 0,
            "attachment_commands_sent": 0,
        }
        self.report_snapshot = ReportSnapshot(initial)
        self.run_service = self.create_service(
            Trigger,
            "/mushroom_pick/run_once",
            self.run_callback,
            callback_group=self.report_group,
        )
        self.report_service = self.create_service(
            Trigger,
            "/mushroom_pick/day15/get_report",
            self.report_callback,
            callback_group=self.report_group,
        )
        self.watchdog_timer = self.create_timer(
            self.watchdog_period_sec,
            self.runtime_watchdog_callback,
            callback_group=self.report_group,
            clock=self.steady_clock,
        )
        self.progress_timer = self.create_timer(
            0.10,
            self.runtime_progress_callback,
            callback_group=self.report_group,
            clock=self.steady_clock,
        )
        self.publish(initial)

    def joint_callback(self, message):
        mapping = dict(zip(message.name, message.position))
        velocities = dict(zip(message.name, message.velocity))
        complete = all(
            name in mapping and name in velocities
            for name in JOINT_NAMES
        )
        if not complete:
            return
        position = tuple(float(mapping[name]) for name in JOINT_NAMES)
        velocity = tuple(float(velocities[name]) for name in JOINT_NAMES)
        if not all(math.isfinite(value) for value in position + velocity):
            return
        with self.runtime_lock:
            self.latest_joint_state = (position, velocity)
            self.latest_joint_received_ns = (
                self.steady_clock.now().nanoseconds
            )

    def global_target_callback(self, message):
        try:
            payload = decode_json_object(message.data, "d435_target")
        except EvidenceRejected:
            return
        with self.runtime_lock:
            self.latest_global_target = payload
            self.latest_global_target_received_ns = (
                self.steady_clock.now().nanoseconds
            )
            self.latest_global_target_sequence += 1

    def d405_alignment_callback(self, message):
        try:
            payload = decode_json_object(message.data, "d405_alignment")
        except EvidenceRejected:
            return
        with self.runtime_lock:
            self.latest_d405_alignment = payload
            self.latest_d405_alignment_received_ns = (
                self.steady_clock.now().nanoseconds
            )
            self.latest_d405_alignment_sequence += 1
            self.d405_alignment_history.append(
                (self.latest_d405_alignment_sequence, payload)
            )

    def d405_proposal_callback(self, message):
        try:
            payload = decode_json_object(message.data, "d405_proposal")
        except EvidenceRejected:
            return
        with self.runtime_lock:
            self.latest_d405_proposal = payload
            self.latest_d405_proposal_sequence += 1

    def current_joint_snapshot(self):
        with self.runtime_lock:
            state = self.latest_joint_state
            received_ns = self.latest_joint_received_ns
        if state is None or received_ns is None:
            raise RuntimeError("joint_state_missing")
        age_sec = (
            self.steady_clock.now().nanoseconds - received_ns
        ) / 1_000_000_000.0
        if age_sec < -0.05 or age_sec > self.joint_state_max_age_sec:
            raise RuntimeError("joint_state_stale")
        position, velocity = state
        if max(abs(value) for value in velocity) > (
            self.max_joint_velocity_rad_sec
        ):
            raise RuntimeError("robot_not_stopped")
        return position, velocity, age_sec

    def dispatch_trajectory(self, command, completion_callback):
        """Send one ledger-issued trajectory and report its action result."""
        if not self.execution_enabled or self.arm_client is None:
            raise RuntimeError("physical_execution_disabled")
        if not isinstance(command, RuntimeCommand):
            raise TypeError("runtime_command_required")
        if command.command_kind != CommandKind.TRAJECTORY:
            raise RuntimeError("trajectory_command_kind_required")
        if command.target_joints is None or len(command.target_joints) != 7:
            raise RuntimeError("trajectory_target_invalid")
        if (
            command.trajectory_duration_sec is None
            or not math.isfinite(command.trajectory_duration_sec)
            or command.trajectory_duration_sec <= 0.0
        ):
            raise RuntimeError("trajectory_duration_invalid")
        self.activate_runtime_command(
            command,
            completion_callback,
            trajectory_wall_timeout_sec(
                command.trajectory_duration_sec,
                self.motion_wall_time_scale,
                self.trajectory_timeout_margin_sec,
            ),
        )

        try:
            self.current_joint_snapshot()
        except RuntimeError as error:
            self.finish_runtime_command(False, str(error))
            return
        if not self.arm_client.server_is_ready():
            self.finish_runtime_command(False, "arm_action_server_not_ready")
            return

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(JOINT_NAMES)
        point = JointTrajectoryPoint()
        point.positions = list(command.target_joints)
        point.velocities = [0.0] * len(JOINT_NAMES)
        duration_ns = int(round(
            command.trajectory_duration_sec * 1_000_000_000
        ))
        point.time_from_start.sec = duration_ns // 1_000_000_000
        point.time_from_start.nanosec = duration_ns % 1_000_000_000
        goal.trajectory.points = [point]
        try:
            future = self.arm_client.send_goal_async(goal)
        except Exception as error:
            self.finish_runtime_command(
                False,
                f"trajectory_send_failed:{error}",
            )
            return
        self.trajectory_commands_sent += 1
        future.add_done_callback(
            lambda response_future, command_id=command.command_id: (
                self.goal_response_callback(response_future, command_id)
            )
        )

    def goal_response_callback(self, future, command_id):
        try:
            goal_handle = future.result()
        except Exception as error:
            self.finish_runtime_command(
                False,
                f"trajectory_goal_response_failed:{error}",
                command_id,
            )
            return
        if not goal_handle.accepted:
            self.finish_runtime_command(
                False,
                "trajectory_goal_rejected",
                command_id,
            )
            return
        with self.runtime_lock:
            active = self.active_runtime_command
            if active is not None and active.command_id == command_id:
                self.active_goal_handle = goal_handle
                accept_result = True
            else:
                accept_result = False
        if not accept_result:
            try:
                goal_handle.cancel_goal_async()
            except Exception:
                pass
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda finished_future, active_id=command_id: (
                self.goal_result_callback(finished_future, active_id)
            )
        )

    def goal_result_callback(self, future, command_id):
        try:
            wrapped = future.result()
            result = wrapped.result
        except Exception as error:
            self.finish_runtime_command(
                False,
                f"trajectory_result_failed:{error}",
                command_id,
            )
            return
        if wrapped.status != GoalStatus.STATUS_SUCCEEDED:
            self.finish_runtime_command(
                False,
                f"trajectory_status_not_succeeded:{wrapped.status}",
                command_id,
            )
            return
        if result.error_code != 0:
            self.finish_runtime_command(
                False,
                f"trajectory_error_code:{result.error_code}",
                command_id,
            )
            return
        try:
            position, _, _ = self.current_joint_snapshot()
        except RuntimeError as error:
            self.finish_runtime_command(False, str(error), command_id)
            return
        with self.runtime_lock:
            command = self.active_runtime_command
        if command is None or command.target_joints is None:
            self.finish_runtime_command(
                False,
                "active_command_missing",
                command_id,
            )
            return
        maximum_error = max(
            abs(actual - expected)
            for actual, expected in zip(position, command.target_joints)
        )
        if maximum_error > self.joint_goal_tolerance_rad:
            self.finish_runtime_command(
                False,
                f"trajectory_joint_error_exceeded:{maximum_error}",
                command_id,
            )
            return
        self.finish_runtime_command(
            True,
            "trajectory_succeeded",
            command_id,
        )

    def dispatch_attachment(self, command, completion_callback):
        """Send one ledger-issued attach or detach command and confirm it."""
        if (
            not self.execution_enabled
            or self.attachment_client is None
            or self.attachment_status_client is None
        ):
            raise RuntimeError("physical_execution_disabled")
        if not isinstance(command, RuntimeCommand):
            raise TypeError("runtime_command_required")
        if command.command_kind not in {
            CommandKind.ATTACH,
            CommandKind.DETACH,
        }:
            raise RuntimeError("attachment_command_kind_required")
        expected = command.command_kind == CommandKind.ATTACH
        if command.attachment_value is not expected:
            raise RuntimeError("attachment_command_value_invalid")
        self.activate_runtime_command(
            command,
            completion_callback,
            self.attachment_timeout_sec,
        )

        if not self.attachment_client.service_is_ready():
            self.finish_runtime_command(
                False,
                "attachment_service_not_ready",
            )
            return
        if not self.attachment_status_client.service_is_ready():
            self.finish_runtime_command(
                False,
                "attachment_status_service_not_ready",
            )
            return

        request = SetBool.Request()
        request.data = expected
        try:
            future = self.attachment_client.call_async(request)
        except Exception as error:
            self.finish_runtime_command(
                False,
                f"attachment_send_failed:{error}",
            )
            return
        self.attachment_commands_sent += 1
        future.add_done_callback(
            lambda result_future, command_id=command.command_id: (
                self.attachment_set_result_callback(
                    result_future,
                    command_id,
                )
            )
        )

    def attachment_set_result_callback(self, future, command_id):
        try:
            result = future.result()
        except Exception as error:
            self.finish_runtime_command(
                False,
                f"attachment_service_failed:{error}",
                command_id,
            )
            return
        if not result.success:
            self.finish_runtime_command(
                False,
                f"attachment_request_rejected:{result.message}",
                command_id,
            )
            return
        try:
            status_future = self.attachment_status_client.call_async(
                Trigger.Request()
            )
        except Exception as error:
            self.finish_runtime_command(
                False,
                f"attachment_status_send_failed:{error}",
                command_id,
            )
            return
        status_future.add_done_callback(
            lambda result_future, active_id=command_id: (
                self.attachment_status_result_callback(
                    result_future,
                    active_id,
                )
            )
        )

    def attachment_status_result_callback(self, future, command_id):
        try:
            status = future.result()
        except Exception as error:
            self.finish_runtime_command(
                False,
                f"attachment_status_failed:{error}",
                command_id,
            )
            return
        with self.runtime_lock:
            command = self.active_runtime_command
        if command is None:
            self.finish_runtime_command(
                False,
                "active_command_missing",
                command_id,
            )
            return
        succeeded, reason = validate_attachment_completion(
            command,
            True,
            "request accepted",
            bool(status.success),
            str(status.message),
        )
        self.finish_runtime_command(succeeded, reason, command_id)

    def activate_runtime_command(
        self,
        command,
        completion_callback,
        timeout_sec,
    ):
        deadline_ns = self.steady_clock.now().nanoseconds + int(
            float(timeout_sec) * 1_000_000_000
        )
        with self.runtime_lock:
            if self.active_runtime_command is not None:
                raise RuntimeError("physical_command_already_active")
            self.active_runtime_command = command
            self.active_completion_callback = completion_callback
            self.active_command_deadline_ns = deadline_ns

    def finish_runtime_command(
        self,
        succeeded,
        reason,
        expected_command_id=None,
    ):
        goal_handle = None
        with self.runtime_lock:
            command = self.active_runtime_command
            if command is None:
                return False
            if (
                expected_command_id is not None
                and command.command_id != expected_command_id
            ):
                return False
            callback = self.active_completion_callback
            if not succeeded:
                goal_handle = self.active_goal_handle
            self.active_runtime_command = None
            self.active_completion_callback = None
            self.active_command_deadline_ns = None
            self.active_goal_handle = None
        if goal_handle is not None:
            try:
                goal_handle.cancel_goal_async()
            except Exception:
                pass
        if callback is not None:
            callback(command, bool(succeeded), str(reason))
        return True

    def runtime_watchdog_callback(self):
        with self.runtime_lock:
            command = self.active_runtime_command
            deadline_ns = self.active_command_deadline_ns
        now_ns = self.steady_clock.now().nanoseconds
        if command is not None and deadline_ns is not None:
            if now_ns > deadline_ns:
                self.finish_runtime_command(
                    False,
                    f"physical_command_timeout:{command.command_id}",
                    command.command_id,
                )
                return
        if self.dry_run_machine is not None:
            if not self.dry_run_machine.terminal:
                try:
                    self.state_deadline.check(
                        self.dry_run_machine.state, now_ns,
                    )
                except StateDeadlineExceeded as error:
                    self.dry_run_machine.fail(str(error))
                    self.publish_dry_run_report()
            return
        with self.transition_lock:
            if self.runtime is None or self.runtime.terminal:
                return
            try:
                self.state_deadline.check(
                    self.runtime.state,
                    self.steady_clock.now().nanoseconds,
                )
            except StateDeadlineExceeded as error:
                with self.runtime_lock:
                    active = self.active_runtime_command
                if active is not None:
                    self.finish_runtime_command(
                        False,
                        str(error),
                        active.command_id,
                    )
                else:
                    self.safe_stop_runtime(str(error))

    def prepare_autonomous_runtime(self):
        """Prepare exactly one service-triggered physical run."""
        if not self.execution_enabled:
            raise RuntimeError("physical_execution_disabled")
        if self.runtime is not None:
            raise RuntimeError("single_runtime_already_prepared")
        run_id = f"day15-{uuid4().hex}"
        with self.transition_lock:
            self.runtime = AutonomousSingleRunRuntime(run_id)
            with self.runtime_lock:
                self.run_target_sequence_baseline = (
                    self.latest_global_target_sequence
                )
            self.runtime.start()
            self.sync_state_deadline()
            self.publish_runtime_report()
            return self.runtime.state

    def runtime_interfaces_ready(self):
        return all((
            self.arm_client is not None,
            self.arm_client.server_is_ready(),
            self.attachment_client is not None,
            self.attachment_client.service_is_ready(),
            self.attachment_status_client.service_is_ready(),
            self.vacuum_contact_client.service_is_ready(),
            self.d405_observation_client.service_is_ready(),
            self.global_detector_enable_client.service_is_ready(),
            self.d405_detector_enable_client.service_is_ready(),
        ))

    def request_close_range_perception(self):
        if self.close_range_perception_ready:
            return
        if self.perception_switch_future is not None:
            return
        if not (
            self.global_detector_enable_client.service_is_ready()
            and self.d405_detector_enable_client.service_is_ready()
        ):
            return
        request = SetBool.Request()
        request.data = False
        self.perception_switch_future = (
            self.global_detector_enable_client.call_async(request)
        )
        self.perception_switch_future.add_done_callback(
            self.global_detector_disabled_callback
        )

    def global_detector_disabled_callback(self, future):
        self.perception_switch_future = None
        if (
            self.runtime is None
            or self.runtime.terminal
            or self.runtime.state != PickState.START_D405_CHAIN
        ):
            return
        try:
            result = future.result()
            if not result.success:
                raise RuntimeError(result.message)
            request = SetBool.Request()
            request.data = True
            self.perception_switch_future = (
                self.d405_detector_enable_client.call_async(request)
            )
            self.perception_switch_future.add_done_callback(
                self.d405_detector_enabled_callback
            )
        except Exception as error:
            self.safe_stop_runtime(
                f"global_detector_disable_failed:{error}"
            )

    def d405_detector_enabled_callback(self, future):
        self.perception_switch_future = None
        if (
            self.runtime is None
            or self.runtime.terminal
            or self.runtime.state != PickState.START_D405_CHAIN
        ):
            return
        try:
            result = future.result()
            if not result.success:
                raise RuntimeError(result.message)
            with self.runtime_lock:
                self.d405_start_alignment_baseline = (
                    self.latest_d405_alignment_sequence
                )
            self.close_range_perception_ready = True
            self.publish_runtime_report()
        except Exception as error:
            self.safe_stop_runtime(
                f"d405_detector_enable_failed:{error}"
            )

    def runtime_progress_callback(self):
        if self.dry_run_machine is not None:
            self.progress_live_dry_run()
            return
        if self.runtime is None or self.runtime.terminal:
            return
        try:
            state = self.runtime.state
            if state == PickState.STARTUP_AUDIT:
                self.current_joint_snapshot()
                if not self.runtime_interfaces_ready():
                    return
                self.pass_runtime_gate("runtime_interfaces_ready")
            elif state in {
                PickState.WAIT_D435_TARGET,
                PickState.SELECT_GLOBAL_TARGET,
                PickState.VALIDATE_GLOBAL_TARGET,
            }:
                self.accept_global_target_evidence()
            elif state == PickState.WAIT_ROBOT_SETTLED:
                self.accept_pregrasp_settled_evidence()
            elif state == PickState.START_D405_CHAIN:
                if not self.close_range_perception_ready:
                    self.request_close_range_perception()
                elif (
                    self.d405_start_future is None
                    and self.fresh_d405_observation_ready()
                ):
                    self.request_d405_observation()
            elif state in {
                PickState.WAIT_D405_SPHERE,
                PickState.VERIFY_TARGET_HANDOFF,
            }:
                self.accept_d405_handoff_evidence()
            elif state in {
                PickState.XY_ALIGN,
                PickState.VERIFY_XY_GATE,
            }:
                self.accept_xy_proposal_evidence()
            elif state in {
                PickState.VERIFY_VACUUM_CONTACT,
                PickState.VERIFY_PLACE_POSE,
            } and self.contact_future is None:
                self.request_contact_gate()
            elif state in {
                PickState.VERIFY_ATTACHED,
                PickState.VERIFY_DETACHED,
                PickState.FINAL_AUDIT,
            } and self.status_gate_future is None:
                self.request_attachment_status_gate()
        except XYKinematicsRejected as error:
            self.safe_stop_runtime(f"xy_kinematics_rejected:{error}")
        except (EvidenceRejected, RuntimeError) as error:
            if self.runtime.state in {
                PickState.WAIT_D405_SPHERE,
                PickState.VERIFY_TARGET_HANDOFF,
            }:
                self.last_d405_gate_rejection = str(error)
            # Sensor evidence and ROS services are asynchronous. The state
            # deadline watchdog performs the bounded failure transition.
            return

    def publish_dry_run_report(self):
        report = self.dry_run_machine.report()
        report.update({
            "schema": "mushroom_pick_live_gate_dry_run/v1",
            "simulation_only": True,
            "live_ros_evidence": True,
            "physical_commands_sent": 0,
            "trajectory_commands_sent": 0,
            "attachment_commands_sent": 0,
        })
        self.publish(report)

    def progress_live_dry_run(self):
        machine = self.dry_run_machine
        if machine.terminal:
            return
        try:
            if machine.state == PickState.STARTUP_AUDIT:
                self.current_joint_snapshot()
                if not self.attachment_status_client.service_is_ready():
                    return
                machine.succeed("live_joint_state_and_follower_ready")
            elif machine.state in {
                PickState.WAIT_D435_TARGET,
                PickState.SELECT_GLOBAL_TARGET,
                PickState.VALIDATE_GLOBAL_TARGET,
            }:
                with self.runtime_lock:
                    payload = self.latest_global_target
                    received_ns = self.latest_global_target_received_ns
                    sequence = self.latest_global_target_sequence
                if payload is None or received_ns is None:
                    return
                now_ns = self.get_clock().now().nanoseconds
                lock = validate_fresh_global_target(
                    payload,
                    now_ns,
                    (self.steady_clock.now().nanoseconds - received_ns)
                    / 1_000_000_000.0,
                    sequence,
                    self.run_target_sequence_baseline,
                    machine.run_id,
                    self.lock_generation + 1,
                )
                if machine.state == PickState.WAIT_D435_TARGET:
                    machine.succeed("fresh_live_d435_target_received")
                elif machine.state == PickState.SELECT_GLOBAL_TARGET:
                    machine.succeed(
                        "live_mushroom_3_selected",
                        {"entity_name": lock.entity_name},
                    )
                else:
                    machine.succeed(
                        "live_global_target_validated",
                        {
                            "entity_name": lock.entity_name,
                            "handoff_id": lock.handoff_id,
                        },
                    )
            else:
                machine.fail("unexpected_live_dry_run_state")

            self.state_deadline.enter(
                machine.state, self.steady_clock.now().nanoseconds,
            )
            self.publish_dry_run_report()
        except (EvidenceRejected, RuntimeError):
            # The live target may be temporarily stale or invalid. The
            # state deadline produces a bounded, reported failure.
            return

    def safe_stop_runtime(self, reason, evidence=None):
        with self.transition_lock:
            if self.runtime is None:
                return
            if self.runtime.ledger.active_command is not None:
                self.runtime.timeout_active_command(reason)
            else:
                self.runtime.safe_stop(reason, evidence)
            self.sync_state_deadline()
            self.publish_runtime_report()

    def sync_state_deadline(self):
        if self.runtime is None:
            return
        self.state_deadline.enter(
            self.runtime.state,
            self.steady_clock.now().nanoseconds,
        )

    def accept_global_target_evidence(self):
        if self.runtime is None:
            raise RuntimeError("autonomous_runtime_not_prepared")
        if self.runtime.state not in {
            PickState.WAIT_D435_TARGET,
            PickState.SELECT_GLOBAL_TARGET,
            PickState.VALIDATE_GLOBAL_TARGET,
        }:
            raise RuntimeError("global_target_gate_not_active")
        with self.runtime_lock:
            payload = self.latest_global_target
            received_ns = self.latest_global_target_received_ns
            sequence = self.latest_global_target_sequence
        if payload is None or received_ns is None:
            raise EvidenceRejected("d435_target_missing")
        now_ns = self.get_clock().now().nanoseconds
        lock = validate_fresh_global_target(
            payload,
            now_ns,
            (self.steady_clock.now().nanoseconds - received_ns)
            / 1_000_000_000.0,
            sequence,
            self.run_target_sequence_baseline,
            self.runtime.machine.run_id,
            self.lock_generation + 1,
        )
        if self.runtime.state == PickState.WAIT_D435_TARGET:
            self.pass_runtime_gate("fresh_d435_target_received")
        if self.runtime.state == PickState.SELECT_GLOBAL_TARGET:
            self.pass_runtime_gate(
                "mushroom_3_selected",
                {"entity_name": lock.entity_name},
            )
        if self.runtime.state == PickState.VALIDATE_GLOBAL_TARGET:
            self.lock_generation += 1
            self.target_lock = lock
            self.pass_runtime_gate(
                "global_target_lock_created",
                {
                    "entity_name": lock.entity_name,
                    "handoff_id": lock.handoff_id,
                    "lock_generation": lock.lock_generation,
                },
            )
        return lock

    def accept_pregrasp_settled_evidence(self):
        if self.runtime is None:
            raise RuntimeError("autonomous_runtime_not_prepared")
        if self.runtime.state != PickState.WAIT_ROBOT_SETTLED:
            raise RuntimeError("robot_settled_gate_not_active")
        positions, velocities, age = self.current_joint_snapshot()
        report = validate_robot_settled(
            positions,
            velocities,
            PREGRASP_JOINTS,
            age,
        )
        self.pass_runtime_gate("pregrasp_robot_settled", report)
        return report

    def request_d405_observation(self):
        if self.runtime is None:
            raise RuntimeError("autonomous_runtime_not_prepared")
        if self.runtime.state not in {
            PickState.START_D405_CHAIN,
            PickState.XY_ALIGN,
        }:
            raise RuntimeError("d405_start_gate_not_active")
        if self.d405_start_future is not None:
            raise RuntimeError("d405_observation_request_active")
        if (
            self.d405_observation_client is None
            or not self.d405_observation_client.service_is_ready()
        ):
            raise RuntimeError("d405_observation_service_not_ready")
        self.d405_request_state = self.runtime.state
        with self.runtime_lock:
            self.d405_proposal_sequence_baseline = (
                self.latest_d405_proposal_sequence
            )
        self.d405_start_future = self.d405_observation_client.call_async(
            Trigger.Request()
        )
        self.d405_start_future.add_done_callback(
            self.d405_observation_started_callback
        )

    def fresh_d405_observation_ready(self):
        with self.runtime_lock:
            alignment = self.latest_d405_alignment
            received_ns = self.latest_d405_alignment_received_ns
            sequence = self.latest_d405_alignment_sequence
        if received_ns is None:
            return False
        age_sec = (
            self.steady_clock.now().nanoseconds - received_ns
        ) / 1_000_000_000.0
        return d405_observation_start_ready(
            alignment,
            age_sec,
            sequence,
            self.d405_start_alignment_baseline,
        )

    def d405_observation_started_callback(self, future):
        self.d405_start_future = None
        if self.runtime is None:
            return
        try:
            result = future.result()
        except Exception as error:
            self.safe_stop_runtime(f"d405_observation_start_failed:{error}")
            return
        requested_state = self.d405_request_state
        self.d405_request_state = None
        if self.runtime.state != requested_state:
            return
        if not result.success:
            self.safe_stop_runtime(
                f"d405_observation_start_rejected:{result.message}"
            )
            return
        if requested_state == PickState.START_D405_CHAIN:
            self.pass_runtime_gate(
                "d405_observation_started",
                {"message": str(result.message)},
            )
        else:
            self.publish_runtime_report()

    def d405_sphere_world(self, alignment):
        target_tcp = alignment.get("target_tcp")
        source_frame = alignment.get("target_frame")
        if not isinstance(target_tcp, dict):
            raise EvidenceRejected("d405_target_tcp_missing")
        point = PointStamped()
        point.header.frame_id = str(source_frame)
        try:
            point.point.x = float(target_tcp["x"])
            point.point.y = float(target_tcp["y"])
            point.point.z = float(target_tcp["z"])
        except (KeyError, TypeError, ValueError) as error:
            raise EvidenceRejected("d405_target_tcp_invalid") from error
        try:
            transform = self.tf_buffer.lookup_transform(
                "world",
                str(source_frame),
                Time(),
                Duration(seconds=0.2),
            )
            world = do_transform_point(point, transform)
        except TransformException as error:
            raise EvidenceRejected("d405_world_tf_unavailable") from error
        return {
            "x": float(world.point.x),
            "y": float(world.point.y),
            "z": float(world.point.z),
        }

    def accept_d405_handoff_evidence(self):
        if self.runtime is None:
            raise RuntimeError("autonomous_runtime_not_prepared")
        if self.target_lock is None:
            raise EvidenceRejected("global_target_lock_missing")
        if self.runtime.state not in {
            PickState.WAIT_D405_SPHERE,
            PickState.VERIFY_TARGET_HANDOFF,
        }:
            raise RuntimeError("d405_handoff_gate_not_active")
        with self.runtime_lock:
            alignment_history = tuple(self.d405_alignment_history)
            proposal = self.latest_d405_proposal
            proposal_sequence = self.latest_d405_proposal_sequence
        if proposal is None:
            raise EvidenceRejected("d405_evidence_missing")
        if proposal_sequence <= self.d405_proposal_sequence_baseline:
            raise EvidenceRejected("d405_proposal_not_new_after_observation")
        alignment = matching_d405_alignment(
            alignment_history,
            proposal,
            self.d405_start_alignment_baseline,
        )
        sphere_world = self.d405_sphere_world(alignment)
        try:
            report = validate_d405_chain(
                self.target_lock,
                alignment,
                proposal,
                sphere_world,
                self.get_clock().now().nanoseconds,
            )
        except EvidenceRejected as error:
            if self.first_d405_candidate_rejection is None:
                self.first_d405_candidate_rejection = str(error)
            raise
        self.last_d405_gate_rejection = None
        if self.runtime.state == PickState.WAIT_D405_SPHERE:
            self.pass_runtime_gate("stable_d405_sphere_received", report)
        if self.runtime.state == PickState.VERIFY_TARGET_HANDOFF:
            self.pass_runtime_gate("d435_d405_handoff_verified", report)
        return report

    def accept_xy_proposal_evidence(self):
        if self.runtime is None:
            raise RuntimeError("autonomous_runtime_not_prepared")
        if self.runtime.state not in {
            PickState.XY_ALIGN,
            PickState.VERIFY_XY_GATE,
        }:
            raise RuntimeError("xy_evidence_gate_not_active")
        if self.pending_xy_budget_gate is not None:
            raise EvidenceRejected("xy_budget_gate_query_active")
        with self.runtime_lock:
            proposal = self.latest_d405_proposal
            proposal_sequence = self.latest_d405_proposal_sequence
        if proposal is None:
            raise EvidenceRejected("xy_proposal_missing")
        if proposal_sequence <= self.d405_proposal_sequence_baseline:
            raise EvidenceRejected("xy_proposal_not_new_after_observation")
        report = validate_xy_proposal(
            proposal,
            self.get_clock().now().nanoseconds,
            self.consumed_proposal_ids,
        )
        if self.runtime.state == PickState.XY_ALIGN and report["aligned"]:
            self.request_xy_budget_gate(report)
        elif self.runtime.state == PickState.XY_ALIGN:
            if (
                self.runtime.ledger.counts[PickState.XY_ALIGN]
                >= STAGE_BY_STATE[PickState.XY_ALIGN].maximum_commands
            ):
                self.request_xy_budget_gate(report)
                return report
            if self.kinematics is None:
                raise RuntimeError("xy_kinematics_not_available")
            positions, velocities, age = self.current_joint_snapshot()
            report["robot"] = validate_robot_settled(
                positions,
                velocities,
                positions,
                age,
            )
            solution = solve_xy_joint_target(
                self.kinematics,
                positions,
                report["delta_tcp"],
                self.xy_kinematic_limits,
            )
            self.pending_xy_solution = {
                "proposal_id": report["proposal_id"],
                "solution": solution,
            }
            self.request_xy_detached_gate()
        elif self.runtime.state == PickState.VERIFY_XY_GATE:
            if not report["aligned"]:
                raise EvidenceRejected("xy_gate_not_within_deadband")
            self.pass_runtime_gate("xy_gate_verified", report)
        return report

    def request_xy_budget_gate(self, report):
        if self.contact_future is not None:
            raise RuntimeError("contact_query_active")
        if not self.vacuum_contact_client.service_is_ready():
            raise RuntimeError("vacuum_contact_service_not_ready")
        self.pending_xy_budget_gate = report
        self.contact_future = self.vacuum_contact_client.call_async(
            Trigger.Request()
        )
        self.contact_future.add_done_callback(
            self.xy_budget_gate_result_callback
        )

    def xy_budget_gate_result_callback(self, future):
        self.contact_future = None
        report = self.pending_xy_budget_gate
        self.pending_xy_budget_gate = None
        if (
            self.runtime is None
            or self.runtime.state != PickState.XY_ALIGN
            or report is None
        ):
            return
        try:
            result = future.result()
            vacuum_report = decode_json_object(
                result.message, "xy_vacuum_geometry"
            )
            positions, velocities, age = self.current_joint_snapshot()
            route_start = validate_robot_settled(
                positions,
                velocities,
                POST_XY_JOINTS,
                age,
            )
            gate = validate_day13_xy_budget_gate(
                report, vacuum_report
            )
            gate["verified_z_route_start"] = route_start
            reason = (
                "xy_deadband_and_independent_geometry_verified"
                if report["aligned"]
                else "day13_five_step_xy_budget_reached"
            )
            self.finish_runtime_xy_alignment(reason, gate)
            self.pass_runtime_gate(
                "xy_independent_vacuum_planar_verified", gate
            )
        except Exception as error:
            self.safe_stop_runtime(f"xy_budget_gate_failed:{error}")

    def request_xy_detached_gate(self):
        if self.pending_xy_solution is None:
            raise RuntimeError("xy_solution_missing")
        if self.status_gate_future is not None:
            raise RuntimeError("attachment_status_query_active")
        if not self.attachment_status_client.service_is_ready():
            raise RuntimeError("attachment_status_service_not_ready")
        self.status_gate_future = self.attachment_status_client.call_async(
            Trigger.Request()
        )
        self.status_gate_future.add_done_callback(
            self.xy_detached_gate_result_callback
        )

    def xy_detached_gate_result_callback(self, future):
        self.status_gate_future = None
        pending = self.pending_xy_solution
        self.pending_xy_solution = None
        if (
            self.runtime is None
            or self.runtime.state != PickState.XY_ALIGN
            or pending is None
        ):
            return
        try:
            result = future.result()
            validate_attachment_status(
                False,
                bool(result.success),
                str(result.message),
            )
            solution = pending["solution"]
            self.dispatch_current_runtime_command(
                solution["target_joints"],
                solution["trajectory_duration_sec"],
            )
        except Exception as error:
            self.safe_stop_runtime(f"xy_detached_gate_failed:{error}")

    def request_contact_gate(self):
        if self.runtime is None:
            raise RuntimeError("autonomous_runtime_not_prepared")
        if self.runtime.state not in {
            PickState.VERIFY_VACUUM_CONTACT,
            PickState.VERIFY_PLACE_POSE,
        }:
            raise RuntimeError("contact_gate_not_active")
        if self.contact_future is not None:
            raise RuntimeError("contact_query_active")
        if (
            self.vacuum_contact_client is None
            or not self.vacuum_contact_client.service_is_ready()
        ):
            raise RuntimeError("vacuum_contact_service_not_ready")
        gate_state = self.runtime.state
        self.contact_future = self.vacuum_contact_client.call_async(
            Trigger.Request()
        )
        self.contact_future.add_done_callback(
            lambda result_future, state=gate_state: (
                self.contact_gate_result_callback(result_future, state)
            )
        )

    def contact_gate_result_callback(self, future, gate_state):
        self.contact_future = None
        if self.runtime is None or self.runtime.state != gate_state:
            return
        try:
            result = future.result()
            report = validate_vacuum_contact_response(
                bool(result.success),
                str(result.message),
                gate_state == PickState.VERIFY_PLACE_POSE,
            )
        except Exception as error:
            self.safe_stop_runtime(f"contact_gate_failed:{error}")
            return
        self.pass_runtime_gate(report["reason"], report)

    def request_attachment_status_gate(self):
        if self.runtime is None:
            raise RuntimeError("autonomous_runtime_not_prepared")
        if self.runtime.state not in {
            PickState.VERIFY_ATTACHED,
            PickState.VERIFY_DETACHED,
            PickState.FINAL_AUDIT,
        }:
            raise RuntimeError("attachment_status_gate_not_active")
        if self.status_gate_future is not None:
            raise RuntimeError("attachment_status_query_active")
        if (
            self.attachment_status_client is None
            or not self.attachment_status_client.service_is_ready()
        ):
            raise RuntimeError("attachment_status_service_not_ready")
        gate_state = self.runtime.state
        self.status_gate_future = self.attachment_status_client.call_async(
            Trigger.Request()
        )
        self.status_gate_future.add_done_callback(
            lambda result_future, state=gate_state: (
                self.attachment_status_gate_result_callback(
                    result_future,
                    state,
                )
            )
        )

    def attachment_status_gate_result_callback(self, future, gate_state):
        self.status_gate_future = None
        if self.runtime is None or self.runtime.state != gate_state:
            return
        expected_attached = gate_state == PickState.VERIFY_ATTACHED
        try:
            result = future.result()
            report = validate_attachment_status(
                expected_attached,
                bool(result.success),
                str(result.message),
            )
            if gate_state == PickState.FINAL_AUDIT:
                positions, velocities, age = self.current_joint_snapshot()
                report["home"] = validate_robot_settled(
                    positions,
                    velocities,
                    HOME_JOINTS,
                    age,
                )
        except Exception as error:
            self.safe_stop_runtime(f"attachment_status_gate_failed:{error}")
            return
        self.pass_runtime_gate(report["reason"], report)

    def pass_runtime_gate(self, reason, evidence=None):
        with self.transition_lock:
            if self.runtime is None:
                raise RuntimeError("autonomous_runtime_not_prepared")
            state = self.runtime.pass_current_gate(reason, evidence)
            self.sync_state_deadline()
            self.publish_runtime_report()
        if state in PHYSICAL_OPERATION_STATES and state != PickState.XY_ALIGN:
            self.dispatch_current_runtime_command()
        return state

    def dispatch_current_runtime_command(
        self,
        dynamic_xy_target=None,
        dynamic_xy_duration_sec=5.0,
    ):
        if self.runtime is None:
            raise RuntimeError("autonomous_runtime_not_prepared")
        command = self.runtime.issue_current_command(
            dynamic_xy_target,
            dynamic_xy_duration_sec,
        )
        if command.command_kind == CommandKind.TRAJECTORY:
            self.dispatch_trajectory(command, self.runtime_command_completed)
        else:
            self.dispatch_attachment(command, self.runtime_command_completed)
        self.publish_runtime_report()
        return command

    def finish_runtime_xy_alignment(self, reason, evidence=None):
        with self.transition_lock:
            if self.runtime is None:
                raise RuntimeError("autonomous_runtime_not_prepared")
            state = self.runtime.finish_xy_alignment(reason, evidence)
            self.sync_state_deadline()
            self.publish_runtime_report()
            return state

    def runtime_command_completed(self, command, succeeded, reason):
        with self.transition_lock:
            if self.runtime is None or command is None:
                return
            try:
                state = self.runtime.complete_command(
                    command.command_id,
                    succeeded,
                    reason,
                )
            except Exception as error:
                self.runtime.safe_stop(f"runtime_completion_failed:{error}")
                state = self.runtime.state
            self.sync_state_deadline()
            self.publish_runtime_report()
        if (
            succeeded
            and not self.runtime.terminal
            and state in PHYSICAL_OPERATION_STATES
            and state != PickState.XY_ALIGN
        ):
            self.dispatch_current_runtime_command()
        elif (
            succeeded
            and not self.runtime.terminal
            and state == PickState.XY_ALIGN
        ):
            try:
                self.request_d405_observation()
            except Exception as error:
                self.safe_stop_runtime(
                    f"xy_reobservation_start_failed:{error}"
                )

    def publish_runtime_report(self):
        report = self.runtime.report()
        if self.last_d405_gate_rejection is not None:
            report["last_d405_gate_rejection"] = self.last_d405_gate_rejection
        if self.first_d405_candidate_rejection is not None:
            report["first_d405_candidate_rejection"] = (
                self.first_d405_candidate_rejection
            )
        report["trajectory_commands_sent"] = self.trajectory_commands_sent
        report["attachment_commands_sent"] = self.attachment_commands_sent
        report["physical_commands_sent"] = (
            self.trajectory_commands_sent + self.attachment_commands_sent
        )
        self.publish(report)

    def publish(self, payload: dict) -> None:
        self.report_snapshot.update(payload)
        message = String()
        message.data = json.dumps(payload, ensure_ascii=False, default=str)
        self.state_publisher.publish(message)

    def run_callback(self, request, response):
        del request
        current = json.loads(self.report_snapshot.message())
        if current.get("state") != "IDLE":
            response.success = False
            response.message = "single_run_already_consumed"
            return response
        if self.execution_enabled:
            try:
                state = self.prepare_autonomous_runtime()
            except Exception as error:
                response.success = False
                response.message = f"run_start_failed:{error}"
                return response
            response.success = True
            response.message = json.dumps({
                "accepted": True,
                "state": state.value,
                "execution_enabled": True,
            })
            return response
        self.dry_run_machine = PickStateMachineCore(
            mode=DryRunMode.LIVE_GATE,
            run_id_factory=lambda: f"day19-live-dry-{uuid4().hex}",
        )
        with self.runtime_lock:
            self.run_target_sequence_baseline = (
                self.latest_global_target_sequence
            )
        state = self.dry_run_machine.start("live_ros_dry_run_requested")
        self.state_deadline.enter(
            state, self.steady_clock.now().nanoseconds,
        )
        self.publish_dry_run_report()
        response.success = True
        response.message = json.dumps({
            "accepted": True,
            "state": state.value,
            "execution_enabled": False,
            "completion_state": PickState.DRY_RUN_COMPLETE.value,
        })
        return response

    def report_callback(self, request, response):
        del request
        response.success = True
        response.message = self.report_snapshot.message()
        return response


def main(args=None):
    rclpy.init(args=args)
    node = Day15AutonomousPick()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
