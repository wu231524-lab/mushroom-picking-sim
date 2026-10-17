import json
import math
from threading import Event
import time
from uuid import uuid4

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from std_msgs.msg import String
from std_srvs.srv import Trigger


MAX_SYNC_DELTA_SEC = 0.20


class D405StepProposal(Node):

    def __init__(self):
        super().__init__('d405_step_proposal')

        self.declare_parameter(
            'input_topic',
            '/d405/mushroom_pick/alignment_error'
        )
        self.declare_parameter(
            'output_topic',
            '/d405/mushroom_pick/step_proposal'
        )
        self.declare_parameter(
            'start_service',
            '/d405/mushroom_pick/start_observation'
        )
        self.declare_parameter(
            'get_last_service',
            '/d405/mushroom_pick/get_last_proposal'
        )
        self.declare_parameter(
            'reset_session_service',
            '/d405/mushroom_pick/reset_session'
        )
        self.declare_parameter(
            'alignment_reset_service',
            '/d405/mushroom_pick/reset_alignment_session'
        )
        self.declare_parameter('reset_ack_timeout_sec', 5.0)
        self.declare_parameter(
            'expected_target_frame',
            'fer_hand_tcp'
        )
        self.declare_parameter(
            'required_observations',
            2
        )
        self.declare_parameter(
            'observation_timeout_sec',
            60.0
        )
        self.declare_parameter(
            'max_step_m',
            0.005
        )
        self.declare_parameter(
            'deadband_m',
            0.002
        )
        self.declare_parameter(
            'max_spread_m',
            0.003
        )
        self.declare_parameter(
            'allow_transient_sync_miss',
            False
        )

        self.input_topic = self.get_parameter(
            'input_topic'
        ).value
        self.output_topic = self.get_parameter(
            'output_topic'
        ).value
        self.start_service_name = self.get_parameter(
            'start_service'
        ).value
        self.get_last_service_name = self.get_parameter(
            'get_last_service'
        ).value
        self.reset_session_service_name = self.get_parameter(
            'reset_session_service'
        ).value
        self.alignment_reset_service_name = self.get_parameter(
            'alignment_reset_service'
        ).value
        self.reset_ack_timeout_sec = float(
            self.get_parameter('reset_ack_timeout_sec').value
        )
        self.expected_target_frame = (
            self.get_parameter(
                'expected_target_frame'
            ).value
        )

        self.required_observations = int(
            self.get_parameter(
                'required_observations'
            ).value
        )
        self.observation_timeout_sec = float(
            self.get_parameter(
                'observation_timeout_sec'
            ).value
        )
        self.max_step_m = float(
            self.get_parameter('max_step_m').value
        )
        self.deadband_m = float(
            self.get_parameter('deadband_m').value
        )
        self.max_spread_m = float(
            self.get_parameter(
                'max_spread_m'
            ).value
        )
        self.allow_transient_sync_miss = bool(
            self.get_parameter(
                'allow_transient_sync_miss'
            ).value
        )

        if self.required_observations < 2:
            self.required_observations = 2

        if self.observation_timeout_sec <= 0.0:
            self.observation_timeout_sec = 60.0

        if self.max_step_m <= 0.0:
            self.max_step_m = 0.005

        if self.deadband_m < 0.0:
            self.deadband_m = 0.0

        if self.max_spread_m < 0.0:
            self.max_spread_m = 0.0
        if not 0.1 <= self.reset_ack_timeout_sec <= 10.0:
            raise RuntimeError('reset_ack_timeout_sec_out_of_range')

        self.observing = False
        self.observations = []
        self.observation_started_ns = None
        self.observation_started_monotonic_ns = None
        self.start_after_stamp_ns = None
        self.last_accepted_stamp_ns = None
        self.last_seen_stamp_ns = None
        self.last_output_json = None
        self.current_session_generation = None
        self.current_lock_id = None
        self.observation_session_generation = None
        self.observation_lock_id = None
        self.last_alignment_valid = False
        self.reset_in_progress = False
        self.reset_requires_new_lock = False

        self.mutation_callback_group = MutuallyExclusiveCallbackGroup()
        self.reset_client_callback_group = ReentrantCallbackGroup()

        proposal_qos = QoSProfile(depth=1)
        proposal_qos.reliability = (
            ReliabilityPolicy.RELIABLE
        )
        proposal_qos.durability = (
            DurabilityPolicy.TRANSIENT_LOCAL
        )

        self.output_publisher = self.create_publisher(
            String,
            self.output_topic,
            proposal_qos
        )
        self.input_subscription = self.create_subscription(
            String,
            self.input_topic,
            self.alignment_callback,
            10
        )
        self.start_service = self.create_service(
            Trigger,
            self.start_service_name,
            self.start_observation_callback,
            callback_group=self.mutation_callback_group
        )
        self.get_last_service = self.create_service(
            Trigger,
            self.get_last_service_name,
            self.get_last_proposal_callback
        )
        self.reset_session_service = self.create_service(
            Trigger,
            self.reset_session_service_name,
            self.reset_session_callback,
            callback_group=self.mutation_callback_group
        )
        self.alignment_reset_client = self.create_client(
            Trigger,
            self.alignment_reset_service_name,
            callback_group=self.reset_client_callback_group
        )
        self.timeout_timer = self.create_timer(
            0.5,
            self.timeout_callback
        )

        self.get_logger().info(
            f'输入局部误差: {self.input_topic}'
        )
        self.get_logger().info(
            f'输出小步建议: {self.output_topic}'
        )
        self.get_logger().info(
            f'开始观察服务: {self.start_service_name}'
        )
        self.get_logger().info(
            f'查询最后建议服务: '
            f'{self.get_last_service_name}'
        )
        self.get_logger().info(
            'Dry-run 模式固定启用，'
            '不会发送机械臂命令'
        )

    @staticmethod
    def stamp_to_nanoseconds(stamp):
        if not isinstance(stamp, dict):
            return None

        try:
            sec = int(stamp['sec'])
            nanosec = int(stamp['nanosec'])
        except (KeyError, TypeError, ValueError):
            return None

        if (
            sec < 0
            or nanosec < 0
            or nanosec >= 1_000_000_000
        ):
            return None

        return sec * 1_000_000_000 + nanosec

    def publish_data(self, data):
        data = dict(data)
        data.setdefault(
            'session_generation',
            self.current_session_generation
        )
        data.setdefault('lock_id', self.current_lock_id)
        serialized = json.dumps(
            data,
            ensure_ascii=False
        )
        self.last_output_json = serialized

        message = String()
        message.data = serialized
        self.output_publisher.publish(message)

    def publish_collecting(self):
        self.publish_data({
            'valid': False,
            'reason': 'collecting_new_observations',
            'observation_active': True,
            'observation_count': len(
                self.observations
            ),
            'required_observations': (
                self.required_observations
            ),
            'proposal_id': None,
            'generated_stamp': None,
            'upstream_sync_evidence': [],
            'single_step_only': True,
            'execute_allowed': False,
            'proposed_delta_tcp': None
        })

    def publish_transient_sync_miss(
        self,
        input_data
    ):
        self.publish_data({
            'valid': False,
            'reason': (
                'transient_sync_miss_ignored; '
                'collecting_new_observations'
            ),
            'observation_active': True,
            'observation_count': len(
                self.observations
            ),
            'required_observations': (
                self.required_observations
            ),
            'proposal_id': None,
            'generated_stamp': None,
            'upstream_sync_evidence': [
                item['upstream_sync']
                for item in self.observations
            ],
            'transient_sync_miss': (
                input_data.get('upstream_sync')
            ),
            'stamp': input_data.get(
                'detection_stamp',
                input_data.get('stamp')
            ),
            'single_step_only': True,
            'execute_allowed': False,
            'proposed_delta_tcp': None
        })

        self.get_logger().warning(
            '忽略一次同步超限样本，'
            '继续等待严格同步观测'
        )

    def stop_with_invalid(
        self,
        reason,
        input_data=None
    ):
        observation_count = len(
            self.observations
        )

        self.observing = False
        self.observations = []
        self.observation_started_ns = None
        self.observation_started_monotonic_ns = None
        self.last_accepted_stamp_ns = None
        self.observation_session_generation = None
        self.observation_lock_id = None

        output = {
            'valid': False,
            'reason': reason,
            'observation_active': False,
            'observation_count': observation_count,
            'required_observations': (
                self.required_observations
            ),
            'proposal_id': None,
            'generated_stamp': None,
            'upstream_sync_evidence': [],
            'single_step_only': True,
            'execute_allowed': False,
            'proposed_delta_tcp': None
        }

        if isinstance(input_data, dict):
            output['stamp'] = input_data.get(
                'detection_stamp',
                input_data.get('stamp')
            )
            output['upstream_sync'] = (
                input_data.get('upstream_sync')
            )

        self.publish_data(output)

        self.get_logger().warning(
            f'小步建议无效: {reason}'
        )

    def get_last_proposal_callback(
        self,
        request,
        response
    ):
        del request

        if self.last_output_json is None:
            response.success = False
            response.message = 'no_proposal_available'
            return response

        response.success = True
        response.message = self.last_output_json
        return response

    def start_observation_callback(
        self,
        request,
        response
    ):
        del request

        if self.reset_in_progress:
            response.success = False
            response.message = 'reset_in_progress'
            return response

        if self.reset_requires_new_lock:
            response.success = False
            response.message = 'reset_acknowledged_waiting_for_new_lock'
            return response

        if self.observing:
            response.success = False
            response.message = (
                'observation_already_active'
            )
            return response

        if self.current_session_generation is None:
            response.success = False
            response.message = 'session_generation_unavailable'
            return response

        if (
            not isinstance(self.current_lock_id, str)
            or not self.current_lock_id
        ):
            response.success = False
            response.message = 'session_lock_unavailable'
            return response

        if not self.last_alignment_valid:
            response.success = False
            response.message = 'latest_alignment_not_valid'
            return response

        self.observing = True
        self.observations = []
        self.observation_started_ns = (
            self.get_clock().now().nanoseconds
        )
        self.observation_started_monotonic_ns = time.monotonic_ns()
        self.start_after_stamp_ns = (
            self.last_seen_stamp_ns
        )
        self.last_accepted_stamp_ns = None
        self.observation_session_generation = (
            self.current_session_generation
        )
        self.observation_lock_id = self.current_lock_id

        response.success = True
        response.message = (
            'observation_started; waiting for '
            f'{self.required_observations} '
            'strictly newer synchronized results'
        )

        self.get_logger().info(
            response.message
        )

        self.publish_collecting()
        return response

    def reset_session_callback(self, request, response):
        del request
        if self.reset_in_progress:
            response.success = False
            response.message = 'reset_in_progress'
            return response
        if self.observing:
            response.success = False
            response.message = 'observation_active'
            return response
        if self.reset_requires_new_lock:
            response.success = False
            response.message = 'duplicate_reset_waiting_for_new_lock'
            return response

        generation_before_reset = (
            self.current_session_generation
        )
        self.reset_in_progress = True
        request_id = 'day18-reset-' + uuid4().hex
        try:
            if not self.alignment_reset_client.wait_for_service(
                timeout_sec=self.reset_ack_timeout_sec
            ):
                response.success = False
                response.message = 'alignment_reset_service_unavailable'
                return response

            future = self.alignment_reset_client.call_async(
                Trigger.Request()
            )
            completed = Event()
            future.add_done_callback(lambda unused: completed.set())
            if not completed.wait(self.reset_ack_timeout_sec):
                response.success = False
                response.message = 'alignment_reset_ack_timeout'
                return response
            try:
                alignment_response = future.result()
            except Exception as error:
                response.success = False
                response.message = (
                    'alignment_reset_call_failed:'
                    f'{type(error).__name__}:{error}'
                )
                return response
            if alignment_response is None or not alignment_response.success:
                reason = (
                    alignment_response.message
                    if alignment_response is not None
                    else 'empty_alignment_reset_response'
                )
                response.success = False
                response.message = 'alignment_reset_rejected:' + reason
                return response
            try:
                alignment_ack = json.loads(alignment_response.message)
                new_generation = int(alignment_ack['new_generation'])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                response.success = False
                response.message = 'alignment_reset_ack_invalid'
                return response
            cleared = alignment_ack.get('cleared')
            if (
                alignment_ack.get('schema')
                != 'day18.d405_alignment_reset.v1'
                or alignment_ack.get('accepted') is not True
                or not isinstance(cleared, dict)
                or cleared.get('lock') is not True
                or cleared.get('proposal') is not False
                or cleared.get('observation') is not False
                or new_generation <= 0
                or (
                    generation_before_reset is not None
                    and new_generation <= generation_before_reset
                )
                or (
                    self.current_session_generation is not None
                    and self.current_session_generation > new_generation
                )
            ):
                response.success = False
                response.message = 'alignment_reset_ack_invalid'
                return response

            self.observing = False
            self.observations = []
            self.observation_started_ns = None
            self.observation_started_monotonic_ns = None
            self.start_after_stamp_ns = None
            self.last_accepted_stamp_ns = None
            self.last_seen_stamp_ns = None
            self.current_session_generation = new_generation
            self.current_lock_id = None
            self.observation_session_generation = None
            self.observation_lock_id = None
            self.last_alignment_valid = False
            self.last_output_json = None
            self.reset_requires_new_lock = True

            ack = {
                'schema': 'day18.d405_session_reset.v1',
                'accepted': True,
                'reason': 'session_reset_complete_with_proposal_ack',
                'request_id': request_id,
                'new_generation': new_generation,
                'alignment_ack': alignment_ack,
                'proposal_ack': True,
                'cleared': {
                    'lock': True,
                    'proposal': True,
                    'observation': True,
                },
            }
            self.publish_data({
                'valid': False,
                'reason': 'session_reset_acknowledged',
                'observation_active': False,
                'observation_count': 0,
                'required_observations': self.required_observations,
                'proposal_id': None,
                'generated_stamp': None,
                'upstream_sync_evidence': [],
                'single_step_only': True,
                'execute_allowed': False,
                'proposed_delta_tcp': None,
                'reset_ack': ack,
            })
            response.success = True
            response.message = json.dumps(ack, sort_keys=True)
            return response
        finally:
            self.reset_in_progress = False

    def timeout_callback(self):
        if not self.observing:
            return

        elapsed_sec = (
            time.monotonic_ns()
            - self.observation_started_monotonic_ns
        ) / 1_000_000_000.0

        if elapsed_sec > self.observation_timeout_sec:
            self.stop_with_invalid(
                'observation_timeout'
            )

    def alignment_callback(self, message):
        try:
            input_data = json.loads(message.data)
        except json.JSONDecodeError:
            if self.observing:
                self.stop_with_invalid(
                    'invalid_alignment_json'
                )
            return

        if not isinstance(input_data, dict):
            if self.observing:
                self.stop_with_invalid(
                    'alignment_is_not_object'
                )
            return

        tracking = input_data.get('target_tracking')
        generation = input_data.get('session_generation')
        if generation is None and isinstance(tracking, dict):
            generation = tracking.get('session_generation')
        lock_id = (
            tracking.get('lock_id')
            if isinstance(tracking, dict)
            else None
        )

        if not isinstance(generation, int) or generation < 0:
            if self.observing:
                self.stop_with_invalid(
                    'session_generation_invalid',
                    input_data
                )
            return

        if (
            self.current_session_generation is not None
            and generation < self.current_session_generation
        ):
            return

        if generation != self.current_session_generation:
            session_changed_while_observing = self.observing
            if session_changed_while_observing:
                self.stop_with_invalid(
                    'session_changed_during_observation',
                    input_data
                )
            self.current_session_generation = generation
            self.current_lock_id = None
            self.last_alignment_valid = False
            self.observations = []
            self.last_accepted_stamp_ns = None
            self.start_after_stamp_ns = None
            self.last_output_json = None
            if not session_changed_while_observing:
                self.publish_data({
                    'valid': False,
                    'reason': 'session_reset',
                    'observation_active': False,
                    'observation_count': 0,
                    'required_observations': self.required_observations,
                    'proposal_id': None,
                    'generated_stamp': None,
                    'upstream_sync_evidence': [],
                    'single_step_only': True,
                    'execute_allowed': False,
                    'proposed_delta_tcp': None,
                })

        if isinstance(lock_id, str) and lock_id:
            self.current_lock_id = lock_id
            if generation == self.current_session_generation:
                self.reset_requires_new_lock = False

        self.last_alignment_valid = input_data.get('valid') is True

        stamp = input_data.get(
            'detection_stamp',
            input_data.get('stamp')
        )
        stamp_ns = self.stamp_to_nanoseconds(
            stamp
        )

        if stamp_ns is not None:
            if (
                self.last_seen_stamp_ns is None
                or stamp_ns > self.last_seen_stamp_ns
            ):
                self.last_seen_stamp_ns = stamp_ns

        if not self.observing:
            return

        if generation != self.observation_session_generation:
            self.stop_with_invalid(
                'session_changed_during_observation',
                input_data
            )
            return

        if lock_id != self.observation_lock_id:
            self.stop_with_invalid(
                'target_lock_changed_during_observation',
                input_data
            )
            return

        if stamp_ns is None:
            self.stop_with_invalid(
                'missing_or_invalid_stamp',
                input_data
            )
            return

        if (
            self.start_after_stamp_ns is not None
            and stamp_ns <= self.start_after_stamp_ns
        ):
            return

        if (
            self.last_accepted_stamp_ns is not None
            and stamp_ns <= self.last_accepted_stamp_ns
        ):
            self.stop_with_invalid(
                'non_increasing_stamp',
                input_data
            )
            return

        if input_data.get('valid') is not True:
            reason = input_data.get(
                'reason',
                'unknown'
            )

            if (
                self.allow_transient_sync_miss
                and reason
                == (
                    'upstream_sync_invalid: '
                    'sync_delta_exceeded'
                )
            ):
                self.publish_transient_sync_miss(
                    input_data
                )
                return

            self.stop_with_invalid(
                f'alignment_invalid: {reason}',
                input_data
            )
            return

        if (
            input_data.get('target_frame')
            != self.expected_target_frame
        ):
            self.stop_with_invalid(
                'unexpected_target_frame',
                input_data
            )
            return

        upstream_sync = input_data.get(
            'upstream_sync'
        )

        if (
            not isinstance(upstream_sync, dict)
            or upstream_sync.get('valid') is not True
        ):
            self.stop_with_invalid(
                'upstream_sync_not_valid',
                input_data
            )
            return

        try:
            sync_delta_sec = abs(float(upstream_sync['delta_sec']))
        except (KeyError, TypeError, ValueError):
            self.stop_with_invalid(
                'upstream_sync_delta_invalid',
                input_data
            )
            return
        if (
            not math.isfinite(sync_delta_sec)
            or sync_delta_sec > MAX_SYNC_DELTA_SEC
        ):
            self.stop_with_invalid(
                'upstream_sync_delta_exceeded',
                input_data
            )
            return

        alignment_error = input_data.get(
            'alignment_error'
        )

        if not isinstance(alignment_error, dict):
            self.stop_with_invalid(
                'alignment_error_missing',
                input_data
            )
            return

        try:
            error_x = float(
                alignment_error[
                    'lateral_error_x'
                ]
            )
            error_y = float(
                alignment_error[
                    'longitudinal_error_y'
                ]
            )
            depth_distance = float(
                alignment_error[
                    'depth_distance'
                ]
            )
        except (KeyError, TypeError, ValueError):
            self.stop_with_invalid(
                'alignment_error_fields_invalid',
                input_data
            )
            return

        if not all(
            math.isfinite(value)
            for value in (
                error_x,
                error_y,
                depth_distance
            )
        ):
            self.stop_with_invalid(
                'non_finite_alignment_error',
                input_data
            )
            return

        if depth_distance <= 0.0:
            self.stop_with_invalid(
                'non_positive_depth_distance',
                input_data
            )
            return

        self.observations.append({
            'stamp': stamp,
            'stamp_ns': stamp_ns,
            'error_x': error_x,
            'error_y': error_y,
            'depth_distance': depth_distance,
            'upstream_sync': upstream_sync
        })
        self.last_accepted_stamp_ns = stamp_ns

        if (
            len(self.observations)
            < self.required_observations
        ):
            self.publish_collecting()
            return

        mean_x = sum(
            item['error_x']
            for item in self.observations
        ) / len(self.observations)
        mean_y = sum(
            item['error_y']
            for item in self.observations
        ) / len(self.observations)
        mean_depth = sum(
            item['depth_distance']
            for item in self.observations
        ) / len(self.observations)

        mean_planar_error = math.hypot(
            mean_x,
            mean_y
        )

        maximum_spread = max(
            math.hypot(
                item['error_x'] - mean_x,
                item['error_y'] - mean_y
            )
            for item in self.observations
        )

        if maximum_spread > self.max_spread_m:
            self.stop_with_invalid(
                'observation_spread_exceeded',
                input_data
            )
            return

        if mean_planar_error > self.deadband_m:
            for item in self.observations:
                direction_dot = (
                    item['error_x'] * mean_x
                    + item['error_y'] * mean_y
                )

                if direction_dot <= 0.0:
                    self.stop_with_invalid(
                        'observation_direction_changed',
                        input_data
                    )
                    return

        if mean_planar_error <= self.deadband_m:
            reason = 'within_deadband'
            delta_x = 0.0
            delta_y = 0.0
        else:
            reason = 'stable_dry_run_proposal'
            scale = min(
                1.0,
                self.max_step_m
                / mean_planar_error
            )
            delta_x = mean_x * scale
            delta_y = mean_y * scale

        first_stamp = self.observations[0][
            'stamp'
        ]
        last_stamp = self.observations[-1][
            'stamp'
        ]
        observation_count = len(
            self.observations
        )

        generated_ns = (
            self.get_clock().now().nanoseconds
        )
        generated_stamp = {
            'sec': generated_ns // 1_000_000_000,
            'nanosec': generated_ns % 1_000_000_000
        }
        proposal_id = (
            f'g{self.current_session_generation}-'
            f'{int(first_stamp["sec"])}.'
            f'{int(first_stamp["nanosec"]):09d}-'
            f'{int(last_stamp["sec"])}.'
            f'{int(last_stamp["nanosec"]):09d}-'
            f'{generated_ns}'
        )
        sync_evidence = [
            item['upstream_sync']
            for item in self.observations
        ]

        output = {
            'valid': True,
            'reason': reason,
            'proposal_id': proposal_id,
            'generated_stamp': generated_stamp,
            'observation_active': False,
            'source_frame': self.expected_target_frame,
            'upstream_sync_evidence': sync_evidence,
            'single_step_only': True,
            'stamp_first': first_stamp,
            'stamp_last': last_stamp,
            'observation_count': observation_count,
            'required_observations': (
                self.required_observations
            ),
            'mean_alignment_error': {
                'x': mean_x,
                'y': mean_y,
                'planar': mean_planar_error,
                'depth_distance': mean_depth
            },
            'maximum_spread': maximum_spread,
            'proposed_delta_tcp': {
                'x': delta_x,
                'y': delta_y,
                'z': 0.0
            },
            'max_step_m': self.max_step_m,
            'deadband_m': self.deadband_m,
            'execute_allowed': False
        }

        self.observing = False
        self.observations = []
        self.observation_started_ns = None
        self.observation_started_monotonic_ns = None
        self.last_accepted_stamp_ns = None
        self.observation_session_generation = None
        self.observation_lock_id = None

        self.publish_data(output)

        self.get_logger().info(
            'Dry-run 建议生成: '
            f'delta=({delta_x:.4f}, '
            f'{delta_y:.4f}, 0.0000) m, '
            'execute_allowed=false'
        )


def main(args=None):
    rclpy.init(args=args)
    node = D405StepProposal()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.remove_node(node)
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
