import json
import math

import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_geometry_msgs import do_transform_point
from tf2_ros import Buffer, TransformException, TransformListener


class D405TcpAlignment(Node):

    def __init__(self):
        super().__init__('d405_tcp_alignment')

        self.declare_parameter(
            'input_topic',
            '/d405/mushroom_detection/positions_3d'
        )
        self.declare_parameter(
            'output_topic',
            '/d405/mushroom_pick/alignment_error'
        )
        self.declare_parameter(
            'expected_source_frame',
            'd405_optical_frame'
        )
        self.declare_parameter(
            'target_frame',
            'fer_hand_tcp'
        )
        self.declare_parameter('image_center_u', 320.0)
        self.declare_parameter('image_center_v', 240.0)
        self.declare_parameter(
            'max_center_distance_px',
            120.0
        )
        self.declare_parameter('min_depth', 0.07)
        self.declare_parameter('max_depth', 0.50)
        self.declare_parameter('desired_tcp_x', 0.0)
        self.declare_parameter('desired_tcp_y', 0.0)
        self.declare_parameter(
            'require_valid_sync',
            True
        )
        self.declare_parameter(
            'lock_target_after_first_selection',
            False
        )
        self.declare_parameter(
            'max_tracking_distance_m',
            0.010
        )
        self.declare_parameter(
            'reset_session_service',
            '/d405/mushroom_pick/reset_alignment_session'
        )
        self.declare_parameter(
            'proposal_topic',
            '/d405/mushroom_pick/step_proposal'
        )
        self.declare_parameter(
            'arm_action_status_topic',
            '/fer_arm_controller/follow_joint_trajectory/_action/status'
        )

        self.input_topic = self.get_parameter(
            'input_topic'
        ).value
        self.output_topic = self.get_parameter(
            'output_topic'
        ).value
        self.expected_source_frame = self.get_parameter(
            'expected_source_frame'
        ).value
        self.target_frame = self.get_parameter(
            'target_frame'
        ).value

        self.image_center_u = float(
            self.get_parameter('image_center_u').value
        )
        self.image_center_v = float(
            self.get_parameter('image_center_v').value
        )
        self.max_center_distance_px = float(
            self.get_parameter(
                'max_center_distance_px'
            ).value
        )
        self.min_depth = float(
            self.get_parameter('min_depth').value
        )
        self.max_depth = float(
            self.get_parameter('max_depth').value
        )
        self.desired_tcp_x = float(
            self.get_parameter('desired_tcp_x').value
        )
        self.desired_tcp_y = float(
            self.get_parameter('desired_tcp_y').value
        )
        self.require_valid_sync = bool(
            self.get_parameter(
                'require_valid_sync'
            ).value
        )
        self.lock_target_after_first_selection = bool(
            self.get_parameter(
                'lock_target_after_first_selection'
            ).value
        )
        self.max_tracking_distance_m = float(
            self.get_parameter(
                'max_tracking_distance_m'
            ).value
        )
        self.reset_session_service_name = str(
            self.get_parameter('reset_session_service').value
        )
        self.proposal_topic = str(
            self.get_parameter('proposal_topic').value
        )
        self.arm_action_status_topic = str(
            self.get_parameter('arm_action_status_topic').value
        )

        if (
            not math.isfinite(
                self.max_tracking_distance_m
            )
            or self.max_tracking_distance_m <= 0.0
        ):
            raise RuntimeError(
                'max_tracking_distance_m_invalid'
            )

        self.locked_target_d405 = None
        self.target_lock_id = None
        self.last_selection_mode = None
        self.last_tracking_distance_m = None
        self.last_selection_failure_reason = None
        self.session_generation = 0
        self.last_proposal_state = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(
            self.tf_buffer,
            self
        )

        self.output_publisher = self.create_publisher(
            String,
            self.output_topic,
            10
        )
        self.input_subscription = self.create_subscription(
            String,
            self.input_topic,
            self.position_callback,
            10
        )
        self.proposal_subscription = self.create_subscription(
            String,
            self.proposal_topic,
            self.proposal_state_callback,
            10
        )
        self.reset_session_service = self.create_service(
            Trigger,
            self.reset_session_service_name,
            self.reset_session_callback
        )

        self.get_logger().info(
            f'输入 D405 三维坐标: {self.input_topic}'
        )
        self.get_logger().info(
            f'输出 TCP 局部误差: {self.output_topic}'
        )
        self.get_logger().info(
            '坐标变换: '
            f'{self.expected_source_frame} -> '
            f'{self.target_frame}'
        )
        self.get_logger().info(
            'D405 到 TCP 局部误差节点已启动'
        )
        self.get_logger().info(
            '目标连续跟踪: '
            f'enabled={self.lock_target_after_first_selection}, '
            'max_tracking_distance_m='
            f'{self.max_tracking_distance_m}'
        )

    def proposal_state_callback(self, message):
        try:
            payload = json.loads(message.data)
        except (json.JSONDecodeError, TypeError):
            return
        if isinstance(payload, dict):
            self.last_proposal_state = payload

    def action_client_count(self):
        try:
            endpoints = self.get_subscriptions_info_by_topic(
                self.arm_action_status_topic
            )
        except Exception:
            return -1
        return sum(
            1 for endpoint in endpoints
            if endpoint.node_name != self.get_name()
        )

    def executor_node_present(self):
        try:
            names = {
                f'/{name}' if namespace == '/' else f'{namespace}/{name}'
                for name, namespace in self.get_node_names_and_namespaces()
            }
        except Exception:
            return True
        return '/d405_step_executor' in names

    def reset_session_callback(self, request, response):
        del request
        observation_active = bool(
            isinstance(self.last_proposal_state, dict)
            and self.last_proposal_state.get('observation_active') is True
        )
        action_clients = self.action_client_count()
        executor_active = self.executor_node_present()
        rejection = None
        if observation_active:
            rejection = 'observation_active'
        elif executor_active:
            rejection = 'executor_active'
        elif action_clients < 0:
            rejection = 'action_client_graph_unavailable'
        elif action_clients != 0:
            rejection = 'action_clients_nonzero'

        if rejection is not None:
            response.success = False
            response.message = json.dumps({
                'schema': 'day18.d405_alignment_reset.v1',
                'accepted': False,
                'reason': rejection,
                'generation': self.session_generation,
                'observation_active': observation_active,
                'executor_active': executor_active,
                'arm_action_client_count': action_clients,
            }, sort_keys=True)
            return response

        old_generation = self.session_generation
        old_lock_id = self.target_lock_id
        self.session_generation += 1
        self.locked_target_d405 = None
        self.target_lock_id = None
        self.last_selection_mode = None
        self.last_tracking_distance_m = None
        self.last_selection_failure_reason = None
        self.publish_invalid('session_reset')
        response.success = True
        response.message = json.dumps({
            'schema': 'day18.d405_alignment_reset.v1',
            'accepted': True,
            'reason': 'alignment_session_reset_complete',
            'old_generation': old_generation,
            'new_generation': self.session_generation,
            'old_lock_id': old_lock_id,
            'new_lock_id': None,
            'cleared': {
                'lock': True,
                'proposal': False,
                'observation': False,
            },
            'camera_restarted': False,
            'detector_restarted': False,
            'localizer_restarted': False,
            'arm_action_client_count': action_clients,
        }, sort_keys=True)
        return response

    def tracking_metadata(self):
        locked_xyz = None

        if self.locked_target_d405 is not None:
            locked_xyz = {
                'x': self.locked_target_d405[0],
                'y': self.locked_target_d405[1],
                'z': self.locked_target_d405[2]
            }

        return {
            'session_generation': self.session_generation,
            'enabled': (
                self.lock_target_after_first_selection
            ),
            'lock_id': self.target_lock_id,
            'selection_mode': self.last_selection_mode,
            'tracking_distance_m': (
                self.last_tracking_distance_m
            ),
            'max_tracking_distance_m': (
                self.max_tracking_distance_m
            ),
            'locked_target_d405': locked_xyz
        }

    def publish_output(self, output_data):
        output_data = dict(output_data)
        output_data['session_generation'] = self.session_generation
        output_message = String()
        output_message.data = json.dumps(
            output_data,
            ensure_ascii=False
        )
        self.output_publisher.publish(output_message)

    def publish_invalid(
        self,
        reason,
        input_data=None,
        source_frame=None,
        target_d405=None
    ):
        stamp = None

        if isinstance(input_data, dict):
            stamp = input_data.get('stamp')

        output_data = {
            'valid': False,
            'reason': reason,
            'source_frame': source_frame,
            'target_frame': self.target_frame,
            'stamp': stamp,
            'detection_stamp': (
                input_data.get('detection_stamp')
                if isinstance(input_data, dict)
                else None
            ),
            'upstream_sync': (
                input_data.get('sync')
                if isinstance(input_data, dict)
                else None
            ),
            'tf_query': 'latest',
            'target_tracking': self.tracking_metadata(),
            'target_d405': target_d405,
            'target_tcp': None,
            'alignment_error': None
        }

        self.publish_output(output_data)

        self.get_logger().warning(
            f'局部误差无效: {reason}',
            throttle_duration_sec=1.0
        )

    def select_center_candidate(self, positions):
        candidates = []
        self.last_selection_failure_reason = None
        self.last_tracking_distance_m = None

        for position in positions:
            if not isinstance(position, dict):
                continue

            try:
                center_u = float(position['center_u'])
                center_v = float(position['center_v'])
                camera_x = float(position['camera_x'])
                camera_y = float(position['camera_y'])
                camera_z = float(position['camera_z'])
                depth = float(
                    position.get('depth', camera_z)
                )
            except (KeyError, TypeError, ValueError):
                continue

            numeric_values = (
                center_u,
                center_v,
                camera_x,
                camera_y,
                camera_z,
                depth
            )

            if not all(
                math.isfinite(value)
                for value in numeric_values
            ):
                continue

            if not (
                self.min_depth
                <= depth
                <= self.max_depth
            ):
                continue

            center_distance_px = math.hypot(
                center_u - self.image_center_u,
                center_v - self.image_center_v
            )

            if (
                center_distance_px
                > self.max_center_distance_px
            ):
                continue

            candidates.append(
                (
                    center_distance_px,
                    position,
                    center_u,
                    center_v,
                    camera_x,
                    camera_y,
                    camera_z,
                    depth
                )
            )

        if not candidates:
            self.last_selection_failure_reason = (
                'no_valid_center_candidate'
            )
            return None

        if (
            self.lock_target_after_first_selection
            and self.locked_target_d405 is not None
        ):
            tracked_candidates = []

            for candidate in candidates:
                tracking_distance_m = math.sqrt(
                    (
                        candidate[4]
                        - self.locked_target_d405[0]
                    ) ** 2
                    + (
                        candidate[5]
                        - self.locked_target_d405[1]
                    ) ** 2
                    + (
                        candidate[6]
                        - self.locked_target_d405[2]
                    ) ** 2
                )
                tracked_candidates.append(
                    (
                        tracking_distance_m,
                        candidate
                    )
                )

            tracked_candidates.sort(
                key=lambda item: item[0]
            )
            tracking_distance_m, selected = (
                tracked_candidates[0]
            )
            self.last_tracking_distance_m = (
                tracking_distance_m
            )

            if (
                tracking_distance_m
                > self.max_tracking_distance_m
            ):
                self.last_selection_mode = (
                    'tracking_jump_rejected'
                )
                self.last_selection_failure_reason = (
                    'target_tracking_distance_exceeded'
                )
                return None

            self.last_selection_mode = (
                'locked_nearest_3d'
            )
        else:
            candidates.sort(key=lambda item: item[0])
            selected = candidates[0]

            if self.lock_target_after_first_selection:
                self.last_selection_mode = (
                    'center_initial_lock'
                )
                self.target_lock_id = (
                    f'd405-g{self.session_generation}-lock-'
                    f'{int(round(selected[2]))}-'
                    f'{int(round(selected[3]))}-'
                    f'{selected[4]:.6f}-'
                    f'{selected[5]:.6f}-'
                    f'{selected[6]:.6f}'
                )
            else:
                self.last_selection_mode = (
                    'center_each_frame'
                )

        if self.lock_target_after_first_selection:
            self.locked_target_d405 = (
                selected[4],
                selected[5],
                selected[6]
            )

        return selected

    def position_callback(self, message):
        try:
            input_data = json.loads(message.data)
        except json.JSONDecodeError:
            self.publish_invalid(
                'invalid_json'
            )
            return

        if not isinstance(input_data, dict):
            self.publish_invalid(
                'input_is_not_object'
            )
            return

        source_frame = input_data.get('frame_id')

        if source_frame != self.expected_source_frame:
            self.publish_invalid(
                'unexpected_source_frame',
                input_data=input_data,
                source_frame=source_frame
            )
            return

        upstream_sync = input_data.get('sync')

        if self.require_valid_sync:
            if not isinstance(upstream_sync, dict):
                self.publish_invalid(
                    'upstream_sync_missing',
                    input_data=input_data,
                    source_frame=source_frame
                )
                return

            if upstream_sync.get('valid') is not True:
                sync_reason = upstream_sync.get(
                    'reason',
                    'unknown'
                )
                self.publish_invalid(
                    'upstream_sync_invalid: '
                    f'{sync_reason}',
                    input_data=input_data,
                    source_frame=source_frame
                )
                return

        positions = input_data.get('positions')

        if not isinstance(positions, list):
            self.publish_invalid(
                'positions_is_not_list',
                input_data=input_data,
                source_frame=source_frame
            )
            return

        selected = self.select_center_candidate(
            positions
        )

        if selected is None:
            self.publish_invalid(
                self.last_selection_failure_reason
                or 'no_valid_center_candidate',
                input_data=input_data,
                source_frame=source_frame
            )
            return

        (
            center_distance_px,
            position,
            center_u,
            center_v,
            camera_x,
            camera_y,
            camera_z,
            depth
        ) = selected

        target_d405 = {
            'x': camera_x,
            'y': camera_y,
            'z': camera_z,
            'depth': depth,
            'center_u': int(round(center_u)),
            'center_v': int(round(center_v)),
            'center_distance_px': center_distance_px,
            'confidence': position.get('confidence')
        }

        # Preserve the DAY13 ROI-sphere quality evidence for the DAY14
        # read-only handoff validator.  Selection and transform behaviour are
        # unchanged; these are additive JSON fields only.
        for key in (
            'localization_method',
            'sphere_radius_m',
            'sphere_fit_rms_m',
            'sphere_inlier_count',
            'roi_foreground_count',
            'surface_nearest_depth_m',
            'source_bbox',
        ):
            if key in position:
                target_d405[key] = position[key]

        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                source_frame,
                Time(),
                Duration(seconds=0.2)
            )
        except TransformException as error:
            self.publish_invalid(
                f'tf_lookup_failed: {error}',
                input_data=input_data,
                source_frame=source_frame,
                target_d405=target_d405
            )
            return

        source_point = PointStamped()
        source_point.header.frame_id = source_frame
        source_point.point.x = camera_x
        source_point.point.y = camera_y
        source_point.point.z = camera_z

        target_point = do_transform_point(
            source_point,
            transform
        )

        tcp_x = float(target_point.point.x)
        tcp_y = float(target_point.point.y)
        tcp_z = float(target_point.point.z)

        if not all(
            math.isfinite(value)
            for value in (tcp_x, tcp_y, tcp_z)
        ):
            self.publish_invalid(
                'non_finite_tcp_coordinate',
                input_data=input_data,
                source_frame=source_frame,
                target_d405=target_d405
            )
            return

        lateral_error_x = (
            tcp_x - self.desired_tcp_x
        )
        longitudinal_error_y = (
            tcp_y - self.desired_tcp_y
        )
        planar_error = math.hypot(
            lateral_error_x,
            longitudinal_error_y
        )

        output_data = {
            'valid': True,
            'reason': 'center_candidate_transformed',
            'source_frame': source_frame,
            'target_frame': self.target_frame,
            'stamp': input_data.get('stamp'),
            'detection_stamp': input_data.get(
                'detection_stamp'
            ),
            'upstream_sync': upstream_sync,
            'tf_query': 'latest',
            'target_tracking': self.tracking_metadata(),
            'target_d405': target_d405,
            'target_tcp': {
                'x': tcp_x,
                'y': tcp_y,
                'z': tcp_z
            },
            'alignment_error': {
                'lateral_error_x': lateral_error_x,
                'longitudinal_error_y': (
                    longitudinal_error_y
                ),
                'planar_error': planar_error,
                'depth_distance': tcp_z
            }
        }

        self.publish_output(output_data)

        self.get_logger().info(
            'TCP 目标: '
            f'XYZ=({tcp_x:.4f}, '
            f'{tcp_y:.4f}, '
            f'{tcp_z:.4f}) m, '
            f'平面误差={planar_error:.4f} m',
            throttle_duration_sec=1.0
        )


def main(args=None):
    rclpy.init(args=args)
    node = D405TcpAlignment()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
