import json
import math

from gazebo_msgs.msg import ModelStates
import rclpy
from gazebo_msgs.srv import GetEntityState, SetEntityState
from rclpy.node import Node
from std_srvs.srv import SetBool, Trigger
from tf2_ros import Buffer, TransformException, TransformListener


def quaternion_normalize(quaternion):
    x, y, z, w = quaternion
    norm = math.sqrt(x*x + y*y + z*z + w*w)

    if norm < 1.0e-12:
        return (0.0, 0.0, 0.0, 1.0)

    return (
        x / norm,
        y / norm,
        z / norm,
        w / norm,
    )


def quaternion_conjugate(quaternion):
    x, y, z, w = quaternion
    return (-x, -y, -z, w)


def quaternion_multiply(first, second):
    x1, y1, z1, w1 = first
    x2, y2, z2, w2 = second

    return (
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    )


def rotate_vector(quaternion, vector):
    quaternion = quaternion_normalize(quaternion)
    vector_quaternion = (
        vector[0],
        vector[1],
        vector[2],
        0.0,
    )

    rotated = quaternion_multiply(
        quaternion_multiply(
            quaternion,
            vector_quaternion,
        ),
        quaternion_conjugate(quaternion),
    )

    return rotated[:3]


class MushroomFollower(Node):

    def __init__(self):
        super().__init__('mushroom_follower')

        self.declare_parameter('entity_name', 'mushroom_3')
        self.declare_parameter('world_frame', 'world')
        self.declare_parameter('tcp_frame', 'fer_hand_tcp')
        self.declare_parameter('update_rate', 30.0)
        self.declare_parameter(
            'vacuum_validation_enabled',
            False
        )
        self.declare_parameter(
            'attachment_commands_enabled',
            True
        )
        self.declare_parameter(
            'vacuum_cap_center_z_m',
            0.095
        )
        self.declare_parameter(
            'vacuum_cap_radius_m',
            0.070
        )
        self.declare_parameter(
            'vacuum_max_planar_error_m',
            0.003
        )
        self.declare_parameter(
            'vacuum_min_surface_gap_m',
            -0.002
        )
        self.declare_parameter(
            'vacuum_max_surface_gap_m',
            0.005
        )
        self.declare_parameter(
            'vacuum_max_axis_error_rad',
            0.17453292519943295
        )
        self.declare_parameter(
            'vacuum_max_model_state_age_sec',
            0.50
        )

        self.entity_name = (
            self.get_parameter('entity_name')
            .get_parameter_value()
            .string_value
        )
        self.world_frame = (
            self.get_parameter('world_frame')
            .get_parameter_value()
            .string_value
        )
        self.tcp_frame = (
            self.get_parameter('tcp_frame')
            .get_parameter_value()
            .string_value
        )
        update_rate = (
            self.get_parameter('update_rate')
            .get_parameter_value()
            .double_value
        )
        self.vacuum_validation_enabled = bool(
            self.get_parameter(
                'vacuum_validation_enabled'
            ).value
        )
        self.attachment_commands_enabled = bool(
            self.get_parameter(
                'attachment_commands_enabled'
            ).value
        )
        self.vacuum_cap_center_z_m = float(
            self.get_parameter(
                'vacuum_cap_center_z_m'
            ).value
        )
        self.vacuum_cap_radius_m = float(
            self.get_parameter(
                'vacuum_cap_radius_m'
            ).value
        )
        self.vacuum_max_planar_error_m = float(
            self.get_parameter(
                'vacuum_max_planar_error_m'
            ).value
        )
        self.vacuum_min_surface_gap_m = float(
            self.get_parameter(
                'vacuum_min_surface_gap_m'
            ).value
        )
        self.vacuum_max_surface_gap_m = float(
            self.get_parameter(
                'vacuum_max_surface_gap_m'
            ).value
        )
        self.vacuum_max_axis_error_rad = float(
            self.get_parameter(
                'vacuum_max_axis_error_rad'
            ).value
        )
        self.vacuum_max_model_state_age_sec = float(
            self.get_parameter(
                'vacuum_max_model_state_age_sec'
            ).value
        )

        if update_rate <= 0.0:
            raise ValueError('update_rate must be positive')

        if self.vacuum_cap_radius_m <= 0.0:
            raise ValueError(
                'vacuum_cap_radius_m must be positive'
            )

        if self.vacuum_max_planar_error_m <= 0.0:
            raise ValueError(
                'vacuum_max_planar_error_m must be positive'
            )

        if (
            self.vacuum_min_surface_gap_m
            > self.vacuum_max_surface_gap_m
        ):
            raise ValueError(
                'vacuum surface gap limits invalid'
            )

        if not (
            0.0
            < self.vacuum_max_axis_error_rad
            < math.pi
        ):
            raise ValueError(
                'vacuum_max_axis_error_rad invalid'
            )

        if self.vacuum_max_model_state_age_sec <= 0.0:
            raise ValueError(
                'vacuum_max_model_state_age_sec '
                'must be positive'
            )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(
            self.tf_buffer,
            self,
        )

        self.get_state_client = self.create_client(
            GetEntityState,
            '/gazebo/get_entity_state',
        )
        self.set_state_client = self.create_client(
            SetEntityState,
            '/gazebo/set_entity_state',
        )

        self.attach_service = self.create_service(
            SetBool,
            '/mushroom_pick/set_attached',
            self.handle_set_attached,
        )

        self.attach_status_service = self.create_service(
            Trigger,
            '/mushroom_pick/get_attached',
            self.handle_get_attached,
        )
        self.vacuum_check_service = self.create_service(
            Trigger,
            '/mushroom_pick/check_vacuum_contact',
            self.handle_check_vacuum_contact,
        )
        self.model_states_subscription = (
            self.create_subscription(
                ModelStates,
                '/gazebo/model_states',
                self.model_states_callback,
                10,
            )
        )

        self.attached = False
        self.attach_requested = False

        self.get_state_future = None
        self.set_state_future = None

        self.relative_translation = None
        self.relative_rotation = None

        self.latest_model_translation = None
        self.latest_model_rotation = None
        self.latest_model_state_received_ns = None

        self.timer = self.create_timer(
            1.0 / update_rate,
            self.update,
        )

        self.get_logger().info(
            'Mushroom follower ready: '
            f'entity={self.entity_name}, '
            f'tcp={self.tcp_frame}, '
            f'rate={update_rate:.1f} Hz, '
            'vacuum_validation='
            f'{self.vacuum_validation_enabled}'
        )

    def model_states_callback(self, message):
        try:
            index = message.name.index(
                self.entity_name
            )
        except ValueError:
            return

        pose = message.pose[index]
        self.latest_model_translation = (
            pose.position.x,
            pose.position.y,
            pose.position.z,
        )
        self.latest_model_rotation = (
            quaternion_normalize((
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ))
        )
        self.latest_model_state_received_ns = (
            self.get_clock().now().nanoseconds
        )

    def vacuum_contact_report(self):
        report = {
            'valid': False,
            'reason': 'unknown',
            'entity_name': self.entity_name,
            'tcp_frame': self.tcp_frame,
            'limits': {
                'max_planar_error_m': (
                    self.vacuum_max_planar_error_m
                ),
                'min_surface_gap_m': (
                    self.vacuum_min_surface_gap_m
                ),
                'max_surface_gap_m': (
                    self.vacuum_max_surface_gap_m
                ),
                'max_axis_error_rad': (
                    self.vacuum_max_axis_error_rad
                ),
            },
        }

        if not self.vacuum_validation_enabled:
            report['reason'] = (
                'vacuum_validation_disabled'
            )
            return report

        if (
            self.latest_model_translation is None
            or self.latest_model_rotation is None
            or self.latest_model_state_received_ns is None
        ):
            report['reason'] = (
                'model_state_unavailable'
            )
            return report

        age_sec = (
            self.get_clock().now().nanoseconds
            - self.latest_model_state_received_ns
        ) / 1_000_000_000.0
        report['model_state_age_sec'] = age_sec

        if (
            age_sec < 0.0
            or age_sec
            > self.vacuum_max_model_state_age_sec
        ):
            report['reason'] = 'model_state_stale'
            return report

        try:
            tcp_translation, tcp_rotation = (
                self.lookup_tcp_pose()
            )
        except TransformException as error:
            report['reason'] = 'tcp_tf_unavailable'
            report['detail'] = str(error)
            return report

        model_translation = (
            self.latest_model_translation
        )
        model_rotation = self.latest_model_rotation
        cap_top_offset = rotate_vector(
            model_rotation,
            (
                0.0,
                0.0,
                self.vacuum_cap_center_z_m
                + self.vacuum_cap_radius_m,
            ),
        )
        cap_top = (
            model_translation[0] + cap_top_offset[0],
            model_translation[1] + cap_top_offset[1],
            model_translation[2] + cap_top_offset[2],
        )
        surface_normal = rotate_vector(
            model_rotation,
            (0.0, 0.0, 1.0),
        )
        tcp_approach_axis = rotate_vector(
            tcp_rotation,
            (0.0, 0.0, 1.0),
        )
        difference = (
            tcp_translation[0] - cap_top[0],
            tcp_translation[1] - cap_top[1],
            tcp_translation[2] - cap_top[2],
        )
        surface_gap = sum(
            difference[index]
            * surface_normal[index]
            for index in range(3)
        )
        tangent = tuple(
            difference[index]
            - surface_gap * surface_normal[index]
            for index in range(3)
        )
        planar_error = math.sqrt(sum(
            value * value
            for value in tangent
        ))
        axis_dot = sum(
            tcp_approach_axis[index]
            * (-surface_normal[index])
            for index in range(3)
        )
        axis_dot = max(-1.0, min(1.0, axis_dot))
        axis_error = math.acos(axis_dot)

        report.update({
            'tcp_world': list(tcp_translation),
            'cap_top_world': list(cap_top),
            'planar_error_m': planar_error,
            'surface_gap_m': surface_gap,
            'axis_error_rad': axis_error,
        })

        failures = []

        if planar_error > self.vacuum_max_planar_error_m:
            failures.append('planar_error_exceeded')

        if surface_gap < self.vacuum_min_surface_gap_m:
            failures.append('surface_penetration_exceeded')

        if surface_gap > self.vacuum_max_surface_gap_m:
            failures.append('surface_gap_exceeded')

        if axis_error > self.vacuum_max_axis_error_rad:
            failures.append('axis_error_exceeded')

        if failures:
            report['reason'] = '+'.join(failures)
            return report

        report['valid'] = True
        report['reason'] = 'vacuum_contact_valid'
        return report

    def handle_check_vacuum_contact(
        self,
        request,
        response
    ):
        del request
        report = self.vacuum_contact_report()
        response.success = report['valid']
        response.message = json.dumps(
            report,
            ensure_ascii=False,
        )
        return response

    def attach_from_cached_model_pose(
        self,
        tcp_translation,
        tcp_rotation
    ):
        tcp_inverse = quaternion_conjugate(
            tcp_rotation
        )
        world_difference = (
            self.latest_model_translation[0]
            - tcp_translation[0],
            self.latest_model_translation[1]
            - tcp_translation[1],
            self.latest_model_translation[2]
            - tcp_translation[2],
        )
        self.relative_translation = rotate_vector(
            tcp_inverse,
            world_difference,
        )
        self.relative_rotation = quaternion_normalize(
            quaternion_multiply(
                tcp_inverse,
                self.latest_model_rotation,
            )
        )
        self.attached = True
        self.attach_requested = False

    def handle_get_attached(self, request, response):
        del request

        response.success = self.attached

        if self.attached:
            response.message = (
                f'{self.entity_name} attached'
            )
        elif self.attach_requested:
            response.message = (
                f'{self.entity_name} attach_pending'
            )
        else:
            response.message = (
                f'{self.entity_name} detached'
            )

        return response

    def handle_set_attached(self, request, response):
        if not self.attachment_commands_enabled:
            response.success = False
            response.message = (
                'attachment_commands_disabled_by_day14_dry_run'
            )
            self.get_logger().warning(response.message)
            return response

        if request.data:
            if self.attached:
                response.success = True
                response.message = (
                    f'{self.entity_name} is already attached'
                )
                return response

            if self.vacuum_validation_enabled:
                report = self.vacuum_contact_report()

                if not report['valid']:
                    response.success = False
                    response.message = json.dumps(
                        report,
                        ensure_ascii=False,
                    )
                    self.get_logger().warning(
                        'Vacuum attach rejected: '
                        f'{report["reason"]}'
                    )
                    return response

                tcp_translation = tuple(
                    report['tcp_world']
                )
                _, tcp_rotation = self.lookup_tcp_pose()
                self.attach_from_cached_model_pose(
                    tcp_translation,
                    tcp_rotation,
                )
                response.success = True
                response.message = json.dumps(
                    report,
                    ensure_ascii=False,
                )
                self.get_logger().info(
                    f'{self.entity_name} VACUUM ATTACHED'
                )
                return response

            self.attach_requested = True
            response.success = True
            response.message = (
                f'Attach requested for {self.entity_name}'
            )
            self.get_logger().info(response.message)
            return response

        self.attached = False
        self.attach_requested = False
        self.get_state_future = None
        self.relative_translation = None
        self.relative_rotation = None

        response.success = True
        response.message = (
            f'{self.entity_name} detached at current pose'
        )
        self.get_logger().info(response.message)
        return response

    def lookup_tcp_pose(self):
        transform = self.tf_buffer.lookup_transform(
            self.world_frame,
            self.tcp_frame,
            rclpy.time.Time(),
        )

        translation = transform.transform.translation
        rotation = transform.transform.rotation

        return (
            (translation.x, translation.y, translation.z),
            quaternion_normalize((
                rotation.x,
                rotation.y,
                rotation.z,
                rotation.w,
            )),
        )

    def request_entity_state(self):
        if not self.get_state_client.service_is_ready():
            return

        request = GetEntityState.Request()
        request.name = self.entity_name
        request.reference_frame = self.world_frame

        self.get_state_future = (
            self.get_state_client.call_async(request)
        )

    def finish_attachment(self):
        if self.get_state_future is None:
            return

        if not self.get_state_future.done():
            return

        try:
            result = self.get_state_future.result()
        except Exception as error:
            self.get_logger().error(
                f'GetEntityState failed: {error}'
            )
            self.get_state_future = None
            return

        self.get_state_future = None

        if not result.success:
            self.get_logger().error(
                'GetEntityState rejected: '
                f'{result.status_message}'
            )
            return

        try:
            tcp_translation, tcp_rotation = (
                self.lookup_tcp_pose()
            )
        except TransformException as error:
            self.get_logger().warning(
                f'TF unavailable during attach: {error}'
            )
            return

        model_position = result.state.pose.position
        model_orientation = result.state.pose.orientation

        model_translation = (
            model_position.x,
            model_position.y,
            model_position.z,
        )
        model_rotation = quaternion_normalize((
            model_orientation.x,
            model_orientation.y,
            model_orientation.z,
            model_orientation.w,
        ))

        tcp_inverse = quaternion_conjugate(tcp_rotation)

        world_difference = (
            model_translation[0] - tcp_translation[0],
            model_translation[1] - tcp_translation[1],
            model_translation[2] - tcp_translation[2],
        )

        self.relative_translation = rotate_vector(
            tcp_inverse,
            world_difference,
        )
        self.relative_rotation = quaternion_normalize(
            quaternion_multiply(
                tcp_inverse,
                model_rotation,
            )
        )

        self.attached = True
        self.attach_requested = False

        self.get_logger().info(
            f'{self.entity_name} ATTACHED; '
            'relative translation='
            f'({self.relative_translation[0]:.4f}, '
            f'{self.relative_translation[1]:.4f}, '
            f'{self.relative_translation[2]:.4f})'
        )

    def send_follow_pose(self):
        if self.set_state_future is not None:
            if not self.set_state_future.done():
                return

            try:
                previous_result = self.set_state_future.result()
                if not previous_result.success:
                    self.get_logger().warning(
                        'SetEntityState was rejected'
                    )
            except Exception as error:
                self.get_logger().warning(
                    f'SetEntityState failed: {error}'
                )

            self.set_state_future = None

        try:
            tcp_translation, tcp_rotation = (
                self.lookup_tcp_pose()
            )
        except TransformException:
            return

        rotated_offset = rotate_vector(
            tcp_rotation,
            self.relative_translation,
        )

        model_translation = (
            tcp_translation[0] + rotated_offset[0],
            tcp_translation[1] + rotated_offset[1],
            tcp_translation[2] + rotated_offset[2],
        )
        model_rotation = quaternion_normalize(
            quaternion_multiply(
                tcp_rotation,
                self.relative_rotation,
            )
        )

        request = SetEntityState.Request()
        request.state.name = self.entity_name
        request.state.reference_frame = self.world_frame

        request.state.pose.position.x = model_translation[0]
        request.state.pose.position.y = model_translation[1]
        request.state.pose.position.z = model_translation[2]

        request.state.pose.orientation.x = model_rotation[0]
        request.state.pose.orientation.y = model_rotation[1]
        request.state.pose.orientation.z = model_rotation[2]
        request.state.pose.orientation.w = model_rotation[3]

        self.set_state_future = (
            self.set_state_client.call_async(request)
        )

    def update(self):
        if self.attach_requested:
            if self.get_state_future is None:
                self.request_entity_state()

            self.finish_attachment()

        if (
            self.attached
            and self.relative_translation is not None
            and self.relative_rotation is not None
            and self.set_state_client.service_is_ready()
        ):
            self.send_follow_pose()


def main(args=None):
    rclpy.init(args=args)
    node = MushroomFollower()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
