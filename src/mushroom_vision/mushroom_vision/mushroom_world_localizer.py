import json

import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import String
from tf2_geometry_msgs import do_transform_point
from tf2_ros import Buffer, TransformException, TransformListener


class MushroomWorldLocalizer(Node):

    def __init__(self):
        super().__init__('mushroom_world_localizer')

        self.declare_parameter(
            'input_topic',
            '/mushroom_detection/positions_3d'
        )
        self.declare_parameter(
            'output_topic',
            '/mushroom_detection/positions_world'
        )
        self.declare_parameter(
            'source_frame',
            'wrist_camera_optical_frame'
        )
        self.declare_parameter(
            'target_frame',
            'world'
        )

        input_topic = self.get_parameter(
            'input_topic'
        ).value
        output_topic = self.get_parameter(
            'output_topic'
        ).value

        self.source_frame = self.get_parameter(
            'source_frame'
        ).value
        self.target_frame = self.get_parameter(
            'target_frame'
        ).value

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(
            self.tf_buffer,
            self
        )

        self.position_subscription = self.create_subscription(
            String,
            input_topic,
            self.position_callback,
            10
        )

        self.world_position_publisher = self.create_publisher(
            String,
            output_topic,
            10
        )

        self.last_log_time_ns = 0

        self.get_logger().info(
            f'订阅相机坐标话题: {input_topic}'
        )
        self.get_logger().info(
            f'发布世界坐标话题: {output_topic}'
        )
        self.get_logger().info(
            f'坐标变换: {self.source_frame} -> '
            f'{self.target_frame}'
        )
        self.get_logger().info(
            '蘑菇世界坐标转换节点已启动'
        )

    def position_callback(self, message):
        try:
            input_data = json.loads(message.data)
        except json.JSONDecodeError as error:
            self.get_logger().error(
                f'三维坐标 JSON 解析失败: {error}'
            )
            return

        positions = input_data.get('positions', [])

        source_frame = input_data.get(
            'frame_id',
            self.source_frame
        )

        if source_frame != self.source_frame:
            self.get_logger().warning(
                f'输入帧为 {source_frame}，'
                f'预期帧为 {self.source_frame}'
            )

        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                source_frame,
                Time(),
                Duration(seconds=0.2)
            )
        except TransformException as error:
            self.get_logger().warning(
                f'无法获取坐标变换 '
                f'{source_frame} -> {self.target_frame}: '
                f'{error}'
            )
            return

        world_positions = []

        for position in positions:
            required_fields = (
                'camera_x',
                'camera_y',
                'camera_z'
            )

            if not all(
                field in position
                for field in required_fields
            ):
                self.get_logger().warning(
                    '目标缺少 camera_x、camera_y '
                    '或 camera_z'
                )
                continue

            camera_point = PointStamped()
            camera_point.header.frame_id = source_frame

            camera_point.point.x = float(
                position['camera_x']
            )
            camera_point.point.y = float(
                position['camera_y']
            )
            camera_point.point.z = float(
                position['camera_z']
            )

            world_point = do_transform_point(
                camera_point,
                transform
            )

            converted_position = {
                'class_id': position.get('class_id'),
                'class_name': position.get('class_name'),
                'confidence': position.get('confidence'),
                'center_u': position.get('center_u'),
                'center_v': position.get('center_v'),
                'camera_x': camera_point.point.x,
                'camera_y': camera_point.point.y,
                'camera_z': camera_point.point.z,
                'world_x': world_point.point.x,
                'world_y': world_point.point.y,
                'world_z': world_point.point.z
            }

            world_positions.append(converted_position)

        output_data = {
            'frame_id': self.target_frame,
            'source_frame_id': source_frame,
            'stamp': input_data.get('stamp'),
            'positions': world_positions
        }

        output_message = String()
        output_message.data = json.dumps(
            output_data,
            ensure_ascii=False
        )
        self.world_position_publisher.publish(
            output_message
        )

        current_time_ns = self.get_clock().now().nanoseconds

        if (
            current_time_ns - self.last_log_time_ns
            >= 1_000_000_000
        ):
            self.last_log_time_ns = current_time_ns

            self.get_logger().info(
                f'输出 {len(world_positions)} '
                '个蘑菇世界坐标'
            )

            for index, position in enumerate(
                world_positions
            ):
                self.get_logger().info(
                    f'目标 {index + 1}: '
                    f'world XYZ=('
                    f'{position["world_x"]:.3f}, '
                    f'{position["world_y"]:.3f}, '
                    f'{position["world_z"]:.3f}) m'
                )


def main(args=None):
    rclpy.init(args=args)
    node = MushroomWorldLocalizer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
