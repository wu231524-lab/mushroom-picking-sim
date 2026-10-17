import json
import math

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener


class MushroomTargetSelector(Node):

    def __init__(self):
        super().__init__('mushroom_target_selector')

        self.declare_parameter(
            'input_topic',
            '/mushroom_detection/positions_world'
        )
        self.declare_parameter(
            'output_topic',
            '/mushroom_pick/selected_target'
        )
        self.declare_parameter('world_frame', 'world')
        self.declare_parameter('base_frame', 'fer_link0')
        self.declare_parameter(
            'target_entity_name',
            'mushroom_3'
        )

        self.declare_parameter('min_world_z', 0.34)
        self.declare_parameter('max_world_z', 0.45)

        self.declare_parameter('mushroom3_x', 0.30)
        self.declare_parameter('mushroom3_y', -0.12)
        self.declare_parameter(
            'mushroom3_radius',
            0.10
        )

        self.declare_parameter(
            'compensation_x',
            0.0087
        )
        self.declare_parameter(
            'compensation_y',
            -0.0084
        )
        self.declare_parameter(
            'compensation_z',
            0.0
        )

        input_topic = self.get_parameter(
            'input_topic'
        ).value
        output_topic = self.get_parameter(
            'output_topic'
        ).value

        self.world_frame = self.get_parameter(
            'world_frame'
        ).value
        self.base_frame = self.get_parameter(
            'base_frame'
        ).value
        self.target_entity_name = self.get_parameter(
            'target_entity_name'
        ).value

        self.min_world_z = float(
            self.get_parameter('min_world_z').value
        )
        self.max_world_z = float(
            self.get_parameter('max_world_z').value
        )

        self.mushroom3_x = float(
            self.get_parameter('mushroom3_x').value
        )
        self.mushroom3_y = float(
            self.get_parameter('mushroom3_y').value
        )
        self.mushroom3_radius = float(
            self.get_parameter(
                'mushroom3_radius'
            ).value
        )

        self.compensation_x = float(
            self.get_parameter(
                'compensation_x'
            ).value
        )
        self.compensation_y = float(
            self.get_parameter(
                'compensation_y'
            ).value
        )
        self.compensation_z = float(
            self.get_parameter(
                'compensation_z'
            ).value
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(
            self.tf_buffer,
            self
        )

        self.subscription = self.create_subscription(
            String,
            input_topic,
            self.positions_callback,
            10
        )
        self.publisher = self.create_publisher(
            String,
            output_topic,
            10
        )

        self.last_log_time_ns = 0

        self.get_logger().info(
            'Mushroom target selector ready: '
            f'z=[{self.min_world_z:.2f}, '
            f'{self.max_world_z:.2f}], '
            f'mushroom3=({self.mushroom3_x:.2f}, '
            f'{self.mushroom3_y:.2f}), '
            f'radius={self.mushroom3_radius:.2f}'
        )

    def publish_result(self, result):
        message = String()
        message.data = json.dumps(
            result,
            ensure_ascii=False
        )
        self.publisher.publish(message)

    def publish_invalid(
        self,
        reason,
        stamp=None,
        source_count=0,
        height_filtered_count=0
    ):
        self.publish_result({
            'valid': False,
            'reason': reason,
            'entity_name': self.target_entity_name,
            'frame_id': self.world_frame,
            'stamp': stamp,
            'source_count': source_count,
            'height_filtered_count': (
                height_filtered_count
            )
        })

    def log_selected_target(self, result):
        current_time_ns = (
            self.get_clock().now().nanoseconds
        )

        if (
            current_time_ns - self.last_log_time_ns
            < 1_000_000_000
        ):
            return

        self.last_log_time_ns = current_time_ns

        raw = result['raw_world']
        compensated = result['compensated_world']

        self.get_logger().info(
            'SELECTED mushroom_3: '
            f'raw=({raw["x"]:.4f}, '
            f'{raw["y"]:.4f}, '
            f'{raw["z"]:.4f}), '
            f'compensated=('
            f'{compensated["x"]:.4f}, '
            f'{compensated["y"]:.4f}, '
            f'{compensated["z"]:.4f}), '
            f'base_distance='
            f'{result["distance_to_base"]:.4f}'
        )

    def positions_callback(self, message):
        try:
            input_data = json.loads(message.data)
        except json.JSONDecodeError as error:
            self.get_logger().error(
                f'positions_world JSON invalid: {error}'
            )
            self.publish_invalid('invalid_json')
            return

        positions = input_data.get('positions', [])
        stamp = input_data.get('stamp')

        if not isinstance(positions, list):
            self.publish_invalid(
                'positions_is_not_a_list',
                stamp=stamp
            )
            return

        try:
            transform = self.tf_buffer.lookup_transform(
                self.world_frame,
                self.base_frame,
                Time(),
                Duration(seconds=0.2)
            )
        except TransformException as error:
            self.get_logger().warning(
                f'Base TF unavailable: {error}'
            )
            self.publish_invalid(
                'base_tf_unavailable',
                stamp=stamp,
                source_count=len(positions)
            )
            return

        base = transform.transform.translation
        candidates = []

        for position in positions:
            try:
                world_x = float(position['world_x'])
                world_y = float(position['world_y'])
                world_z = float(position['world_z'])
            except (KeyError, TypeError, ValueError):
                continue

            if not all(math.isfinite(value) for value in (
                world_x,
                world_y,
                world_z
            )):
                continue

            if not (
                self.min_world_z
                <= world_z
                <= self.max_world_z
            ):
                continue

            distance_to_base = math.sqrt(
                (world_x - base.x) ** 2
                + (world_y - base.y) ** 2
                + (world_z - base.z) ** 2
            )

            candidates.append({
                'source': position,
                'world_x': world_x,
                'world_y': world_y,
                'world_z': world_z,
                'distance_to_base': distance_to_base
            })

        if not candidates:
            self.publish_invalid(
                'no_height_valid_target',
                stamp=stamp,
                source_count=len(positions),
                height_filtered_count=0
            )
            return

        selected = min(
            candidates,
            key=lambda item: item['distance_to_base']
        )

        distance_to_mushroom3 = math.hypot(
            selected['world_x'] - self.mushroom3_x,
            selected['world_y'] - self.mushroom3_y
        )

        if distance_to_mushroom3 > self.mushroom3_radius:
            self.publish_invalid(
                'nearest_target_outside_mushroom3_region',
                stamp=stamp,
                source_count=len(positions),
                height_filtered_count=len(candidates)
            )
            return

        raw_world = {
            'x': selected['world_x'],
            'y': selected['world_y'],
            'z': selected['world_z']
        }

        compensated_world = {
            'x': (
                selected['world_x']
                + self.compensation_x
            ),
            'y': (
                selected['world_y']
                + self.compensation_y
            ),
            'z': (
                selected['world_z']
                + self.compensation_z
            )
        }

        result = {
            'valid': True,
            'reason': 'mushroom3_selected',
            'entity_name': self.target_entity_name,
            'frame_id': self.world_frame,
            'stamp': stamp,
            'source_count': len(positions),
            'height_filtered_count': len(candidates),
            'distance_to_base': (
                selected['distance_to_base']
            ),
            'distance_to_mushroom3_nominal': (
                distance_to_mushroom3
            ),
            'raw_world': raw_world,
            'compensated_world': compensated_world,
            'source_detection': selected['source']
        }

        self.publish_result(result)
        self.log_selected_target(result)


def main(args=None):
    rclpy.init(args=args)
    node = MushroomTargetSelector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
