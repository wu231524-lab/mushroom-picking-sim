import json
import traceback
from collections import deque

import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import (
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_srvs.srv import SetBool
from ultralytics import YOLO


class MushroomDetector(Node):
    def __init__(self):
        super().__init__('mushroom_detector')

        self.declare_parameter(
            'model_path',
            '/home/jay/mushroom_ws/models/yolo/best.pt'
        )
        self.declare_parameter(
            'image_topic',
            '/wrist_camera/camera/image_raw'
        )
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('include_image_metadata', False)
        self.declare_parameter('image_queue_depth', 10)
        self.declare_parameter('enable_depth_pairing', False)
        self.declare_parameter(
            'depth_topic',
            '/wrist_camera/camera/depth/image_raw'
        )
        self.declare_parameter(
            'paired_depth_topic',
            '/mushroom_detection/paired_depth'
        )
        self.declare_parameter('depth_pair_history_size', 20)
        self.declare_parameter('paired_depth_queue_depth', 4)
        self.declare_parameter('max_input_sync_delta_sec', 0.20)
        self.declare_parameter('inference_enabled', True)

        model_path = self.get_parameter('model_path').value
        image_topic = self.get_parameter('image_topic').value
        depth_topic = self.get_parameter('depth_topic').value
        paired_depth_topic = self.get_parameter(
            'paired_depth_topic'
        ).value
        self.confidence_threshold = float(
            self.get_parameter('confidence_threshold').value
        )
        self.include_image_metadata = bool(
            self.get_parameter('include_image_metadata').value
        )
        self.image_queue_depth = max(
            1,
            int(self.get_parameter('image_queue_depth').value)
        )
        self.enable_depth_pairing = bool(
            self.get_parameter('enable_depth_pairing').value
        )
        self.depth_pair_history_size = max(
            1,
            int(self.get_parameter('depth_pair_history_size').value)
        )
        self.paired_depth_queue_depth = int(
            self.get_parameter('paired_depth_queue_depth').value
        )
        self.max_input_sync_delta_sec = float(
            self.get_parameter('max_input_sync_delta_sec').value
        )
        self.inference_enabled = bool(
            self.get_parameter('inference_enabled').value
        )

        if not (
            0.0 <= self.max_input_sync_delta_sec <= 0.20
        ):
            raise RuntimeError(
                'max_input_sync_delta_sec must be within [0.0, 0.20]'
            )
        if self.enable_depth_pairing and not self.include_image_metadata:
            raise RuntimeError(
                'depth pairing requires include_image_metadata=true'
            )
        if not (1 <= self.paired_depth_queue_depth <= 8):
            raise RuntimeError(
                'paired_depth_queue_depth must be within [1, 8]'
            )

        self.bridge = CvBridge()
        self.model = YOLO(model_path)
        self.depth_pair_history = deque(
            maxlen=self.depth_pair_history_size
        )
        # The paired-depth stream is a strict one-sample handoff contract.
        # A depth frame already published for an earlier inference must never
        # be published again for a later RGB frame: the exact-depth localizer
        # intentionally treats duplicate or non-increasing stamps as a
        # fail-closed stream violation.
        self.last_paired_depth_stamp_ns = None
        self.last_pairing_rejection_ns = 0

        sensor_qos = QoSProfile(
            history=qos_profile_sensor_data.history,
            depth=self.depth_pair_history_size,
            reliability=qos_profile_sensor_data.reliability,
            durability=qos_profile_sensor_data.durability
        )
        paired_depth_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=self.paired_depth_queue_depth,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=qos_profile_sensor_data.durability
        )

        self.depth_subscription = None
        self.paired_depth_publisher = None
        if self.enable_depth_pairing:
            # Subscribe before RGB so an already-received depth sample can be
            # selected before the expensive YOLO callback begins.
            self.depth_subscription = self.create_subscription(
                Image,
                depth_topic,
                self.depth_callback,
                sensor_qos
            )
            self.paired_depth_publisher = self.create_publisher(
                Image,
                paired_depth_topic,
                paired_depth_qos
            )

        self.image_subscription = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            self.image_queue_depth
        )

        self.annotated_image_publisher = self.create_publisher(
            Image,
            '/mushroom_detection/annotated_image',
            10
        )
        self.detection_publisher = self.create_publisher(
            String,
            '/mushroom_detection/detections',
            10
        )
        self.enable_service = self.create_service(
            SetBool,
            '~/set_enabled',
            self.set_enabled_callback,
        )

        self.get_logger().info(f'模型路径: {model_path}')
        self.get_logger().info(f'订阅图像话题: {image_topic}')
        self.get_logger().info(
            f'图像订阅队列深度: {self.image_queue_depth}'
        )
        if self.enable_depth_pairing:
            self.get_logger().info(
                'YOLO 前深度配对已启用: '
                f'depth_topic={depth_topic}, '
                f'paired_depth_topic={paired_depth_topic}, '
                f'max_delta={self.max_input_sync_delta_sec:.3f} s, '
                f'history={self.depth_pair_history_size}, '
                'transport=RELIABLE/KEEP_LAST, '
                f'transport_depth={self.paired_depth_queue_depth}'
            )
        self.get_logger().info('蘑菇检测节点已启动')

    def set_enabled_callback(self, request, response):
        self.inference_enabled = bool(request.data)
        response.success = True
        response.message = (
            'inference_enabled'
            if self.inference_enabled
            else 'inference_disabled'
        )
        self.get_logger().info(response.message)
        return response

    @staticmethod
    def stamp_ns(header):
        return (
            int(header.stamp.sec) * 1_000_000_000
            + int(header.stamp.nanosec)
        )

    def depth_callback(self, message):
        try:
            self.depth_pair_history.append({
                'stamp_ns': self.stamp_ns(message.header),
                'message': message
            })
        except Exception as error:
            self.get_logger().error(
                f'缓存待配对深度消息失败: {error}'
            )

    def select_depth_before_inference(self, image_message):
        if not self.depth_pair_history:
            return None, None, 'depth_pair_history_empty'

        image_stamp_ns = self.stamp_ns(image_message.header)
        eligible_samples = [
            sample for sample in self.depth_pair_history
            if (
                self.last_paired_depth_stamp_ns is None
                or sample['stamp_ns'] > self.last_paired_depth_stamp_ns
            )
        ]
        if not eligible_samples:
            return (
                None,
                None,
                'depth_pair_no_strictly_newer_sample'
            )
        sample = min(
            eligible_samples,
            key=lambda item: abs(item['stamp_ns'] - image_stamp_ns)
        )
        delta_sec = abs(
            sample['stamp_ns'] - image_stamp_ns
        ) / 1_000_000_000.0

        if delta_sec > self.max_input_sync_delta_sec:
            return None, delta_sec, 'input_sync_delta_exceeded'
        return sample, delta_sec, 'paired_before_inference'

    def image_callback(self, image_message):
        if not self.inference_enabled:
            return

        selected_depth = None
        input_sync_delta_sec = None
        input_sync_reason = 'depth_pairing_disabled'

        if self.enable_depth_pairing:
            selected_depth, input_sync_delta_sec, input_sync_reason = (
                self.select_depth_before_inference(image_message)
            )
            if selected_depth is None:
                now_ns = self.get_clock().now().nanoseconds
                if now_ns - self.last_pairing_rejection_ns >= 1_000_000_000:
                    self.last_pairing_rejection_ns = now_ns
                    self.get_logger().warning(
                        'YOLO 前 RGB/深度配对拒绝: '
                        f'reason={input_sync_reason}, '
                        f'delta={input_sync_delta_sec}'
                    )
                return

            # Publish the exact bounded depth sample immediately after the
            # pre-inference pair is selected. ROS DDS does not guarantee
            # ordering across separate topics; publishing only after YOLO
            # could let the detection reach the localizer first.
            self.paired_depth_publisher.publish(
                selected_depth['message']
            )
            self.last_paired_depth_stamp_ns = selected_depth['stamp_ns']

        try:
            cv_image = self.bridge.imgmsg_to_cv2(
                image_message,
                desired_encoding='bgr8'
            )
            results = self.model.predict(
                source=cv_image,
                conf=self.confidence_threshold,
                verbose=False
            )

            detections = []
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = (
                        box.xyxy[0].cpu().numpy().tolist()
                    )
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.model.names[class_id]
                    detections.append({
                        'class_id': class_id,
                        'class_name': class_name,
                        'confidence': confidence,
                        'bbox': [x1, y1, x2, y2],
                        'center_u': int((x1 + x2) / 2.0),
                        'center_v': int((y1 + y2) / 2.0)
                    })

            annotated_image = results[0].plot()
            annotated_message = self.bridge.cv2_to_imgmsg(
                annotated_image,
                encoding='bgr8'
            )
            annotated_message.header = image_message.header
            self.annotated_image_publisher.publish(annotated_message)

            if self.include_image_metadata:
                detection_output = {
                    'frame_id': image_message.header.frame_id,
                    'stamp': {
                        'sec': image_message.header.stamp.sec,
                        'nanosec': image_message.header.stamp.nanosec
                    },
                    'detections': detections
                }
                if self.enable_depth_pairing:
                    depth_header = selected_depth['message'].header
                    detection_output['input_pairing'] = {
                        'valid': True,
                        'reason': input_sync_reason,
                        'delta_sec': input_sync_delta_sec,
                        'max_delta_sec': self.max_input_sync_delta_sec,
                        'depth_stamp': {
                            'sec': depth_header.stamp.sec,
                            'nanosec': depth_header.stamp.nanosec
                        }
                    }
            else:
                detection_output = detections

            detection_message = String()
            detection_message.data = json.dumps(
                detection_output,
                ensure_ascii=False
            )

            self.detection_publisher.publish(detection_message)

            self.get_logger().info(
                f'检测到 {len(detections)} 个目标',
                throttle_duration_sec=1.0
            )
        except Exception as error:
            self.get_logger().error(
                f'图像检测失败：{type(error).__name__}: {error!r}\n'
                f'{traceback.format_exc()}'
            )


def main(args=None):
    rclpy.init(args=args)
    node = MushroomDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
