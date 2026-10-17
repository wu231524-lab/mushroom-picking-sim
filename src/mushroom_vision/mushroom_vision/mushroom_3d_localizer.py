import json
import math
from collections import deque

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import (
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String


class Mushroom3DLocalizer(Node):

    def __init__(self):
        super().__init__('mushroom_3d_localizer')

        self.declare_parameter(
            'detection_topic',
            '/mushroom_detection/detections'
        )
        self.declare_parameter(
            'depth_topic',
            '/wrist_camera/camera/depth/image_raw'
        )
        self.declare_parameter(
            'camera_info_topic',
            '/wrist_camera/camera/depth/camera_info'
        )
        self.declare_parameter(
            'output_topic',
            '/mushroom_detection/positions_3d'
        )

        self.declare_parameter(
            'optical_frame_id',
            'wrist_camera_optical_frame'
        )
        self.declare_parameter(
            'depth_window_size',
            5
        )
        self.declare_parameter(
            'enable_time_sync',
            False
        )
        self.declare_parameter(
            'depth_history_size',
            20
        )
        self.declare_parameter(
            'reliable_depth_transport',
            False
        )
        self.declare_parameter(
            'require_exact_paired_depth',
            False
        )
        self.declare_parameter(
            'depth_transport_queue_depth',
            4
        )
        self.declare_parameter(
            'max_sync_delta_sec',
            0.20
        )
        self.declare_parameter(
            'localization_mode',
            'center_window'
        )
        self.declare_parameter(
            'roi_foreground_max_depth_m',
            0.30
        )
        self.declare_parameter(
            'roi_min_component_pixels',
            1000
        )
        self.declare_parameter(
            'sphere_min_radius_m',
            0.05
        )
        self.declare_parameter(
            'sphere_max_radius_m',
            0.09
        )
        self.declare_parameter(
            'sphere_max_rms_m',
            0.001
        )

        detection_topic = self.get_parameter(
            'detection_topic'
        ).value

        depth_topic = self.get_parameter(
            'depth_topic'
        ).value

        camera_info_topic = self.get_parameter(
            'camera_info_topic'
        ).value

        output_topic = self.get_parameter(
            'output_topic'
        ).value

        self.optical_frame_id = self.get_parameter(
            'optical_frame_id'
        ).value

        self.depth_window_size = int(
            self.get_parameter(
                'depth_window_size'
            ).value
        )

        self.enable_time_sync = bool(
            self.get_parameter(
                'enable_time_sync'
            ).value
        )
        self.depth_history_size = int(
            self.get_parameter(
                'depth_history_size'
            ).value
        )
        self.reliable_depth_transport = bool(
            self.get_parameter(
                'reliable_depth_transport'
            ).value
        )
        self.require_exact_paired_depth = bool(
            self.get_parameter(
                'require_exact_paired_depth'
            ).value
        )
        self.depth_transport_queue_depth = int(
            self.get_parameter(
                'depth_transport_queue_depth'
            ).value
        )
        self.max_sync_delta_sec = float(
            self.get_parameter(
                'max_sync_delta_sec'
            ).value
        )
        self.localization_mode = str(
            self.get_parameter(
                'localization_mode'
            ).value
        )
        self.roi_foreground_max_depth_m = float(
            self.get_parameter(
                'roi_foreground_max_depth_m'
            ).value
        )
        self.roi_min_component_pixels = int(
            self.get_parameter(
                'roi_min_component_pixels'
            ).value
        )
        self.sphere_min_radius_m = float(
            self.get_parameter(
                'sphere_min_radius_m'
            ).value
        )
        self.sphere_max_radius_m = float(
            self.get_parameter(
                'sphere_max_radius_m'
            ).value
        )
        self.sphere_max_rms_m = float(
            self.get_parameter(
                'sphere_max_rms_m'
            ).value
        )

        if self.localization_mode not in (
            'center_window',
            'roi_sphere'
        ):
            raise RuntimeError(
                'localization_mode must be '
                'center_window or roi_sphere'
            )

        if self.roi_min_component_pixels < 50:
            raise RuntimeError(
                'roi_min_component_pixels_too_small'
            )

        if not (
            0.07
            < self.roi_foreground_max_depth_m
            <= 0.50
        ):
            raise RuntimeError(
                'roi_foreground_max_depth_m_invalid'
            )

        if not (
            0.0
            < self.sphere_min_radius_m
            < self.sphere_max_radius_m
        ):
            raise RuntimeError(
                'sphere_radius_limits_invalid'
            )

        if self.sphere_max_rms_m <= 0.0:
            raise RuntimeError(
                'sphere_max_rms_m_invalid'
            )

        if self.depth_history_size < 1:
            self.depth_history_size = 1

        if not (1 <= self.depth_transport_queue_depth <= 8):
            raise RuntimeError(
                'depth_transport_queue_depth must be within [1, 8]'
            )

        if not (
            math.isfinite(self.max_sync_delta_sec)
            and 0.0 <= self.max_sync_delta_sec <= 0.20
        ):
            raise RuntimeError(
                'max_sync_delta_sec must be within [0.0, 0.20]'
            )

        if self.require_exact_paired_depth and not self.enable_time_sync:
            raise RuntimeError(
                'exact paired depth requires enable_time_sync=true'
            )

        if self.depth_window_size < 1:
            self.depth_window_size = 1

        if self.depth_window_size % 2 == 0:
            self.depth_window_size += 1

        self.bridge = CvBridge()

        self.depth_image = None
        self.depth_encoding = None
        self.depth_header = None
        self.depth_history = deque(
            maxlen=self.depth_history_size
        )

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self.last_log_time_ns = 0
        self.last_exact_depth_stamp_ns = None
        self.exact_depth_stream_failure = None
        self.last_exact_detection_stamp_ns = None

        # Raw camera transport keeps sensor-data QoS.  The D405 launch opts
        # into a small RELIABLE queue only for the low-rate paired-depth
        # stream selected before YOLO.  This prevents a valid selected image
        # from being lost while keeping memory and back-pressure bounded.
        depth_subscription_qos = QoSProfile(
            history=(
                QoSHistoryPolicy.KEEP_LAST
                if self.reliable_depth_transport
                else qos_profile_sensor_data.history
            ),
            depth=(
                self.depth_transport_queue_depth
                if self.reliable_depth_transport
                else self.depth_history_size
                if self.enable_time_sync
                else qos_profile_sensor_data.depth
            ),
            reliability=(
                QoSReliabilityPolicy.RELIABLE
                if self.reliable_depth_transport
                else qos_profile_sensor_data.reliability
            ),
            durability=(
                qos_profile_sensor_data.durability
            )
        )

        self.depth_subscription = self.create_subscription(
            Image,
            depth_topic,
            self.depth_callback,
            depth_subscription_qos
        )

        self.camera_info_subscription = self.create_subscription(
            CameraInfo,
            camera_info_topic,
            self.camera_info_callback,
            qos_profile_sensor_data
        )

        self.detection_subscription = self.create_subscription(
            String,
            detection_topic,
            self.detection_callback,
            10
        )

        self.position_publisher = self.create_publisher(
            String,
            output_topic,
            10
        )

        self.get_logger().info(
            f'检测结果话题: {detection_topic}'
        )
        self.get_logger().info(
            f'深度图话题: {depth_topic}'
        )
        self.get_logger().info(
            '深度传输 QoS: '
            + (
                'RELIABLE/KEEP_LAST/'
                f'{self.depth_transport_queue_depth}'
                if self.reliable_depth_transport
                else 'SENSOR_DATA'
            )
        )
        self.get_logger().info(
            f'相机内参话题: {camera_info_topic}'
        )
        self.get_logger().info(
            f'三维坐标话题: {output_topic}'
        )
        self.get_logger().info(
            '蘑菇三维定位节点已启动'
        )
        self.get_logger().info(
            '定位模式: '
            f'{self.localization_mode}'
        )

    def camera_info_callback(self, message):
        fx = float(message.k[0])
        fy = float(message.k[4])
        cx = float(message.k[2])
        cy = float(message.k[5])

        if fx <= 0.0 or fy <= 0.0:
            self.get_logger().error('相机内参中的焦距无效')
            return

        first_camera_info = self.fx is None

        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

        if first_camera_info:
            self.get_logger().info(
                '已获得相机内参: '
                f'fx={self.fx:.4f}, fy={self.fy:.4f}, '
                f'cx={self.cx:.4f}, cy={self.cy:.4f}'
            )

    def depth_callback(self, message):
        if self.enable_time_sync:
            # Store the bounded ROS message first and defer cv_bridge/NumPy
            # conversion until a detection selects this exact sample.  This
            # makes the high-rate callback cheap and reduces transport loss
            # under the VM's saturated CPU without weakening synchronization.
            try:
                stamp_ns = (
                    int(message.header.stamp.sec)
                    * 1_000_000_000
                    + int(message.header.stamp.nanosec)
                )
                if self.require_exact_paired_depth:
                    if (
                        self.last_exact_depth_stamp_ns is not None
                        and stamp_ns <= self.last_exact_depth_stamp_ns
                    ):
                        self.exact_depth_stream_failure = (
                            'exact_paired_depth_duplicate'
                            if stamp_ns == self.last_exact_depth_stamp_ns
                            else 'exact_paired_depth_stamp_not_increasing'
                        )
                        self.get_logger().error(
                            self.exact_depth_stream_failure
                        )
                        return
                    self.last_exact_depth_stamp_ns = stamp_ns
                self.depth_image = message
                self.depth_encoding = message.encoding
                self.depth_header = message.header
                self.depth_history.append({
                    'stamp_ns': stamp_ns,
                    'message': message,
                    'encoding': message.encoding,
                    'header': message.header
                })
            except Exception as error:
                self.get_logger().error(
                    '缓存原始深度消息失败: '
                    f'{error}'
                )
            return

        try:
            depth_image = self.bridge.imgmsg_to_cv2(
                message,
                desired_encoding='passthrough'
            )

            depth_array = np.asarray(depth_image)

            self.depth_image = depth_array
            self.depth_encoding = message.encoding
            self.depth_header = message.header

        except Exception as error:
            self.get_logger().error(
                f'深度图转换失败: {error}'
            )

    @staticmethod
    def stamp_to_nanoseconds(stamp_data):
        if not isinstance(stamp_data, dict):
            return None

        try:
            sec = int(stamp_data['sec'])
            nanosec = int(stamp_data['nanosec'])
        except (KeyError, TypeError, ValueError):
            return None

        if (
            sec < 0
            or nanosec < 0
            or nanosec >= 1_000_000_000
        ):
            return None

        return sec * 1_000_000_000 + nanosec

    def find_closest_depth_sample(
        self,
        detection_stamp
    ):
        detection_stamp_ns = (
            self.stamp_to_nanoseconds(
                detection_stamp
            )
        )

        if detection_stamp_ns is None:
            return (
                None,
                None,
                'invalid_detection_stamp'
            )

        if not self.depth_history:
            return (
                None,
                None,
                'depth_history_empty'
            )

        sample = min(
            self.depth_history,
            key=lambda item: abs(
                item['stamp_ns']
                - detection_stamp_ns
            )
        )

        delta_sec = abs(
            sample['stamp_ns']
            - detection_stamp_ns
        ) / 1_000_000_000.0

        if delta_sec > self.max_sync_delta_sec:
            return (
                None,
                delta_sec,
                'sync_delta_exceeded'
            )

        return (
            sample,
            delta_sec,
            'synchronized'
        )

    def find_exact_depth_sample(
        self,
        detection_stamp,
        input_pairing
    ):
        detection_stamp_ns = self.stamp_to_nanoseconds(
            detection_stamp
        )
        if detection_stamp_ns is None:
            return None, None, 'invalid_detection_stamp'
        if not isinstance(input_pairing, dict):
            return None, None, 'input_pairing_missing'
        if self.exact_depth_stream_failure is not None:
            return None, None, self.exact_depth_stream_failure
        if input_pairing.get('valid') is not True:
            return None, None, 'input_pairing_invalid'
        if input_pairing.get('reason') != 'paired_before_inference':
            return None, None, 'input_pairing_reason_invalid'

        depth_stamp = input_pairing.get('depth_stamp')
        depth_stamp_ns = self.stamp_to_nanoseconds(depth_stamp)
        if depth_stamp_ns is None:
            return None, None, 'input_pairing_depth_stamp_invalid'

        try:
            declared_delta = float(input_pairing['delta_sec'])
            declared_limit = float(input_pairing['max_delta_sec'])
        except (KeyError, TypeError, ValueError):
            return None, None, 'input_pairing_fields_invalid'
        if not all(
            math.isfinite(value)
            for value in (declared_delta, declared_limit)
        ):
            return None, None, 'input_pairing_non_finite'
        if not 0.0 <= declared_limit <= 0.20:
            return None, declared_delta, 'input_pairing_limit_invalid'
        if not 0.0 <= declared_delta <= declared_limit:
            return None, declared_delta, 'input_pairing_delta_exceeded'

        actual_delta = abs(
            detection_stamp_ns - depth_stamp_ns
        ) / 1_000_000_000.0
        if abs(actual_delta - declared_delta) > 1.0e-9:
            return None, actual_delta, 'input_pairing_delta_mismatch'
        if actual_delta > self.max_sync_delta_sec:
            return None, actual_delta, 'sync_delta_exceeded'

        if (
            self.last_exact_detection_stamp_ns is not None
            and detection_stamp_ns <= self.last_exact_detection_stamp_ns
        ):
            return None, actual_delta, 'detection_stamp_not_increasing'

        matches = [
            sample for sample in self.depth_history
            if sample['stamp_ns'] == depth_stamp_ns
        ]
        if not matches:
            return None, actual_delta, 'exact_paired_depth_missing'
        if len(matches) != 1:
            return None, actual_delta, 'exact_paired_depth_duplicate'

        self.last_exact_detection_stamp_ns = detection_stamp_ns
        return matches[0], actual_delta, 'exact_paired_depth'

    def read_depth_in_metres(
        self,
        center_u,
        center_v,
        depth_image=None,
        depth_encoding=None
    ):
        if depth_image is None:
            depth_image = self.depth_image
            depth_encoding = self.depth_encoding

        if depth_image is None:
            return None

        image_height, image_width = (
            depth_image.shape[:2]
        )

        if not (
            0 <= center_u < image_width
            and 0 <= center_v < image_height
        ):
            return None

        radius = self.depth_window_size // 2

        x_min = max(0, center_u - radius)
        x_max = min(
            image_width,
            center_u + radius + 1
        )
        y_min = max(0, center_v - radius)
        y_max = min(
            image_height,
            center_v + radius + 1
        )

        depth_region = depth_image[
            y_min:y_max,
            x_min:x_max
        ].astype(np.float32)

        if depth_encoding == '16UC1':
            depth_region *= 0.001
        elif depth_encoding != '32FC1':
            self.get_logger().warning(
                f'未识别的深度编码 {depth_encoding}，'
                '暂按米处理'
            )

        valid_depths = depth_region[
            np.isfinite(depth_region)
            & (depth_region > 0.0)
        ]

        if valid_depths.size == 0:
            return None

        return float(np.median(valid_depths))

    @staticmethod
    def convert_depth_to_metres(
        depth_image,
        depth_encoding
    ):
        depth_metres = depth_image.astype(
            np.float64
        )

        if depth_encoding == '16UC1':
            depth_metres *= 0.001

        return depth_metres

    @staticmethod
    def fit_sphere(points):
        squared = np.sum(
            points * points,
            axis=1
        )
        matrix = np.column_stack((
            2.0 * points[:, 0],
            2.0 * points[:, 1],
            2.0 * points[:, 2],
            np.ones(points.shape[0])
        ))

        solution, _, _, _ = np.linalg.lstsq(
            matrix,
            squared,
            rcond=None
        )

        center = solution[:3]
        radius_squared = (
            solution[3]
            + np.dot(center, center)
        )

        if (
            not math.isfinite(radius_squared)
            or radius_squared <= 0.0
        ):
            return None, None

        return center, math.sqrt(radius_squared)

    def localize_roi_sphere(
        self,
        detection,
        depth_image,
        depth_encoding
    ):
        bbox = detection.get('bbox')

        if not (
            isinstance(bbox, list)
            and len(bbox) == 4
        ):
            return None, 'bbox_missing'

        try:
            x1, y1, x2, y2 = (
                float(value) for value in bbox
            )
        except (TypeError, ValueError):
            return None, 'bbox_invalid'

        if not all(
            math.isfinite(value)
            for value in (x1, y1, x2, y2)
        ):
            return None, 'bbox_non_finite'

        image_height, image_width = (
            depth_image.shape[:2]
        )
        x_min = max(0, int(math.floor(x1)))
        y_min = max(0, int(math.floor(y1)))
        x_max = min(
            image_width,
            int(math.ceil(x2)) + 1
        )
        y_max = min(
            image_height,
            int(math.ceil(y2)) + 1
        )

        if x_max <= x_min or y_max <= y_min:
            return None, 'bbox_out_of_image'

        depth_metres = self.convert_depth_to_metres(
            depth_image,
            depth_encoding
        )
        roi = depth_metres[
            y_min:y_max,
            x_min:x_max
        ]
        foreground = (
            np.isfinite(roi)
            & (roi >= 0.07)
            & (
                roi
                <= self.roi_foreground_max_depth_m
            )
        ).astype(np.uint8)

        label_count, labels, stats, _ = (
            cv2.connectedComponentsWithStats(
                foreground,
                connectivity=8
            )
        )

        if label_count <= 1:
            return None, 'roi_no_foreground_component'

        component = 1 + int(
            np.argmax(
                stats[1:, cv2.CC_STAT_AREA]
            )
        )
        v_local, u_local = np.where(
            labels == component
        )

        if (
            u_local.size
            < self.roi_min_component_pixels
        ):
            return None, 'roi_component_too_small'

        pixel_u = (
            u_local.astype(np.float64)
            + x_min
        )
        pixel_v = (
            v_local.astype(np.float64)
            + y_min
        )
        point_z = roi[v_local, u_local]
        point_x = (
            (pixel_u - self.cx)
            * point_z
            / self.fx
        )
        point_y = (
            (pixel_v - self.cy)
            * point_z
            / self.fy
        )
        points = np.column_stack((
            point_x,
            point_y,
            point_z
        ))
        active = np.ones(
            points.shape[0],
            dtype=bool
        )

        for _ in range(6):
            center, radius = self.fit_sphere(
                points[active]
            )

            if center is None:
                return None, 'sphere_fit_failed'

            radial = np.linalg.norm(
                points - center,
                axis=1
            )
            residual = np.abs(radial - radius)
            active_residual = residual[active]
            median = float(
                np.median(active_residual)
            )
            mad = float(np.median(np.abs(
                active_residual - median
            )))
            threshold = max(
                0.00075,
                median + 3.5 * 1.4826 * mad
            )
            updated = residual <= threshold

            if (
                np.count_nonzero(updated)
                < self.roi_min_component_pixels
            ):
                break

            if np.array_equal(updated, active):
                break

            active = updated

        center, radius = self.fit_sphere(
            points[active]
        )

        if center is None:
            return None, 'sphere_fit_failed'

        radial_residual = (
            np.linalg.norm(
                points[active] - center,
                axis=1
            )
            - radius
        )
        rms = float(np.sqrt(np.mean(
            radial_residual ** 2
        )))

        if not (
            self.sphere_min_radius_m
            <= radius
            <= self.sphere_max_radius_m
        ):
            return None, 'sphere_radius_out_of_range'

        if rms > self.sphere_max_rms_m:
            return None, 'sphere_fit_rms_exceeded'

        projected_u = (
            self.fx * center[0] / center[2]
            + self.cx
        )
        projected_v = (
            self.fy * center[1] / center[2]
            + self.cy
        )

        return ({
            'center_u': float(projected_u),
            'center_v': float(projected_v),
            'camera_x': float(center[0]),
            'camera_y': float(center[1]),
            'camera_z': float(center[2]),
            'depth': float(center[2]),
            'localization_method': 'roi_sphere_fit',
            'sphere_radius_m': float(radius),
            'sphere_fit_rms_m': rms,
            'sphere_inlier_count': int(
                np.count_nonzero(active)
            ),
            'roi_foreground_count': int(
                points.shape[0]
            ),
            'surface_nearest_depth_m': float(
                np.min(point_z)
            ),
            'source_bbox': [x1, y1, x2, y2]
        }, None)

    def detection_callback(self, message):
        depth_ready = (
            bool(self.depth_history)
            if self.enable_time_sync
            else (
                self.depth_image is not None
                and self.depth_header is not None
            )
        )
        if not depth_ready:
            self.get_logger().warning(
                '尚未收到深度图'
            )
            return

        if None in (self.fx, self.fy, self.cx, self.cy):
            self.get_logger().warning(
                '尚未收到相机内参'
            )
            return

        try:
            input_data = json.loads(message.data)
        except json.JSONDecodeError as error:
            self.get_logger().error(
                f'检测结果 JSON 解析失败: {error}'
            )
            return

        detection_stamp = None
        detection_frame_id = None
        input_pairing = None

        if isinstance(input_data, list):
            detections = input_data
            stamped_input = False
        elif isinstance(input_data, dict):
            detections = input_data.get(
                'detections'
            )
            detection_stamp = input_data.get(
                'stamp'
            )
            detection_frame_id = input_data.get(
                'frame_id'
            )
            input_pairing = input_data.get(
                'input_pairing'
            )
            stamped_input = True
        else:
            self.get_logger().error(
                '检测结果必须是 JSON 列表或对象'
            )
            return

        if not isinstance(detections, list):
            self.get_logger().error(
                'detections 字段必须是 JSON 列表'
            )
            return

        def publish_sync_failure(reason, delta_sec=None):
            output = {
                'frame_id': self.optical_frame_id,
                'stamp': detection_stamp,
                'detection_frame_id': (
                    detection_frame_id
                ),
                'detection_stamp': detection_stamp,
                'sync': {
                    'enabled': True,
                    'valid': False,
                    'reason': reason,
                    'delta_sec': delta_sec,
                    'depth_stamp': None,
                    'exact_pairing_match': False,
                    'input_pairing': input_pairing,
                },
                'positions': []
            }

            output_message = String()
            output_message.data = json.dumps(
                output,
                ensure_ascii=False
            )
            self.position_publisher.publish(
                output_message
            )

            self.get_logger().warning(
                f'检测与深度同步失败: {reason}, '
                f'delta={delta_sec}',
                throttle_duration_sec=1.0
            )

        selected_depth_image = self.depth_image
        selected_depth_encoding = (
            self.depth_encoding
        )
        selected_depth_header = self.depth_header
        sync_delta_sec = None
        sync_reason = 'latest_depth_legacy'

        if self.enable_time_sync:
            if not stamped_input:
                publish_sync_failure(
                    'stamped_detection_required'
                )
                return

            if (
                detection_frame_id
                != self.optical_frame_id
            ):
                publish_sync_failure(
                    'unexpected_detection_frame'
                )
                return

            if self.require_exact_paired_depth:
                sample, sync_delta_sec, sync_reason = (
                    self.find_exact_depth_sample(
                        detection_stamp,
                        input_pairing
                    )
                )
            else:
                sample, sync_delta_sec, sync_reason = (
                    self.find_closest_depth_sample(
                        detection_stamp
                    )
                )

            if sample is None:
                publish_sync_failure(
                    sync_reason,
                    sync_delta_sec
                )
                return

            try:
                selected_depth_image = np.asarray(
                    self.bridge.imgmsg_to_cv2(
                        sample['message'],
                        desired_encoding='passthrough'
                    )
                )
            except Exception as error:
                self.get_logger().error(
                    '所选同步深度图转换失败: '
                    f'{error}'
                )
                publish_sync_failure(
                    'selected_depth_conversion_failed',
                    sync_delta_sec
                )
                return
            selected_depth_encoding = (
                sample['encoding']
            )
            selected_depth_header = (
                sample['header']
            )

        depth_stamp = {
            'sec': selected_depth_header.stamp.sec,
            'nanosec': (
                selected_depth_header.stamp.nanosec
            )
        }

        positions_3d = []

        for detection in detections:
            if not isinstance(detection, dict):
                continue

            if (
                'center_u' not in detection
                or 'center_v' not in detection
            ):
                continue

            try:
                center_u = int(
                    detection['center_u']
                )
                center_v = int(
                    detection['center_v']
                )
            except (TypeError, ValueError):
                continue

            sphere_result = None

            if self.localization_mode == 'roi_sphere':
                sphere_result, failure_reason = (
                    self.localize_roi_sphere(
                        detection,
                        selected_depth_image,
                        selected_depth_encoding
                    )
                )

                if sphere_result is None:
                    self.get_logger().warning(
                        'ROI 球面定位失败: '
                        f'{failure_reason}',
                        throttle_duration_sec=1.0
                    )
                    continue

                center_u = sphere_result['center_u']
                center_v = sphere_result['center_v']
                depth_z = sphere_result['depth']
            else:
                depth_z = self.read_depth_in_metres(
                    center_u,
                    center_v,
                    depth_image=selected_depth_image,
                    depth_encoding=(
                        selected_depth_encoding
                    )
                )

            if (
                depth_z is None
                or not math.isfinite(depth_z)
                or depth_z <= 0.0
            ):
                self.get_logger().warning(
                    f'像素 ({center_u}, {center_v}) '
                    '没有有效深度'
                )
                continue

            if sphere_result is not None:
                camera_x = sphere_result['camera_x']
                camera_y = sphere_result['camera_y']
                camera_z = sphere_result['camera_z']
            else:
                camera_x = (
                    (center_u - self.cx)
                    * depth_z
                    / self.fx
                )
                camera_y = (
                    (center_v - self.cy)
                    * depth_z
                    / self.fy
                )
                camera_z = depth_z

            position_3d = {
                'class_id': detection.get(
                    'class_id'
                ),
                'class_name': detection.get(
                    'class_name'
                ),
                'confidence': detection.get(
                    'confidence'
                ),
                'center_u': center_u,
                'center_v': center_v,
                'depth': camera_z,
                'camera_x': camera_x,
                'camera_y': camera_y,
                'camera_z': camera_z
            }

            if sphere_result is not None:
                position_3d.update({
                    key: value
                    for key, value
                    in sphere_result.items()
                    if key not in (
                        'center_u',
                        'center_v',
                        'camera_x',
                        'camera_y',
                        'camera_z',
                        'depth'
                    )
                })

            positions_3d.append(position_3d)

        output = {
            'frame_id': self.optical_frame_id,
            'stamp': depth_stamp,
            'positions': positions_3d
        }

        if self.enable_time_sync:
            output.update({
                'detection_frame_id': (
                    detection_frame_id
                ),
                'detection_stamp': detection_stamp,
                'sync': {
                    'enabled': True,
                    'valid': True,
                    'reason': sync_reason,
                    'delta_sec': sync_delta_sec,
                    'depth_stamp': depth_stamp,
                    'exact_pairing_match': (
                        self.require_exact_paired_depth
                    ),
                    'input_pairing': input_pairing,
                }
            })

        output_message = String()
        output_message.data = json.dumps(
            output,
            ensure_ascii=False
        )
        self.position_publisher.publish(
            output_message
        )

        current_time_ns = (
            self.get_clock().now().nanoseconds
        )

        if (
            current_time_ns - self.last_log_time_ns
            >= 1_000_000_000
        ):
            self.last_log_time_ns = current_time_ns

            if self.enable_time_sync:
                self.get_logger().info(
                    '时间同步成功: '
                    f'delta={sync_delta_sec:.6f} s'
                )

            self.get_logger().info(
                f'输出 {len(positions_3d)} '
                '个蘑菇三维坐标'
            )

            for index, position in enumerate(
                positions_3d
            ):
                self.get_logger().info(
                    f'目标 {index + 1}: '
                    f'像素=('
                    f'{position["center_u"]}, '
                    f'{position["center_v"]}), '
                    f'XYZ=('
                    f'{position["camera_x"]:.3f}, '
                    f'{position["camera_y"]:.3f}, '
                    f'{position["camera_z"]:.3f}) m'
                )


def main(args=None):
    rclpy.init(args=args)
    node = Mushroom3DLocalizer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
