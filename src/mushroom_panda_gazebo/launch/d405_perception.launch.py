"""Start close-range D405 perception without an execution controller."""

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    model_path = LaunchConfiguration("model_path")
    detector_enabled = LaunchConfiguration("detector_enabled")

    return LaunchDescription([
        DeclareLaunchArgument(
            "model_path",
            default_value=(
                get_package_share_directory('mushroom_vision')
                + '/models/best.pt'
            ),
        ),
        DeclareLaunchArgument(
            "detector_enabled",
            default_value="false",
        ),
        Node(
            package="mushroom_vision",
            executable="mushroom_detector",
            name="d405_mushroom_detector",
            output="screen",
            parameters=[{
                "model_path": model_path,
                "image_topic": "/d405/camera/image_raw",
                "confidence_threshold": 0.5,
                "include_image_metadata": True,
                "image_queue_depth": 1,
                "enable_depth_pairing": True,
                "depth_topic": "/d405/camera/depth/image_raw",
                "paired_depth_topic":
                    "/d405/mushroom_detection/paired_depth",
                "depth_pair_history_size": 120,
                "paired_depth_queue_depth": 4,
                "max_input_sync_delta_sec": 0.20,
                "inference_enabled": ParameterValue(
                    detector_enabled,
                    value_type=bool,
                ),
            }],
            remappings=[
                ("/mushroom_detection/detections",
                 "/d405/mushroom_detection/detections"),
                ("/mushroom_detection/annotated_image",
                 "/d405/mushroom_detection/annotated_image"),
            ],
        ),
        Node(
            package="mushroom_vision",
            executable="mushroom_3d_localizer",
            name="d405_roi_sphere_localizer",
            output="screen",
            parameters=[{
                "use_sim_time": True,
                "detection_topic": "/d405/mushroom_detection/detections",
                "depth_topic":
                    "/d405/mushroom_detection/paired_depth",
                "camera_info_topic": "/d405/camera/depth/camera_info",
                "output_topic": "/d405/mushroom_detection/positions_3d",
                "optical_frame_id": "d405_optical_frame",
                "enable_time_sync": True,
                "depth_history_size": 120,
                "reliable_depth_transport": True,
                "require_exact_paired_depth": True,
                "depth_transport_queue_depth": 4,
                "max_sync_delta_sec": 0.20,
                "localization_mode": "roi_sphere",
                "roi_foreground_max_depth_m": 0.30,
                "roi_min_component_pixels": 1000,
                "sphere_min_radius_m": 0.05,
                "sphere_max_radius_m": 0.09,
                "sphere_max_rms_m": 0.001,
            }],
        ),
        Node(
            package="mushroom_vision",
            executable="d405_tcp_alignment",
            name="d405_roi_sphere_alignment",
            output="screen",
            parameters=[{
                "use_sim_time": True,
                "input_topic": "/d405/mushroom_detection/positions_3d",
                "output_topic": "/d405/mushroom_pick/sphere_alignment_error",
                "expected_source_frame": "d405_optical_frame",
                "target_frame": "fer_hand_tcp",
                "require_valid_sync": True,
                "lock_target_after_first_selection": True,
                "max_tracking_distance_m": 0.016,
                "reset_session_service":
                    "/d405/mushroom_pick/reset_alignment_session",
                "proposal_topic": "/d405/mushroom_pick/step_proposal",
            }],
        ),
        Node(
            package="mushroom_vision",
            executable="d405_step_proposal",
            name="d405_step_proposal",
            output="screen",
            parameters=[{
                "use_sim_time": True,
                "input_topic": "/d405/mushroom_pick/sphere_alignment_error",
                "output_topic": "/d405/mushroom_pick/step_proposal",
                "start_service": "/d405/mushroom_pick/start_observation",
                "get_last_service": "/d405/mushroom_pick/get_last_proposal",
                "reset_session_service":
                    "/d405/mushroom_pick/reset_session",
                "alignment_reset_service":
                    "/d405/mushroom_pick/reset_alignment_session",
                "reset_ack_timeout_sec": 5.0,
                "expected_target_frame": "fer_hand_tcp",
                "required_observations": 2,
                "observation_timeout_sec": 60.0,
                "max_step_m": 0.002,
                "deadband_m": 0.002,
                "max_spread_m": 0.001,
                "allow_transient_sync_miss": True,
            }],
        ),
    ])
