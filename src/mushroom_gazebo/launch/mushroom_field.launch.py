import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node


def generate_launch_description():
    package_share_dir = get_package_share_directory(
        'mushroom_gazebo'
    )

    world_file = os.path.join(
        package_share_dir,
        'worlds',
        'mushroom_field.world'
    )

    return LaunchDescription([
        ExecuteProcess(
            cmd=[
                'gazebo',
                '--verbose',
                world_file
            ],
            output='screen'
        ),

        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='wrist_camera_optical_tf',
            arguments=[
                '--x', '0',
                '--y', '0',
                '--z', '2.0',
                '--roll', '3.14159265359',
                '--pitch', '0',
                '--yaw', '-1.57079632679',
                '--frame-id', 'world',
                '--child-frame-id',
                'wrist_camera_optical_frame'
            ],
            output='screen'
        )
    ])
