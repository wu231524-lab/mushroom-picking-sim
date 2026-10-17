"""Launch the complete simulator, both perception stages, and one controller."""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    package_share = get_package_share_directory('mushroom_panda_gazebo')
    model_path = LaunchConfiguration('model_path')
    execution_enabled = LaunchConfiguration('execution_enabled')
    gui = LaunchConfiguration('gui')

    scene = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            package_share + '/launch/mushroom_pick_demo.launch.py'
        ),
        launch_arguments={
            'model_path': model_path,
            'execution_enabled': execution_enabled,
            'gui': gui,
        }.items(),
    )
    d405 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            package_share + '/launch/d405_perception.launch.py'
        ),
        launch_arguments={
            'model_path': model_path,
            'detector_enabled': 'false',
        }.items(),
    )
    controller = Node(
        package='mushroom_panda_gazebo',
        executable='pick_controller',
        name='mushroom_pick_controller',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'execution_enabled': ParameterValue(
                execution_enabled,
                value_type=bool,
            ),
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'model_path',
            default_value=(
                get_package_share_directory('mushroom_vision')
                + '/models/best.pt'
            ),
        ),
        DeclareLaunchArgument(
            'execution_enabled',
            default_value='false',
            description='Enable physical commands; /mushroom_pick/run_once '
                        'still starts exactly one run.',
        ),
        DeclareLaunchArgument(
            'gui',
            default_value='true',
            description='Show Gazebo client; false keeps the simulator headless.',
        ),
        scene,
        d405,
        controller,
    ])
