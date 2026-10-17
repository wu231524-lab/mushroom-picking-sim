import re

import xacro
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    model_path = LaunchConfiguration('model_path')
    execution_enabled = LaunchConfiguration('execution_enabled')
    gui = LaunchConfiguration('gui')
    package_share = get_package_share_directory(
        'mushroom_panda_gazebo'
    )

    gazebo_share = get_package_share_directory('gazebo_ros')
    mushroom_gazebo_share = get_package_share_directory(
        'mushroom_gazebo'
    )

    world_file = (
        mushroom_gazebo_share
        + '/worlds/mushroom_field.world'
    )
    camera_sdf_file = (
        mushroom_gazebo_share
        + '/models/wrist_camera/model.sdf'
    )

    xacro_file = (
        package_share + '/urdf/fer_gazebo.urdf.xacro'
    )

    controllers_file = (
        package_share + '/config/ros2_controllers.yaml'
    )

    robot_description_document = xacro.process_file(
        xacro_file,
        mappings={
            'controllers_file': controllers_file,
            'enable_control': 'true',
            'base_x': '0.30',
            'base_y': '-0.65',
            'base_z': '0.0',
            'base_yaw': '1.57079632679',
        },
    )

    robot_description_content = (
        robot_description_document.toxml()
    )

    # DAY8：移除 FER 原始网格碰撞体，避免轻量球形模型的
    # 碰撞接触导致 Gazebo 物理解算发散。
    def remove_mesh_collision(match):
        collision_block = match.group(0)
        if '<mesh' in collision_block:
            return ''
        return collision_block

    robot_description_content = re.sub(
        r'<collision\b[^>]*>.*?</collision>',
        remove_mesh_collision,
        robot_description_content,
        flags=re.DOTALL,
    )

    # 剩余 mesh 均为视觉网格，将其替换为轻量球体。
    robot_description_content = re.sub(
        r'<mesh\s+filename="[^"]+"\s*/>',
        '<sphere radius="0.04"/>',
        robot_description_content,
    )

        # gazebo_ros2_control 在当前 ROS 2 Humble 环境中不能把
    # 带 XML 声明、注释和大量换行的 URDF解析为参数覆盖值。
    robot_description_content = re.sub(
        r'<\?xml[^>]*\?>',
        '',
        robot_description_content,
    )

    robot_description_content = re.sub(
        r'<!--.*?-->',
        '',
        robot_description_content,
        flags=re.DOTALL,
    )

    robot_description_content = re.sub(
        r'>\s+<',
        '><',
        robot_description_content,
    ).strip()

    # 保存本次实际传给 Gazebo 的描述，便于检查。
    with open(
        '/tmp/mushroom_pick_demo_runtime.urdf',
        'w',
        encoding='utf-8',
    ) as runtime_urdf:
        runtime_urdf.write(robot_description_content)

    if '<mesh' in robot_description_content:
        raise RuntimeError(
            'Mesh replacement failed; refusing to start Gazebo.'
        )

    robot_description = {
        'robot_description': ParameterValue(
            robot_description_content,
            value_type=str,
        )
    }

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            gazebo_share + '/launch/gazebo.launch.py'
        ),
        launch_arguments={
            'world': world_file,
            'pause': 'false',
            'verbose': 'false',
            'gui': gui,
        }.items(),
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[robot_description],
    )

    spawn_robot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        output='screen',
        arguments=[
            '-entity', 'mushroom_fer_day8',
            '-topic', 'robot_description',
            '-x', '0.0',
            '-y', '0.0',
            '-z', '0.0',
        ],
    )

    spawn_camera = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        output='screen',
        arguments=[
            '-entity', 'wrist_camera',
            '-file', camera_sdf_file,
            '-x', '0.14',
            '-y', '0.95',
            '-z', '0.90',
            '-R', '0.0',
            '-P', '-0.78539816339',
            '-Y', '-1.57079632679',
        ],
    )

    camera_optical_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='wrist_camera_optical_tf',
        output='screen',
        arguments=[
            '--x', '0.14',
            '--y', '0.95',
            '--z', '0.90',
            '--roll', '-2.35619449019',
            '--pitch', '0',
            '--yaw', '3.14159265359',
            '--frame-id', 'world',
            '--child-frame-id',
            'wrist_camera_optical_frame',
        ],
    )
    mushroom_detector = Node(
        package='mushroom_vision',
        executable='mushroom_detector',
        name='mushroom_detector',
        output='screen',
        parameters=[{
            'model_path': model_path,
            'image_topic': (
                '/wrist_camera/camera/image_raw'
            ),
            'confidence_threshold': 0.5,
        }],
    )

    mushroom_3d_localizer = Node(
        package='mushroom_vision',
        executable='mushroom_3d_localizer',
        name='mushroom_3d_localizer',
        output='screen',
        parameters=[{
            'detection_topic': (
                '/mushroom_detection/detections'
            ),
            'depth_topic': (
                '/wrist_camera/camera/depth/image_raw'
            ),
            'camera_info_topic': (
                '/wrist_camera/camera/depth/camera_info'
            ),
            'output_topic': (
                '/mushroom_detection/positions_3d'
            ),
            'optical_frame_id': (
                'wrist_camera_optical_frame'
            ),
            'depth_window_size': 5,
        }],
    )

    mushroom_world_localizer = Node(
        package='mushroom_vision',
        executable='mushroom_world_localizer',
        name='mushroom_world_localizer',
        output='screen',
        parameters=[{
            'input_topic': (
                '/mushroom_detection/positions_3d'
            ),
            'output_topic': (
                '/mushroom_detection/positions_world'
            ),
            'source_frame': (
                'wrist_camera_optical_frame'
            ),
            'target_frame': 'world',
        }],
    )

    mushroom_target_selector = Node(
        package='mushroom_panda_gazebo',
        executable='mushroom_target_selector',
        name='mushroom_target_selector',
        output='screen',
        parameters=[{
            'input_topic': (
                '/mushroom_detection/positions_world'
            ),
            'output_topic': (
                '/mushroom_pick/selected_target'
            ),
            'world_frame': 'world',
            'base_frame': 'fer_link0',
            'min_world_z': 0.34,
            'max_world_z': 0.45,
            'mushroom3_x': 0.30,
            'mushroom3_y': -0.12,
            'mushroom3_radius': 0.10,
            'compensation_x': 0.0087,
            'compensation_y': -0.0084,
            'compensation_z': 0.0,
        }],
    )

    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        output='screen',
        arguments=[
            'joint_state_broadcaster',
            '--controller-manager', '/controller_manager',
            '--controller-manager-timeout', '120',
            '--service-call-timeout', '30',
        ],
    )

    arm_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        output='screen',
        arguments=[
            'fer_arm_controller',
            '--controller-manager', '/controller_manager',
            '--controller-manager-timeout', '120',
            '--service-call-timeout', '30',
        ],
    )

    hand_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        output='screen',
        arguments=[
            'fer_hand_controller',
            '--controller-manager', '/controller_manager',
            '--controller-manager-timeout', '120',
            '--service-call-timeout', '30',
        ],
    )

    mushroom_follower = Node(
        package='mushroom_panda_gazebo',
        executable='mushroom_follower',
        name='mushroom_follower',
        output='screen',
        parameters=[{
            'entity_name': 'mushroom_3',
            'world_frame': 'world',
            'tcp_frame': 'fer_hand_tcp',
            'update_rate': 30.0,
            'vacuum_validation_enabled': True,
            'attachment_commands_enabled': ParameterValue(
                execution_enabled,
                value_type=bool,
            ),
            'vacuum_cap_center_z_m': 0.095,
            'vacuum_cap_radius_m': 0.070,
            'vacuum_max_planar_error_m': 0.003,
            'vacuum_min_surface_gap_m': -0.002,
            'vacuum_max_surface_gap_m': 0.005,
            'vacuum_max_axis_error_rad': (
                0.17453292519943295
            ),
            'vacuum_max_model_state_age_sec': 0.50,
        }],
    )

    start_joint_state_broadcaster = RegisterEventHandler(
        OnProcessExit(
            target_action=spawn_robot,
            on_exit=[joint_state_broadcaster_spawner],
        )
    )

    start_arm_controller = RegisterEventHandler(
        OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[arm_controller_spawner],
        )
    )

    start_hand_controller = RegisterEventHandler(
        OnProcessExit(
            target_action=arm_controller_spawner,
            on_exit=[hand_controller_spawner],
        )
    )

    start_mushroom_follower = RegisterEventHandler(
        OnProcessExit(
            target_action=hand_controller_spawner,
            on_exit=[mushroom_follower],
        )
    )

    start_vision_nodes = RegisterEventHandler(
        OnProcessExit(
            target_action=spawn_camera,
            on_exit=[
                mushroom_detector,
                mushroom_3d_localizer,
                mushroom_world_localizer,
                mushroom_target_selector,
            ],
        )
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
            description=(
                'Enable attachment commands for the explicitly triggered '
                'single simulated run.'
            ),
        ),
        DeclareLaunchArgument(
            'gui',
            default_value='true',
            description='Show Gazebo client; false runs only gzserver.',
        ),
        gazebo,
        robot_state_publisher,
        camera_optical_tf,
        start_vision_nodes,
        spawn_robot,
        spawn_camera,
        start_joint_state_broadcaster,
        start_arm_controller,
        start_hand_controller,
        start_mushroom_follower,
    ])
