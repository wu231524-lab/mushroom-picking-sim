# Third-party components

The project code and the included mushroom detector checkpoint are distributed under AGPL-3.0-only, as selected by the project maintainer on 2026-10-17. See [LICENSE](LICENSE). This does not replace the licenses of independent third-party dependencies.

| Component | Use in this repository | Upstream source / license information |
| --- | --- | --- |
| Ultralytics YOLO | Detector API and model inference | [Source and AGPL license](https://github.com/ultralytics/ultralytics), [licensing guidance](https://docs.ultralytics.com/help/contributing/#open-sourcing-your-project-under-agpl-30) |
| PyTorch | Inference backend used by Ultralytics | [Source and license](https://github.com/pytorch/pytorch) |
| ROS 2 / rclpy / TF2 / ros2_control | Communication, transforms and robot control | [ROS 2](https://github.com/ros2), [geometry2](https://github.com/ros2/geometry2), [ros2_control](https://github.com/ros-controls/ros2_control); consult the individual installed package licenses |
| Gazebo Classic / gazebo_ros_pkgs / gazebo_ros2_control | Simulator, RGB-D and state/control plugins | [Gazebo Classic](https://github.com/gazebosim/gazebo-classic), [ROS plugins](https://github.com/ros-simulation/gazebo_ros_pkgs), [control plugin](https://github.com/ros-controls/gazebo_ros2_control) |
| franka_description | FER kinematics, limits and inertial descriptions loaded by Xacro | [Source and license](https://github.com/frankarobotics/franka_description); installed as an external ROS dependency |
| OpenCV / cv_bridge | Image conversion and connected components | [OpenCV](https://github.com/opencv/opencv), [vision_opencv](https://github.com/ros-perception/vision_opencv) |
| NumPy | Point-cloud fitting and robot kinematics | [Source and license](https://github.com/numpy/numpy) |

These dependencies are installed separately; their source trees and robot meshes are not vendored here. Preserve the applicable upstream notices when redistributing them. Gazebo's `sun` and `ground_plane` models are resolved from the user's installed model resources.

The maintainer confirmed permission to redistribute `src/mushroom_vision/models/best.pt`. Training data, training scripts and evaluation metadata are not included in this repository; see the [model note](src/mushroom_vision/models/README.md). No upstream component is presented as an original contribution of this project.
