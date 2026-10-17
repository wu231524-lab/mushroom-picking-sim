# Mushroom Picking Simulation / 蘑菇采摘仿真系统

基于 **ROS 2 Humble、Gazebo Classic、Franka FER 和双 RGB-D 视觉**的蘑菇自主采摘仿真项目。

系统通过全局视觉选择目标，在预抓取位置切换至末端 D405 视觉，使用 ROI 深度球面拟合和小步 XY 闭环完成对准，再执行接近、模拟真空吸附、提升、搬运、放置与归位。

**当前已验证：一次固定场景下 `mushroom_3` 的单颗端到端自主仿真采摘。下一目标：一次启动、一次触发，连续正确地自主完成多颗蘑菇采摘与放置。** 多颗循环目前尚未实现。

## 已验证结果与当前边界

- 2026-09-27 的 DAY19 运行记录：Ubuntu 三包构建成功，默认无运动验收通过，一次主控制器运行到达 `DONE`。
- 完整运行包含 68 条关节轨迹、一次吸附和一次脱附；机械臂回到 HOME，蘑菇放置于 `(0.500, -0.300, 0.230) m`。
- D405 根据新观测生成 XY 修正；预抓取、Z 接近、提升、搬运和返回仍主要使用 DAY13 已验证的固定关节路线。
- 仿真吸附由带几何接触判断的 Gazebo entity-state follower 实现，机器人使用简化外观与碰撞、关闭连杆重力。
- 当前结果不代表任意目标规划、多颗连续采摘、高保真吸附力学或实机验证。后续开发以多颗连续采摘为优先。

详见 [项目状态与路线](docs/PROJECT_STATUS.md)。

## 技术栈与结构

| 层次 | 技术 |
| --- | --- |
| 仿真环境 | Ubuntu 22.04、ROS 2 Humble、Gazebo Classic 11 |
| 机器人与控制 | Franka FER 七轴模型、URDF/Xacro、ros2_control、FollowJointTrajectory |
| 感知 | Ultralytics YOLO、RGB-D、OpenCV、cv_bridge、NumPy、TF2 |
| 对准与执行 | ROI 球面拟合、三维最近邻跟踪、雅可比阻尼最小二乘、状态机 |
| 构建与测试 | colcon、ament_python、setuptools、unittest |

```text
src/
├── mushroom_gazebo/        # 菇床、蘑菇世界与固定相机模型
├── mushroom_vision/        # YOLO、深度定位、D405 对准与建议
│   └── models/best.pt      # 随包安装的检测权重
└── mushroom_panda_gazebo/  # FER、控制器、主控制器、运动学与测试
docs/
├── PROJECT_STATUS.md
└── ARCHITECTURE.md
```

虽然包名包含 `panda`，当前模型及关节名称使用 Franka FER 的 `fer_joint1` 至 `fer_joint7`。全局相机沿用 `/wrist_camera` 接口，承担 D435 类任务；末端相机使用 `/d405` 接口。

## Ubuntu 安装与构建

以下命令在 Ubuntu 22.04 / ROS 2 Humble 环境执行。Windows 用于源码开发，完整仿真在 Ubuntu 中运行。

```bash
git clone https://github.com/wu231524-lab/mushroom-picking-sim.git
cd mushroom-picking-sim
source /opt/ros/humble/setup.bash

sudo apt update
sudo apt install -y python3-colcon-common-extensions python3-rosdep python3-pip
```

如果这台 Ubuntu 尚未初始化 rosdep，先运行一次 `sudo rosdep init`，随后安装依赖并构建：

```bash
rosdep update
rosdep install --from-paths src --ignore-src -r -y
python3 -m pip install --user -r requirements.txt
colcon build --symlink-install
source install/setup.bash
```

工作区依赖 Gazebo Classic ROS 插件、gazebo_ros2_control、franka_description 和相应控制器，具体依赖见三个包的 `package.xml`。YOLO 权重会安装到 `mushroom_vision` 的 `share/models/best.pt`。

## 启动与单次运行

### 默认无运动模式

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch mushroom_panda_gazebo pick_system.launch.py gui:=false
```

`gui:=false` 保持 Gazebo 服务端、相机和物理仿真运行，仅关闭图形客户端。需要显示 Gazebo 时使用 `gui:=true`。

在另一个已加载相同 ROS 与工作区环境的终端触发：

```bash
ros2 service call /mushroom_pick/run_once std_srvs/srv/Trigger '{}'
```

默认模式使用实时关节和全局目标证据，在进入首个运动状态 `MOVE_PREGRASP` 前结束为 `DRY_RUN_COMPLETE`；它不会执行采摘，也不会验证后续真实运动流程。

### 启用一次有运动仿真

先正常关闭上面的 launch，再重新启动：

```bash
ros2 launch mushroom_panda_gazebo pick_system.launch.py execution_enabled:=true gui:=false
```

启动本身不会开始采摘。在另一个已加载环境的终端调用一次：

```bash
ros2 service call /mushroom_pick/run_once std_srvs/srv/Trigger '{}'
```

读取状态报告：

```bash
ros2 service call /mushroom_pick/day15/get_report std_srvs/srv/Trigger '{}'
```

当前每个控制器进程只接受一次运行请求。完成或失败后应正常关闭并重新启动；连续多颗功能将由后续批次状态机实现。状态话题为 `/mushroom_pick/day15/state`。

## 本地单元测试

现有 5 个测试文件包含 14 项 DAY19 回归测试，可不启动 ROS/Gazebo 运行；需要 Python 与 NumPy：

```bash
PYTHONPATH="$PWD/src/mushroom_panda_gazebo" python3 -m unittest discover \
  -s src/mushroom_panda_gazebo/test -v
```

这些测试覆盖目标时钟、慢仿真期限、D405 启动、观测配对和 XY/Z 交接条件；完整仿真仍需在 Ubuntu 中运行验证。

## 下一阶段

1. 根据目标位置生成运动，支持切换吸附对象，完成第二个位置的单颗采摘。
2. 实现未采摘/处理中/已放置/跳过目标管理，区分采摘区和放置区。
3. 一次触发连续两颗，再连续现有四颗，整批结束后归位。
4. 重复整批运行，针对实际阻塞改进感知和执行。

保持必要的执行约束，不将全面审查、外部包装或高保真物理作为多颗开发前置条件。

## 模型与许可证

Copyright (C) 2026 wu231524-lab.

本仓库的项目代码及随附 `best.pt` 检测权重按 **GNU Affero General Public License v3.0（AGPL-3.0-only）**发布，全文见 [LICENSE](LICENSE)。模型用途与已知信息见 [模型说明](src/mushroom_vision/models/README.md)。

Ultralytics、ROS 2、Gazebo、Franka 描述包等第三方依赖保留各自许可证。来源与说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
