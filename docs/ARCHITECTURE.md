# 系统架构

## 三个 ROS 2 包

| 包 | 职责 |
| --- | --- |
| mushroom_gazebo | 菇床与四颗蘑菇世界、固定 RGB-D 相机资源 |
| mushroom_vision | YOLO 检测、深度反投影、world 定位、D405 对准与小步建议 |
| mushroom_panda_gazebo | FER 模型、控制器、顶层 launch、吸附跟随、运动学和主状态机 |

## 启动结构

```text
pick_system.launch.py
├── mushroom_pick_demo.launch.py
│   ├── Gazebo world / FER / ros2_control
│   ├── 全局 RGB-D / YOLO / 3D / world / selector
│   └── mushroom_follower
├── d405_perception.launch.py
│   ├── D405 YOLO / paired-depth
│   ├── ROI sphere localizer
│   ├── TCP alignment / target lock
│   └── step proposal
└── pick_controller
```

全局 detector 默认推理开启，D405 detector 默认关闭。主控制器到达 PREGRASP 并停稳后切换推理，不依赖外部 shell 管理业务状态。

## 感知与运动

1. 全局 YOLO 输出框中心，以邻域深度中位数及内参反投影为相机点，经 TF2 转换到 world。
2. selector 根据高度、距基座距离和当前 `mushroom_3` 名义区域选择目标。
3. 固定关节路线将 FER 移到 PREGRASP。
4. D405 在 YOLO 推理前选择时间差不超过 0.20 秒的深度帧；低频 paired-depth 使用有界 RELIABLE 队列，下游精确匹配该时间戳。
5. 检测 ROI 内通过深度阈值和最大连通区域提取点簇，最小二乘球面拟合配合 MAD 迭代筛选获得球心和质量指标。
6. 目标以局部三维最近邻保持连续，经 TF2 转到 TCP，生成平面误差。
7. proposal 等待两次新的稳定同步观测，保持单步、无执行权限和 Z=0；主控制器用雅可比阻尼最小二乘求关节增量，并用 FK 复核。
8. 对准交接后，Z、提升、搬运、放置和返回沿固定的已验证关节路线执行。

`d405_step_executor` 是保留的手动单步组件；顶层主链的 XY 由 `pick_controller` 直接协调，不启动该独立 executor。

## 状态与执行

ROS 入口位于 `day15_autonomous_pick.py`，console executable 为 `pick_controller`。文件名中的 DAY15 是实现沿革，当前包含 DAY19 的集成修复。

- `pick_state.py` / `pick_state_machine_core.py`：状态名和无外部副作用的转换。
- `day15_autonomous_runtime.py` / `day15_command_runtime.py`：单次运行与单条在途命令账本。
- `day15_runtime_evidence.py` / `target_handoff.py`：目标、观测、几何和附着证据。
- `day15_xy_kinematics.py`：URDF 运动链、FK、Jacobian 与 DLS。
- `day15_verified_trajectories.py`：63 条固定轨迹点，加最多 5 条动态 XY，共最多 68 条轨迹。
- `day15_state_deadlines.py`：单调墙钟状态期限和有界慢仿真倍率。

默认 live Dry Run 在 `MOVE_PREGRASP` 前结束为 `DRY_RUN_COMPLETE`。`day15_autonomous_replay.py` 可用于纯内存完整路径回放；回放 `DONE` 不代表执行过仿真运动。

## 吸附模型与接口

follower 根据 TCP 和蘑菇实体位姿计算平面误差、表面间隙与轴向误差。接触门通过后记录相对位姿，并以 entity-state 更新让蘑菇随末端运动；脱附后停止跟随。

| 接口 | 用途 |
| --- | --- |
| /mushroom_pick/run_once | 当前进程的一次采摘或 live Dry Run |
| /mushroom_pick/day15/get_report | 最新状态报告 |
| /fer_arm_controller/follow_joint_trajectory | 七轴轨迹 Action |
| /mushroom_pick/check_vacuum_contact | 几何接触查询 |
| /mushroom_pick/set_attached | 设置模拟吸附/脱附 |
| /mushroom_pick/get_attached | 附着状态查询 |

`get_attached` 使用非对称 Trigger 语义：`success=true` 且消息为 attached 表示附着；`success=false` 且消息为 detached 表示未附着。应同时读取字段与消息。
