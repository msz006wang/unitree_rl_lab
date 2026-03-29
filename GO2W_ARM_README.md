# GO2W-Arm 训练模型配置 (基于robot_lab_locomanip升级)

## 概述

基于项目中的GO2W训练和推理模型，并为go2w_arm创建了完整的训练模型配置。已完成基于robot_lab_locomanip项目的参数和策略升级。

## 文件结构

```
/home/jay/unitree_rl_lab/
├── source/unitree_rl_lab/unitree_rl_lab/
│   ├── assets/robots/
│   │   └── unitree.py                         # 添加了GO2W-Arm机器人配置
│   └── tasks/locomotion/robots/
│       └── go2w_arm/                         # 新建的go2w_arm目录
│           ├── __init__.py                    # 环境注册
│           ├── velocity_env_cfg.py            # ARX5机械臂配置 (默认)
│           └── velocity_env_cfg_piper.py     # Piper机械臂配置
├── scripts/
│   └── train_go2w_arm.sh                  # 训练脚本
└── test_go2w_arm_simple.py                 # 配置测试脚本
```

## 支持的机械臂

### 1. ARX5 机械臂
- **配置文件**: `velocity_env_cfg.py`
- **URDF路径**: `~/isaac_project/unitree_ros/robots/go2w_arm_description/urdf/go2w_arx5_description.urdf`
- **关节配置**:
  - 6个机械臂关节: `arm_joint1` 到 `arm_joint6`
  - 包含机械爪 (gripper)
  - 最大力矩: 15.0 Nm
  - 最大速度: 10.0 rad/s

### 2. Piper 机械臂
- **配置文件**: `velocity_env_cfg_piper.py`
- **URDF路径**: `~/isaac_project/unitree_ros/robots/go2w_arm_description/urdf/go2w_piper_description.urdf`
- **关节配置**:
  - 6个机械臂关节: `arm_joint1` 到 `arm_joint6`
  - 包含双爪机械爪 (2 prismatic joints)
  - 最大力矩: 20.0 Nm
  - 最大速度: 10.0 rad/s

## 训练环境

### 已注册的Gym环境

1. **Unitree-Go2WArm-Velocity-Flat-v0**
   - 平地地形
   - 适用于初期训练和快速原型验证
   - 禁用高度扫描和地形课程

2. **Unitree-Go2WArm-Velocity-Rough-v0**
   - 粗糙地形
   - 使用地形生成器创建复杂地形
   - 包含地形难度课程学习

3. **Unitree-Go2WArm-Velocity**
   - 传统兼容版本（等同于粗糙地形）

## 训练策略（基于GO2W）

### 关键特性

1. **轮腿混合控制**:
   - 12个腿部关节（hip, thigh, calf）
   - 4个轮子关节（continuous joints）
   - 6个机械臂关节

2. **动作空间**:
   - 腿部关节: 位置控制，scale=0.25
   - 轮子关节: 速度控制，scale=5.0
   - 机械臂关节: 位置控制，scale=0.2

3. **观测空间**:
   - 基础线速度和角速度
   - 投影重力
   - 速度命令
   - 关节位置和速度（排除轮子位置）
   - 前一个动作
   - 高度扫描（仅粗糙地形）

4. **奖励函数**:
   - 线速度跟踪: weight=3.0
   - 角速度跟踪: weight=1.5
   - 姿态保持: weight=1.0
   - 关节扭矩惩罚: weight=-2.5e-5
   - 关节加速度惩罚: weight=-2.5e-7
   - 动作率惩罚: weight=-0.01
   - 对称性奖励: weight=-0.05

## 使用方法

### 1. 选择机械臂配置

如果要使用不同的机械臂，需要修改导入：

**使用ARX5**（默认）:
```python
from unitree_rl_lab.assets.robots.unitree import (
    UNITREE_GO2W_ARM_ARX5_CFG as ROBOT_CFG
)
```

**使用Piper**:
```python
from unitree_rl_lab.assets.robots.unitree import (
    UNITREE_GO2W_ARM_PIPER_CFG as ROBOT_CFG
)
```

或者将 `velocity_env_cfg_piper.py` 重命名为 `velocity_env_cfg.py`

### 2. 训练命令

**使用训练脚本**:
```bash
cd /home/jay/unitree_rl_lab
./scripts/train_go2w_arm.sh arx5_flat      # ARX5平地训练
./scripts/train_go2w_arm.sh arx5_rough     # ARX5粗糙地形训练
./scripts/train_go2w_arm.sh piper_flat      # Piper平地训练
./scripts/train_go2w_arm.sh piper_rough     # Piper粗糙地形训练
```

**直接使用train.py**:
```bash
python /home/jay/unitree_rl_lab/scripts/rsl_rl/train.py \
    --task Unitree-Go2WArm-Velocity-Flat-v0 \
    --headless
```

### 3. 推理/播放

```bash
python /home/jay/unitree_rl_lab/scripts/rsl_rl/play.py \
    --task Unitree-Go2WArm-Velocity-Flat-v0 \
    --checkpoint path/to/checkpoint.pth
```

## 机器人配置详情

### 初始姿态
```python
init_state=ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.4),  # 初始高度
    joint_pos={
        # 腿部姿态（与GO2W相同）
        ".*L_hip_joint": 0.1,
        ".*R_hip_joint": -0.1,
        "F[L,R]_thigh_joint": 0.8,
        "R[L,R]_thigh_joint": 1.0,
        ".*_calf_joint": -1.5,
        ".*_foot_joint": 0.0,
        # 机械臂初始姿态
        "arm_joint1": 0.0,
        "arm_joint2": 0.0,
        "arm_joint3": 0.0,
        "arm_joint4": 0.0,
        "arm_joint5": 0.0,
        "arm_joint6": 0.0,
    },
)
```

### 执行器配置
```python
actuators={
    "legs": DCMotorCfg(      # 12个腿部关节
        joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
        effort_limit=23.5,
        velocity_limit=30.0,
        stiffness=25.0,
        damping=0.5,
    ),
    "wheels": ImplicitActuatorCfg(  # 4个轮子关节
        joint_names_expr=[".*_foot_joint"],
        effort_limit=23.5,
        velocity_limit=30.0,
        stiffness=0.0,
        damping=0.5,
    ),
    "arm": DCMotorCfg(       # 6个机械臂关节
        joint_names_expr=["arm_joint1", "arm_joint2", "arm_joint3",
                       "arm_joint4", "arm_joint5", "arm_joint6"],
        effort_limit=20.0,  # Piper: 20.0, ARX5: 15.0
        velocity_limit=10.0,
        stiffness=25.0,
        damping=0.5,
    ),
}
```

## 与GO2W的主要区别

| 特性 | GO2W | GO2W-Arm |
|------|-------|-----------|
| 关节总数 | 16 (12腿 + 4轮) | 22 (12腿 + 4轮 + 6臂) |
| 机械臂 | 无 | ARX5 或 Piper (6关节) |
| 腿部配置 | 完全相同 | 完全相同 |
| 训练策略 | 轮腿混合 | 轮腿混合 + 机械臂控制 |
| 奖励函数 | 轮腿专用 | 轮腿专用（机械臂无特定奖励） |

## 测试

运行配置测试脚本：
```bash
python /home/jay/unitree_rl_lab/test_go2w_arm_simple.py
```

测试内容包括：
- 文件结构验证
- 机器人配置导入
- 速度环境配置导入
- URDF文件存在性检查

## 注意事项

1. **机械臂控制**: 当前配置中，机械臂关节包含在训练中，但没有特定的机械臂奖励函数。训练可能会让机械臂保持某个姿态以减少对腿部控制的影响。

2. **模型路径**: 确保URDF文件路径正确：
   ```
   ~/isaac_project/unitree_ros/robots/go2w_arm_description/
   ```

3. **切换机械臂**: 如需在ARX5和Piper之间切换，修改配置文件中的导入语句或重命名配置文件。

4. **训练资源**: 由于增加了6个机械臂关节，训练可能需要更多的计算资源。

## 未来改进方向

1. 添加机械臂特定的奖励函数
2. 实现机械臂任务（如抓取、操作）
3. 支持机械臂轨迹跟踪
4. 添加机械臂-腿部协调任务
5. 实现移动操作（mobile manipulation）策略

## 参考资料

- [GO2W训练指南](GO2W_TRAINING_GUIDE.md)
- [IsaacLab文档](https://isaac-sim.github.io/IsaacLab/)
- [Unitree机器人官网](https://www.unitree.com/)
