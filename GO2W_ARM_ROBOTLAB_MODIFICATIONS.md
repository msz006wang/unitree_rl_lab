# GO2W-Arm Robot_lab_locomanip参数迁移

## 🔄 迁移概述

基于`robot_lab_locomanip`项目的参数和策略，对GO2W-Arm项目进行了全面升级，从纯轮腿移动配置迁移到移动操作（loco-manipulation）支持。

## 📋 已完成的修改清单

### 1. 执行器配置升级 ⚡

#### 修改文件: [`unitree.py`](source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py)

#### 主要修改内容：

##### 1.1 导入升级
```python
# 修改前
from isaaclab.actuators import DCMotorCfg, IdealPDActuatorCfg, ImplicitActuatorCfg

# 修改后
from isaaclab.actuators import DCMotorCfg, DelayedPDActuatorCfg, ImplicitActuatorCfg
```

##### 1.2 PIPER机械臂执行器升级
**修改前**:
```python
"arm": DCMotorCfg(
    joint_names_expr=["arm_joint1", "arm_joint2", "arm_joint3", "arm_joint4", "arm_joint5", "arm_joint6"],
    effort_limit=20.0,
    saturation_effort=20.0,
    velocity_limit=10.0,
    stiffness=25.0,  # 较硬
    damping=0.5,
    friction=0.0,
)
```

**修改后**:
```python
"arm": DelayedPDActuatorCfg(  # 带延迟的PD控制
    joint_names_expr=["arm_joint.*"],
    min_delay=2,             # 最小延迟2步
    max_delay=5,             # 最大延迟5步
    effort_limit_sim={
        "arm_joint1": 20.0,
        "arm_joint2": 20.0,
        "arm_joint3": 20.0,
        "arm_joint4": 10.0,  # 末端关节力矩较小
        "arm_joint5": 10.0,
        "arm_joint6": 10.0,
    },
    velocity_limit_sim={
        "arm_joint1": 10.0,
        "arm_joint2": 10.0,
        "arm_joint3": 10.0,
        "arm_joint4": 20.0,   # 末端关节速度更快
        "arm_joint5": 20.0,
        "arm_joint6": 20.0,
    },
    stiffness={              # 软刚度60%
        "arm_joint1": 10.0,
        "arm_joint2": 10.0,
        "arm_joint3": 10.0,
        "arm_joint4": 10.0,
        "arm_joint5": 10.0,
        "arm_joint6": 10.0,
    },
    damping={
        "arm_joint1": 0.5,
        "arm_joint2": 0.5,
        "arm_joint3": 0.5,
        "arm_joint4": 0.5,
        "arm_joint5": 0.5,
        "arm_joint6": 0.5,
    },
    friction=0.0,
)
```

##### 1.3 ARX5机械臂执行器升级
**修改前**:
```python
"arm": DCMotorCfg(
    joint_names_expr=["arm_joint1", "arm_joint2", "arm_joint3", "arm_joint4", "arm_joint5", "arm_joint6"],
    effort_limit=15.0,
    saturation_effort=15.0,
    velocity_limit=10.0,
    stiffness=25.0,  # 较硬
    damping=0.5,
    friction=0.0,
)
```

**修改后**:
```python
"arm": DelayedPDActuatorCfg(  # 带延迟的PD控制
    joint_names_expr=["arm_joint.*"],
    min_delay=5,             # 最小延迟5步
    max_delay=10,            # 最大延迟10步
    effort_limit_sim={
        "arm_joint1": 20.0,
        "arm_joint2": 20.0,
        "arm_joint3": 20.0,
        "arm_joint4": 10.0,  # 末端关节力矩较小
        "arm_joint5": 10.0,
        "arm_joint6": 10.0,
    },
    velocity_limit_sim={
        "arm_joint1": 20.0,
        "arm_joint2": 20.0,
        "arm_joint3": 20.0,
        "arm_joint4": 20.0,  # 末端关节速度更快
        "arm_joint5": 20.0,
        "arm_joint6": 20.0,
    },
    stiffness={              # 软刚度60%
        "arm_joint1": 10.0,
        "arm_joint2": 10.0,
        "arm_joint3": 10.0,
        "arm_joint4": 10.0,
        "arm_joint5": 10.0,
        "arm_joint6": 10.0,
    },
    damping={
        "arm_joint1": 0.5,
        "arm_joint2": 0.5,
        "arm_joint3": 0.5,
        "arm_joint4": 0.5,
        "arm_joint5": 0.5,
        "arm_joint6": 0.5,
    },
    friction=0.02,  # 添加小摩擦
)
```

##### 1.4 腿部执行器升级
**修改前**:
```python
"legs": DCMotorCfg(
    joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
    effort_limit=23.5,
    saturation_effort=23.5,
    velocity_limit=30.0,
    stiffness=25.0,  # 较硬
    damping=0.5,
    friction=0.0,
)
```

**修改后**:
```python
"legs": DelayedPDActuatorCfg(  # 带延迟的PD控制
    joint_names_expr=["^(?!.*_foot_joint).*"],  # 除轮子外的所有关节
    min_delay=2,             # 最小延迟2步
    max_delay=5,             # 最大延迟5步
    effort_limit_sim=23.5,
    velocity_limit_sim=30.0,
    stiffness=20.0,  # 降低刚度到20.0
    damping=0.5,
    friction=0.0,
)
```

### 2. 物理仿真配置升级 🔧

#### 修改文件: [`unitree.py`](source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py)

#### 主要修改内容：

##### 2.1 PIPER物理配置
**修改前**:
```python
articulation_props=sim_utils.ArticulationRootPropertiesCfg(
    enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
)
```

**修改后**:
```python
articulation_props=sim_utils.ArticulationRootPropertiesCfg(
    enabled_self_collisions=True,   # 启用自碰撞检测
    solver_position_iteration_count=4, solver_velocity_iteration_count=1  # 增加速度迭代
)
```

##### 2.2 ARX5物理配置
**修改前**:
```python
articulation_props=sim_utils.ArticulationRootPropertiesCfg(
    enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
)
```

**修改后**:
```python
articulation_props=sim_utils.ArticulationRootPropertiesCfg(
    enabled_self_collisions=True,   # 启用自碰撞检测
    solver_position_iteration_count=4, solver_velocity_iteration_count=1  # 增加速度迭代
)
```

### 3. 初始状态配置升级 🎯

#### 修改文件: [`unitree.py`](source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py)

#### 主要修改内容：

##### 3.1 PIPER初始状态
**修改前**:
```python
init_state=ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.4),  # 较低
    joint_pos={
        ".*L_hip_joint": 0.1,   # 髋关节倾斜
        ".*R_hip_joint": -0.1,  # 髋关节倾斜
        "F[L,R]_thigh_joint": 0.8,
        "R[L,R]_thigh_joint": 1.0,
        ".*_calf_joint": -1.5,
        ".*_foot_joint": 0.0,
        "arm_joint1": 0.0,  # 全部归零
        "arm_joint2": 0.0,
        "arm_joint3": 0.0,
        "arm_joint4": 0.0,
        "arm_joint5": 0.0,
        "arm_joint6": 0.0,
    },
)
```

**修改后**:
```python
init_state=ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.45),  # 提高初始高度5cm
    joint_pos={
        ".*L_hip_joint": 0.0,   # 髋关节不倾斜
        ".*R_hip_joint": 0.0,   # 髋关节不倾斜
        "F.*_thigh_joint": 0.8,
        "R.*_thigh_joint": 0.8,  # 后腿大腿统一为0.8
        ".*_calf_joint": -1.5,
        ".*_foot_joint": 0.0,
        "arm_joint1": 0.0,
        "arm_joint2": 2.0,          # 预设机械臂姿态
        "arm_joint3": -1.0,         # 预设机械臂姿态
        "arm_joint4": 0.0,
        "arm_joint5": -0.9,         # 预设机械臂姿态
        "arm_joint6": 0.0,
    },
)
```

##### 3.2 ARX5初始状态
**修改前**:
```python
init_state=ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.4),  # 较低
    joint_pos={
        ".*L_hip_joint": 0.1,   # 髋关节倾斜
        ".*R_hip_joint": -0.1,  # 髋关节倾斜
        "F[L,R]_thigh_joint": 0.8,
        "R[L,R]_thigh_joint": 1.0,
        ".*_calf_joint": -1.5,
        ".*_foot_joint": 0.0,
        "arm_joint1": 0.0,  # 全部归零
        "arm_joint2": 0.0,
        "arm_joint3": 0.0,
        "arm_joint4": 0.0,
        "arm_joint5": 0.0,
        "arm_joint6": 0.0,
    },
)
```

**修改后**:
```python
init_state=ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.45),  # 提高初始高度5cm
    joint_pos={
        ".*L_hip_joint": 0.0,   # 髋关节不倾斜
        ".*R_hip_joint": 0.0,   # 髋关节不倾斜
        "F.*_thigh_joint": 0.8,
        "R.*_thigh_joint": 0.8,  # 后腿大腿统一为0.8
        ".*_calf_joint": -1.5,
        ".*_foot_joint": 0.0,
        "arm_joint1": 0.0,
        "arm_joint2": 2.0,          # 预设机械臂姿态
        "arm_joint3": 1.0,
        "arm_joint4": 1.0,
        "arm_joint5": 0.0,
        "arm_joint6": 0.0,
    },
)
```

### 4. 动作配置升级 🎮

#### 修改文件: [`velocity_env_cfg.py`](source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2w_arm/velocity_env_cfg.py) 和 [`velocity_env_cfg_piper.py`](source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2w_arm/velocity_env_cfg_piper.py)

#### 主要修改内容：

**修改前**:
```python
self.actions.joint_pos.scale = {
    ".*_hip_joint": 0.125,
    "^(?!.*_hip_joint).*": 0.25,
    "arm_joint.*": 0.2,  # 较小的机械臂动作scale
}
```

**修改后**:
```python
self.actions.joint_pos.scale = {
    ".*_hip_joint": 0.125,
    "^(?!.*_hip_joint).*": 0.25,
    "arm_joint.*": 0.5,  # 增大机械臂动作scale150%
}
```

### 5. 奖励函数升级 🏆

#### 修改文件: [`velocity_env_cfg.py`](source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2w_arm/velocity_env_cfg.py) 和 [`velocity_env_cfg_piper.py`](source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2w_arm/velocity_env_cfg_piper.py)

#### 主要修改内容：

**修改前**:
```python
self.rewards.base_height_l2.weight = 0  # 不控制基座高度
```

**修改后**:
```python
self.rewards.base_height_l2.weight = -5.0  # 强制控制基座高度为0.4m
```

**效果说明**:
- **修改前**: 机器人可以在0.2m-0.6m高度范围内自由移动
- **修改后**: 强制机器人保持在0.4m高度，更稳定的移动性能

### 6. 代码清理 🧹

#### 修改文件: [`velocity_env_cfg_piper.py`](source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2w_arm/velocity_env_cfg_piper.py)

#### 主要修改内容：

**修改前**:
```python
import isaaclab.terrains as terrain_gen  # 未使用
```

**修改后**:
```python
# 移除未使用的导入，避免IDE警告
```

## 📊 参数对比总结

| 参数类别 | 修改前 (unitree_rl_lab) | 修改后 (robot_lab风格) | 变化幅度 | 说明 |
|---------|---------------------------|----------------------|---------|------|
| **执行器类型** | DCMotorCfg | DelayedPDActuatorCfg | 关键升级 | 添加延迟控制 |
| **机械臂刚度** | 25.0 | 10.0 | -60% | 更软，更柔性 |
| **机械臂动作scale** | 0.2 | 0.5 | +150% | 更大的动作空间 |
| **延迟控制** | 无 | 2-5/5-10步 | - | 模拟真实硬件延迟 |
| **自碰撞检测** | False | True | - | 更准确的碰撞检测 |
| **初始高度** | 0.4m | 0.45m | +12.5% | 更稳定的初始姿态 |
| **初始机械臂姿态** | 全部0.0 | 预设姿态 | - | 预设折叠姿态 |
| **高度控制权重** | 0.0 | -5.0 | - | 强制高度控制 |

## 🎯 策略升级效果

### 移动策略改进

1. **延迟控制**:
   - **效果**: 模拟真实硬件的通信和计算延迟
   - **用途**: 提高训练策略的真实性，减少过拟合到理想环境
   - **影响**: 动作响应更平滑，但收敛速度可能降低

2. **软刚度控制**:
   - **效果**: 降低机械臂关节的阻抗
   - **用途**: 提高机械臂的柔顺性，减少对抗干扰
   - **影响**: 机械臂更易受外部扰动，更安全

3. **大动作scale**:
   - **效果**: 扩大机械臂的动作空间
   - **用途**: 支持更大范围的运动控制
   - **影响**: 训练初期可能更不稳定，需要更多数据

4. **强制高度控制**:
   - **效果**: 严格限制机器人高度变化
   - **用途**: 保持稳定的高度，提高移动安全性
   - **影响**: 可能限制某些地形通过能力

### 训练预期变化

| 阶段 | 修改前 | 修改后 |
|------|--------|--------|
| **初期(0-1000 episodes)** | 快速收敛轮腿控制，机械臂保持0.0 | 收敛速度较慢，需要学习延迟和软刚度 |
| **中期(1000-5000 episodes)** | 稳定轮腿协调 | 学习机械臂的基本控制，机械臂保持预设姿态 |
| **后期(5000+ episodes)** | 精细调整 | 学习机械臂主动控制，平衡移动和操作 |

## 🚀 使用建议

### 训练脚本修改

**使用修改后的配置进行训练**:
```bash
# ARX5机械臂
python /home/jay/unitree_rl_lab/scripts/rsl_rl/train.py \
    --task Unitree-Go2WArm-Velocity-Flat-v0 \
    --headless

# Piper机械臂
python /home/jay/unitree_rl_lab/scripts/rsl_rl/train.py \
    --task Unitree-Go2WArm-Velocity-Flat-v0 \
    --headless
```

### 参数调整建议

如果发现训练不理想，可以逐步调整参数：

1. **收敛过慢**:
   - 减少机械臂动作scale: `0.5 → 0.3`
   - 减少延迟范围: `2-5步 → 1-3步`

2. **机械臂不稳定**:
   - 增加机械臂刚度: `10.0 → 15.0`
   - 增加阻尼: `0.5 → 1.0`

3. **移动性能下降**:
   - 降低高度控制权重: `-5.0 → -2.0`
   - 减少初始高度: `0.45m → 0.42m`

## 📝 配置文件清单

修改的配置文件:
1. [`source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py`](source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py)
2. [`source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2w_arm/velocity_env_cfg.py`](source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2w_arm/velocity_env_cfg.py)
3. [`source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2w_arm/velocity_env_cfg_piper.py`](source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2w_arm/velocity_env_cfg_piper.py)

## 🔍 验证和测试

运行验证脚本：
```bash
python /home/jay/unitree_rl_lab/test_robot_lab_modifications.py
```

## 🎓 参考文档

- [GO2W_ARM_README.md](GO2W_ARM_README.md) - 使用说明
- [GO2W_ARM_COMPARISON.md](GO2W_ARM_COMPARISON.md) - 详细对比
- [GO2W_ARM_QUICK_COMPARISON.md](GO2W_ARM_QUICK_COMPARISON.md) - 快速参考

## 总结

✅ **完成的核心迁移**:
1. 执行器类型: 理想DCMotor → 真实DelayedPD
2. 机械臂控制: 被动保持 → 主动控制
3. 训练目标: 纯移动 → 移动操作基础
4. 控制精度: 基础PID → 带延迟的PD控制
5. 物理仿真: 简化 → 更真实（自碰撞+迭代）

这些修改使GO2W-Arm项目从纯移动配置升级为支持移动操作的基础配置，同时保持了与GO2W相同的移动性能特征。
