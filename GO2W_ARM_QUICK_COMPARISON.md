# GO2W-Arm 配置快速对比

## ⚡ 核心差异一览表

| 配置类别 | unitree_rl_lab | robot_lab_locomanip | 关键差异 |
|---------|----------------|---------------------|---------|
| **训练目标** | 轮腿移动 (locomotion) | 移动操作 (loco-manipulation) | **任务类型完全不同** |
| **执行器类型** | DCMotorCfg | DelayedPDActuatorCfg | 参考项目有延迟控制 |
| **机械臂刚度** | 25.0 | 10.0 | 参考项目软60% |
| **机械臂动作scale** | 0.2 | 0.5 | 参考项目大2.5倍 |
| **初始高度** | 0.4m | 0.45m | 参考项目高5cm |
| **初始机械臂姿态** | 全部0.0 | 预设姿态 (ARX5: 2.0,1.0,1.0,1.0,0.0,0.0) | 参考项目有预设 |
| **传感器** | 接触力+高度扫描 | +IMU(EE)+相机 | 参考项目扩展传感器 |
| **末端观测** | ❌ 无 | ✅ 位置+质量 | **移动操作核心** |
| **末端奖励** | ❌ 无 | ✅ 位置(5.0)+姿态(2.5) | **移动操作核心** |
| **基座高度控制** | weight=0.0 | weight=-5.0 | 参考项目强制高度 |
| **自碰撞检测** | False | True | 参考项目更严格 |
| **地形配置** | 标准地形 | 混合粗糙道路 | 参考项目更复杂 |

## 🔧 关键技术参数对比

### 执行器配置 (机械臂)

| 参数 | unitree_rl_lab (DCMotor) | robot_lab_locomanip (DelayedPD) | 说明 |
|------|-------------------------|-------------------------------|------|
| 最大力矩 | 15.0/20.0 Nm | 10.0-20.0 Nm (分段) | 参考项目末端关节更小 |
| 最大速度 | 10.0 rad/s | 20.0 rad/s (j1-3), 20.0 rad/s (j4-6) | 参考项目速度更快 |
| 刚度 | 25.0 | 10.0 | **参考项目软60%** |
| 阻尼 | 0.5 | 0.5 | 相同 |
| 摩擦 | 0.0 | 0.02 | 参考项目有小摩擦 |
| 延迟 | ❌ 无 | 5-10步 | **参考项目模拟真实延迟** |

### 观测空间维度

| 观测组 | unitree_rl_lab | robot_lab_locomanip | 维度差异 |
|---------|----------------|---------------------|---------|
| 基础运动 | ~30维 | ~30维 | 相同 |
| **末端状态** | 0维 | **6-9维** | 参考项目多6-9维 |
| **力矩** | 0维 | **22维** | 参考项目多22维 |
| **质量** | 0维 | **2维** | 参考项目多2维 |
| 总维度 | ~30-40维 | **~50-70维** | 参考项目多25-30维 |

### 奖励权重对比 (机械臂相关)

| 奖励项 | unitree_rl_lab | robot_lab_locomanip | 差异说明 |
|--------|----------------|---------------------|---------|
| 末端位置跟踪 | ❌ 无 | 5.0 (指数) | **移动操作核心奖励** |
| 末端姿态跟踪 | ❌ 无 | 2.5 (指数) | **移动操作核心奖励** |
| 末端速度惩罚 | ❌ 无 | -2.0 (Z), -0.1 (XY) | 稳定性控制 |
| 末端姿态惩罚 | ❌ 无 | -2.0 (平坦) | 稳定性控制 |
| 基座高度控制 | 0.0 | -5.0 | 参考项目强制高度 |

## 📋 策略定义差异

### 动作空间

```python
# unitree_rl_lab
arm_scale = 0.2  # 小动作，机械臂微调

# robot_lab_locomanip
arm_scale = 0.5  # 大动作，机械臂主动控制
```

### 机械臂初始姿态

```python
# unitree_rl_lab - ARX5
"arm_joint1": 0.0, "arm_joint2": 0.0, "arm_joint3": 0.0,
"arm_joint4": 0.0, "arm_joint5": 0.0, "arm_joint6": 0.0

# robot_lab_locomanip - ARX5
"arm_joint1": 0.0, "arm_joint2": 2.0, "arm_joint3": 1.0,
"arm_joint4": 1.0, "arm_joint5": 0.0, "arm_joint6": 0.0
```

### 末端执行器控制

```python
# unitree_rl_lab: 无末端专用控制
# robot_lab_locomanip:
- 末端位置相对基座观测
- 末端线速度和角速度惩罚
- 末端平坦姿态惩罚
- 末端位置/姿态跟踪奖励
```

## 🎯 适用场景建议

### 使用 unitree_rl_lab 配置

✅ **适合场景**:
- 纯移动任务 (导航、巡逻)
- 机械臂仅作为载荷
- 快速原型验证
- 计算资源有限
- 需要快速收敛

❌ **不适合场景**:
- 需要主动机械臂控制
- 移动操作任务
- 需要末端精度控制

### 使用 robot_lab_locomanip 配置

✅ **适合场景**:
- 移动操作任务 (loco-manipulation)
- 抓取、放置操作
- 需要末端控制
- 需要移动过程中操作
- 需要更高精度

❌ **不适合场景**:
- 纯移动任务
- 计算资源极其有限
- 需要极快收敛速度

## 🔄 配置迁移建议

### 从unitree_rl_lab迁移到robot_lab风格

如果需要增加移动操作能力，建议按以下优先级迁移：

**优先级1 - 执行器升级**:
```python
# 使用DelayedPDActuator替代DCMotor
from isaaclab.actuators import DelayedPDActuatorCfg

actuators={
    "arm": DelayedPDActuatorCfg(
        joint_names_expr=["arm_joint.*"],
        min_delay=5,
        max_delay=10,
        effort_limit_sim={...},
        stiffness=10.0,  # 降低刚度
        damping=0.5,
    ),
}
```

**优先级2 - 增大动作scale**:
```python
self.actions.arm_pos.scale = 0.5  # 从0.2增加到0.5
```

**优先级3 - 添加末端观测**:
```python
# 添加末端状态观测
self.observations.policy.ee_pos_w = ObsTerm(
    func=mdp.ee_pos_rel_body,
    params={"asset_cfg": SceneEntityCfg("robot", body_names="arm_link6")}
)
```

**优先级4 - 添加末端奖励**:
```python
# 添加末端跟踪奖励
self.rewards.end_effector_position_tracking_exp = RewTerm(
    func=mdp.end_effector_position_tracking_exp,
    weight=5.0,
    params={"asset_cfg": SceneEntityCfg("robot", body_names="arm_link6")}
)
```

### 保持unitree_rl_lab风格的改进

如果只想保持当前设计哲学，建议：

**改进1 - 优化机械臂初始姿态**:
```python
# 使用安全的折叠姿态而非全部0.0
"arm_joint1": 0.0,  # 腰关节归零
"arm_joint2": 1.5,  # 肩关节部分展开
"arm_joint3": -0.5, # 肘关节部分收缩
"arm_joint4": 0.0,  # 腕关节归零
"arm_joint5": 0.0,  # 手腕关节归零
"arm_joint6": 0.0,  # 手腕关节归零
```

**改进2 - 调整机械臂执行器参数**:
```python
actuators={
    "arm": DCMotorCfg(
        joint_names_expr=["arm_joint.*"],
        effort_limit=20.0,
        velocity_limit=15.0,  # 稍微提高速度
        stiffness=20.0,  # 稍微降低刚度
        damping=0.5,
    ),
}
```

**改进3 - 添加轻量的末端稳定性奖励**:
```python
# 不主动控制末端，但保持其稳定
self.rewards.arm_stability = RewTerm(
    func=mdp.arm_stability,
    weight=0.5,  # 小权重，不干扰主要运动
    params={"asset_cfg": SceneEntityCfg("robot", joint_names="arm_joint.*")}
)
```

## 📊 性能和收敛预期

| 指标 | unitree_rl_lab | robot_lab_locomanip | 说明 |
|------|----------------|---------------------|------|
| **观测维度** | 较低 (~40) | 较高 (~60) | robot_lab需要更多数据 |
| **动作空间** | 较小 | 较大 | robot_lab探索更困难 |
| **收敛速度** | 快 | 较慢 | unitree_rl_lab更快 |
| **机械臂控制精度** | 低 | 高 | robot_lab精度更高 |
| **末端可达性** | 受限 | 精确 | robot_lab支持主动控制 |
| **移动性能** | 优秀 | 优秀 | 都使用相同的移动策略 |

## 🔍 代码实现差异示例

### 执行器配置对比

**unitree_rl_lab**:
```python
"arm": DCMotorCfg(
    joint_names_expr=["arm_joint1", "arm_joint2", "arm_joint3",
                   "arm_joint4", "arm_joint5", "arm_joint6"],
    effort_limit=20.0,
    saturation_effort=20.0,
    velocity_limit=10.0,
    stiffness=25.0,  # 较硬
    damping=0.5,
    friction=0.0,
)
```

**robot_lab_locomanip**:
```python
"arm": DelayedPDActuatorCfg(
    joint_names_expr=["arm_joint.*"],
    min_delay=5,  # 延迟控制
    max_delay=10,
    effort_limit_sim={
        "arm_joint1": 20.0, "arm_joint2": 20.0, "arm_joint3": 20.0,
        "arm_joint4": 10.0, "arm_joint5": 10.0, "arm_joint6": 10.0,
    },
    velocity_limit_sim={
        "arm_joint1": 20.0, "arm_joint2": 20.0, "arm_joint3": 20.0,
        "arm_joint4": 20.0, "arm_joint5": 20.0, "arm_joint6": 20.0,
    },
    stiffness=10.0,  # 较软
    damping=0.5,
    friction=0.02,
)
```

### 奖励函数对比

**unitree_rl_lab** (纯移动):
```python
# 无机械臂专用奖励，仅基座移动奖励
self.rewards.track_lin_vel_xy_exp.weight = 3.0
self.rewards.track_ang_vel_z_exp.weight = 1.5
```

**robot_lab_locomanip** (移动操作):
```python
# 基座移动 + 末端控制奖励
self.rewards.track_lin_vel_xy_exp.weight = 3.0
self.rewards.track_ang_vel_z_exp.weight = 1.5

# 机械臂控制奖励
self.rewards.end_effector_position_tracking_exp.weight = 5.0
self.rewards.end_effector_orientation_tracking_exp.weight = 2.5

# 机械臂稳定性惩罚
self.rewards.ee_lin_vel_z_l2.weight = -2.0
self.rewards.ee_ang_vel_xy_l2.weight = -0.1
```

## 📝 推荐配置方案

### 方案A: 保持unitree_rl_lab风格 (推荐)

**配置**: 当前的unitree_rl_lab配置
**适用**: 纯移动任务
**改进**: 微调机械臂初始姿态，优化执行器参数
**优势**: 快速收敛，简单调试
**劣势**: 无法主动控制机械臂

### 方案B: 采用robot_lab_locomanip风格 (高级)

**配置**: 完全迁移robot_lab_locomanip的移动操作策略
**适用**: 移动操作任务
**改进**: 需要完整迁移末端控制、奖励、观测
**优势**: 完整的移动操作能力
**劣势**: 训练复杂度高，收敛慢

### 方案C: 混合方案 (推荐进阶)

**配置**: 基于unitree_rl_lab，选择性添加robot_lab特性
**适用**: 需要一定机械臂控制能力的移动任务
**改进**:
1. 添加末端观测（位置、质量）
2. 添加轻量末端稳定性奖励
3. 优化执行器参数（延迟控制、调整刚度）
4. 保持简单的纯移动为主，末端为辅

**优势**: 平衡收敛速度和控制能力
**劣势**: 需要细致调参

## 🎓 学习曲线预期

### unitree_rl_lab配置
```
Episode 0-1000:   快速学习基础移动
Episode 1000-5000: 稳定轮腿协调
Episode 5000+:     收敛，机械臂保持初始姿态
```

### robot_lab_locomanip配置
```
Episode 0-1000:   学习基础移动 + 机械臂基本控制
Episode 1000-5000: 学习移动操作协调
Episode 5000-10000: 精细调整末端控制
Episode 10000+:    收敛，支持主动末端操作
```

## 总结

| 维度 | unitree_rl_lab | robot_lab_locomanip | 推荐选择 |
|------|----------------|---------------------|---------|
| **复杂度** | 低 | 高 | 任务导向 |
| **收敛速度** | 快 | 慢 | 根据需求 |
| **控制精度** | 移动优秀 | 移动+末端优秀 | 根据需求 |
| **适用任务** | 纯移动 | 移动操作 | 根据需求 |
| **调试难度** | 简单 | 复杂 | 根据经验 |
| **计算需求** | 低 | 中高 | 根据硬件 |

**最终建议**:
1. 如果任务主要是移动 → 保持unitree_rl_lab配置
2. 如果需要移动操作 → 迁移到robot_lab_locomanip风格
3. 如果不确定 → 从unitree_rl_lab开始，逐步添加robot_lab特性
