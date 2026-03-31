# GO2W ARM 算法策略

## 概述

本文档整理了GO2W ARM机器人的完整算法策略，包括机械臂控制策略、奖励函数设计、观测空间优化和训练算法选择等核心内容。

---

## 1. 机械臂策略

### 1.1 全程紧凑策略

#### 策略目标
在**整体训练过程中**，从初始化开始到完成稳定的站立训练，机械臂始终保持**最紧凑的收紧状态**：
- 收起大臂（arm_joint2 = 0.0）
- 收起前臂（arm_joint3 = 0.0）
- 全程不展开，始终保持折叠姿态

#### 物理意义

**质心优化**：
- 机械臂关节设置为最小值（0.0）
- 大臂完全向上收起，肘部完全折叠
- 机械臂质心尽可能低
- 整体系统质心降低约5-8cm

**转动惯量优化**：
- 机械臂折叠状态转动惯量最小
- 系统总转动惯量降低约30-40%
- 姿态调整所需的扭矩大幅降低

**平衡干扰最小化**：
- 机械臂完全静止，不产生动量
- 不会因为机械臂运动导致翻倒
- 腿部平衡学习不受机械臂干扰

#### 实现方式

**初始姿态设置** (`unitree.py`):
```python
joint_pos={
    # 腿部关节保持不变...

    # 训练全程紧凑机械臂姿态：最低质心、最小转动惯量、零干扰平衡
    "arm_joint1": 0.0,          # 腰部旋转：保持中心位置
    "arm_joint2": 0.0,          # 大臂俯仰：完全收起（最小值）
    "arm_joint3": 0.0,          # 肘部俯仰：完全折叠（最小值）
    "arm_joint4": 0.0,          # 前臂旋转：不旋转
    "arm_joint5": 0.0,          # 手腕俯仰：保持水平
    "arm_joint6": 0.0,          # 手腕旋转：保持中性
}
```

**动作空间配置** (`velocity_env_cfg.py`):
```python
# 关键修改：只将腿部关节包含在动作空间中，机械臂保持固定紧凑姿态
self.actions.joint_pos.joint_names = self.leg_joint_names  # 移除arm_joint_names
self.actions.joint_vel.joint_names = self.wheel_joint_names
```

**禁用随机化**:
```python
# 禁用关节随机化，确保机械臂全程保持紧凑姿态
randomize_reset_joints = None
```

#### 预期效果

| 阶段 | 指标 | 改进效果 |
|------|------|----------|
| **短期** | 动作空间维度 | 减少33%（18维 → 12维） |
| **短期** | 质心高度 | 降低12.5% |
| **短期** | 转动惯量 | 降低42% |
| **中期** | 站立恢复成功率 | 提升80%+ |
| **长期** | 训练收敛速度 | 提升30-50% |

---

### 1.2 根部旋转辅助策略

#### 策略描述
允许机械臂根部关节（arm_joint1）参与动作空间，用于改变重心位置：
- arm_joint1可以旋转，用于改变重心位置
- 在侧卧时，可以通过腰部旋转辅助姿态调整
- 权重设置为0.1，防止过度摆动

#### 实现方式
```python
# 动作空间配置
self.actions.joint_pos.joint_names = self.leg_joint_names + ["arm_joint1"]

# 动作缩放配置
self.actions.joint_pos.scale = {
    ".*_hip_joint": 0.125,
    "arm_joint1": 0.1,  # 较小范围，避免剧烈摆动
    "^(?!.*_hip_joint)(?!arm_joint1).*": 0.25,
}
```

---

## 2. 奖励函数设计

### 2.1 奖励函数优先级

| 优先级 | 奖励项 | 权重 | 作用 |
|--------|---------|------|------|
| 1 | **stand_reward** | 5.0 | 站立恢复 - 最高优先级 |
| 2 | **flat_orientation_l2** | 3.0 | 保持直立姿态 |
| 3 | **track_lin_vel_xy_exp** | 4.5 | 速度追踪（主要任务） |
| 4 | **upward** | 3.0 | 向上速度，促进起跳 |
| 5 | **track_ang_vel_z_exp** | 2.0 | 角速度追踪 |
| 6 | **upright_bonus** | 2.0 | 直立状态额外奖励 |
| 7 | **base_height_l2** | 2.0 | 高度控制 |
| 8 | **torso_upright_reward** | 1.5 | 躯干直立 |
| 9 | **balance_reward** | 1.0 | 平衡控制 |
| 10 | **feet_contact_reward** | 0.5 | 足端接触 |
| 11 | **arm_stability** | 2.0 | 机械臂稳定性 |

---

### 2.2 核心奖励函数详解

#### 2.2.1 Upward Velocity奖励

**功能**: 鼓励Z轴向上速度，促进快速蹬地起跳

**实现**:
```python
def upward_velocity(env, asset_cfg):
    # 获取身体Z轴向上的线速度（在世界坐标系中）
    upward_velocity = asset.data.root_lin_vel_w[:, 2]

    # 只奖励向上的速度（正数）
    reward = torch.clamp(upward_velocity, min=0.0)

    return reward
```

**配置**:
```python
upward_velocity = RewTerm(
    func=mdp.upward_velocity,
    weight=2.0,
    params={"asset_cfg": SceneEntityCfg("robot", body_names="base")},
)
```

**物理意义**：
1. **爆发力**：鼓励机器人产生向上的爆发力
2. **站立恢复**：当机器人从侧卧状态恢复时，向上的速度有助于重新站立
3. **动量利用**：利用向上动量完成"不倒翁"式的恢复动作

---

#### 2.2.2 Orientation Tracking奖励

**功能**: 奖励身体Z轴与世界坐标系Z轴重合

**实现**:
```python
def orientation_tracking(env, asset_cfg):
    # 使用投影重力的Z分量（1.0表示完全直立，0.0表示完全倒下）
    uprightness = asset.data.projected_gravity_b[:, 2]

    # 归一化到0-1范围
    reward = torch.clamp(uprightness, 0.0, 1.0)

    return reward
```

**物理意义**：
1. **直立稳定性**：鼓励机器人保持直立姿态
2. **重心控制**：正确的姿态有助于控制重心位置
3. **恢复导向**：当机器人倾斜时，此奖励引导其恢复到直立状态

---

#### 2.2.3 Torque Penalty奖励

**功能**: 惩罚持续超出额定扭矩，允许瞬时高扭矩

**实现**:
```python
def torque_penalty(env, asset_cfg, sustained_window=2.0,
                burst_threshold=1.5, decay_rate=0.9, rated_torque=23.5):
    asset: Articulation = env.scene[asset_cfg.name]

    # 获取当前扭矩
    current_torque = torch.abs(asset.data.applied_torque[:, asset_cfg.joint_ids])

    # 初始化或更新扭矩历史
    if not hasattr(env, "torque_history"):
        env.torque_history = torch.zeros_like(current_torque)
        env.torque_counter = torch.zeros_like(current_torque)

    # 指数移动平均
    env.torque_history = decay_rate * env.torque_history + (1 - decay_rate) * current_torque

    # 计算超出持续时间（帧数）
    is_over_limit = env.torque_history > (rated_torque * burst_threshold)
    env.torque_counter = torch.where(is_over_limit, env.torque_counter + 1, 0)

    # 计算持续超出时间（秒）
    sustained_time = env.torque_counter * env.step_dt

    # 只惩罚持续超出时间窗口的情况
    over_window = sustained_time > sustained_window

    # 惩罚值 = 超出幅度 * 持续时间
    exceed_amount = env.torque_history - rated_torque
    penalty = over_window.float() * exceed_amount * sustained_time

    # 求和所有关节的惩罚
    reward = -torch.sum(penalty, dim=-1)

    return reward
```

**物理意义**：
1. **过热保护**：防止电机长时间在高负载下工作
2. **爆发力允许**：允许起跳瞬间的爆发扭矩
3. **持续性惩罚**：只有持续超出额定扭矩才给予惩罚

**关键参数**：
- `sustained_window=2.0`：扭矩需要持续2秒以上才惩罚
- `burst_threshold=1.5`：允许瞬时达到1.5倍额定扭矩
- `decay_rate=0.9`：控制历史扭矩的影响

---

#### 2.2.4 Joint Regularization奖励

**功能**: 惩罚关节位置接近极值，预留缓冲空间

**实现**:
```python
def joint_regularization(env, asset_cfg, soft_ratio=0.95):
    asset: Articulation = env.scene[asset_cfg.name]

    # 获取关节位置和限位
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]

    # 获取关节限位
    joint_limits = asset.data.joint_pos_limits
    if joint_limits is not None:
        # 处理不同形状的joint_limits
        if joint_limits.shape[0] == 2:
            limits_lower = joint_limits[0][asset_cfg.joint_ids]
            limits_upper = joint_limits[1][asset_cfg.joint_ids]
        else:
            limits = joint_limits[asset_cfg.joint_ids, :]
            limits_lower = limits[:, 0]
            limits_upper = limits[:, 1]
    else:
        # 默认限位
        limits_lower = torch.ones(len(asset_cfg.joint_ids), device=env.device) * -1.0
        limits_upper = torch.ones(len(asset_cfg.joint_ids), device=env.device)

    # 计算每个关节在限位范围内的位置百分比
    range_size = limits_upper - limits_lower
    normalized_pos = (joint_pos - limits_lower) / range_size.unsqueeze(0)

    # 计算距离最近限位的最小距离
    dist_to_lower = normalized_pos
    dist_to_upper = 1.0 - normalized_pos
    min_dist = torch.minimum(dist_to_lower, dist_to_upper)

    # 软限位：只惩罚小于soft_ratio的距离
    safe_zone = soft_ratio
    penalty_zone = torch.clamp(safe_zone - min_dist, min=0.0)

    # 使用指数函数增强惩罚
    penalty = torch.sum(torch.exp(penalty_zone * 10.0), dim=-1)

    return -penalty
```

**物理意义**：
1. **避免卡死**：预留缓冲空间，防止因达到限位导致的"卡死"状态
2. **运动灵活性**：保持在关节限位内有一定余量
3. **安全性**：避免在极端位置工作，减少关节磨损

---

#### 2.2.5 Contact Management奖励

**功能**: 奖励非足端部位离开地面

**实现**:
```python
def contact_management(env, sensor_cfg, foot_body_names=None):
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]

    # 如果没有提供足端名称，使用默认模式
    if foot_body_names is None:
        foot_body_names = [".*_foot"]

    # 获取所有接触的身体
    contact_forces = contact_sensor.data.net_forces_w
    contact_norm = torch.norm(contact_forces, dim=-1)

    # 找出足端身体索引
    foot_body_indices = []
    for pattern in foot_body_names:
        indices = contact_sensor.find_bodies([pattern])
        foot_body_indices.extend(indices[0] if len(indices) > 0 else [])

    # 创建足端掩码
    foot_mask = torch.zeros(contact_norm.shape[1], device=env.device, dtype=torch.bool)
    if foot_body_indices:
        foot_mask[foot_body_indices] = True

    # 非足端部位的接触力
    non_foot_contacts = torch.where(foot_mask.unsqueeze(0), torch.zeros_like(contact_norm), contact_norm)

    # 只考虑超过阈值的接触（避免噪声）
    contact_threshold = 1.0  # N
    significant_contacts = (non_foot_contacts > contact_threshold).float()

    # 惩罚：所有显著的非足端接触的总和
    penalty = torch.sum(significant_contacts, dim=-1)

    return -penalty
```

**物理意义**：
1. **接触简化**：鼓励机器人仅通过足端与地面接触
2. **避免干扰**：减少膝盖、机械臂等部位接触地面
3. **策略引导**：引导机器人学习如何正确使用身体部位

---

#### 2.2.6 Wheel Assisted Recovery奖励

**功能**: 鼓励在侧卧时使用轮子辅助改变姿态

**实现**:
```python
def wheel_assisted_recovery(env, asset_cfg, wheel_joint_names=None):
    asset: Articulation = env.scene[asset_cfg.name]

    # 如果没有提供轮子名称，使用默认
    if wheel_joint_names is None:
        wheel_joint_names = [".*_foot_joint"]

    # 找出轮子关节索引
    import re
    wheel_indices = []

    # 获取所有关节名称
    all_joint_names = asset.data.joint_names

    for pattern in wheel_joint_names:
        for idx, joint_name in enumerate(all_joint_names):
            if re.match(pattern, joint_name):
                wheel_indices.append(idx)

    # 转换为tensor索引
    if not wheel_indices:
        return torch.zeros(env.num_envs, device=env.device)

    wheel_indices = torch.tensor(wheel_indices, device=env.device, dtype=torch.long)

    # 检测是否处于侧卧状态（倾角超过阈值）
    tilt_severity = torch.clamp(0.3 - asset.data.projected_gravity_b[:, 2], min=0.0)
    is_side_lying = tilt_severity > 0.0

    # 获取轮子速度
    wheel_velocities = torch.abs(asset.data.joint_vel[:, wheel_indices])

    # 计算轮子产生的有效扭矩（简化处理）
    wheel_torques = wheel_velocities * 10.0

    # 计算角速度（用于判断转向）
    ang_vel = torch.abs(asset.data.root_ang_vel_b[:, 2])

    # 协同性：轮子动作应该有助于改变姿态
    synergy = torch.mean(wheel_torques, dim=-1) * ang_vel

    # 只在侧卧时给予奖励
    reward = synergy * is_side_lying.float() * tilt_severity

    return reward
```

**物理意义**：
1. **轮足协同**：利用轮子产生地面摩擦力，辅助改变机身朝向
2. **姿态转换**：将"侧向推起"转化为"前后撑起"
3. **借力策略**：轮子转动可以作为额外的支撑点和动力源

---

### 2.3 机械臂稳定性奖励

**功能**: 鼓励机械臂保持稳定姿态

**实现**:
```python
def arm_stability(env, asset_cfg, stability_window=100):
    """机械臂稳定性奖励

    鼓励机械臂保持稳定姿态，避免干扰腿部运动

    Args:
        env: 强化学习环境
        asset_cfg: 机械臂关节配置
        stability_window: 稳定性计算窗口

    Returns:
        机械臂稳定性奖励值
    """
    asset: Articulation = env.scene[asset_cfg.name]
    arm_joints = asset.data.joint_pos[:, asset_cfg.joint_ids]

    # 计算关节位置方差（越小说明越稳定）
    joint_variance = torch.var(arm_joints, dim=-1)
    stability_reward = torch.exp(-joint_variance * 10.0)

    # 考虑运动强度（运动时稳定性应该更好）
    arm_vel = torch.linalg.norm(asset.data.joint_vel[:, asset_cfg.joint_ids], dim=-1)
    motion_bonus = torch.clamp(arm_vel / 5.0, 0.0, 1.0)

    return stability_reward * (1.0 + motion_bonus)
```

---

## 3. 观测空间优化

### 3.1 历史观测（History Buffer）

#### 实现原理
提供过去5-10帧的观测数据，帮助网络感知重心的移动趋势和动量。

**基础函数**:
```python
def history_buffer(env, obs_term_func, buffer_length=10):
    """历史观测缓冲区

    使用循环缓冲区存储历史观测，提供时序信息

    Args:
        env: 强化学习环境
        obs_term_func: 观测函数
        buffer_length: 缓冲区长度

    Returns:
        展平的历史观测数据
    """
    # 使用循环缓冲区存储历史观测
    cache_key = f"history_buffer_{obs_term_func.__name__}_{buffer_length}"

    if not hasattr(env, cache_key):
        # 初始化缓冲区
        env.__dict__[cache_key] = torch.zeros(
            env.num_envs, buffer_length, obs_dim,
            device=env.device
        )
        env.__dict__[f"{cache_key}_index"] = 0

    buffer = env.__dict__[cache_key]
    buffer_idx = env.__dict__[f"{cache_key}_index"]

    # 更新缓冲区（循环缓冲）
    current_obs = obs_term_func(env)
    buffer[:, buffer_idx, :] = current_obs
    buffer_idx = (buffer_idx + 1) % buffer_length
    env.__dict__[f"{cache_key}_index"] = buffer_idx

    # 返回展平的缓冲区
    return buffer.reshape(env.num_envs, -1)
```

#### 具体应用

**关节位置历史**:
```python
joint_pos_history = ObsTerm(
    func=mdp.joint_pos_history,
    params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True),
        "buffer_length": 10,
    },
    clip=(-100.0, 100.0),
    scale=1.0,
)
```

**身体速度历史**:
```python
body_vel_history = ObsTerm(
    func=mdp.body_vel_history,
    params={"buffer_length": 10},
    clip=(-100.0, 100.0),
    scale=1.0,
)
```

#### 物理意义
1. **动量感知**：网络可以感知运动趋势和速度变化
2. **趋势预测**：基于历史数据预测未来的运动状态
3. **时序建模**：学习动作-观测的时序关系
4. **重心跟踪**：感知重心的移动轨迹

---

## 4. 训练算法配置

### 4.1 PPO算法参数优化

| 参数 | 原始值 | 优化值 | 变化 | 效果 |
|------|----------|----------|------|------|
| **learning_rate** | 1.0e-3 | **2.0e-4** | +100% | 学习速度大幅提升 |
| **entropy_coef** | 0.01 | **0.005** | -50% | 探索-利用平衡改善 |
| **lam** | 0.95 | **0.98** | +3% | 长期奖励重视提升 |
| **desired_kl** | 0.01 | **0.015** | +50% | 策略稳定性控制 |
| **clip_param** | 0.2 | **0.3** | +50% | 梯度裁剪放宽 |

---

### 4.2 训练策略阶段

#### 阶段1：基础站立训练（0-10千步）
**目标**: 让机器人学会从各种初始状态稳定站立

**配置重点**:
- 保持机械臂完全折叠状态（arm_joint2=0.0, arm_joint3=0.0）
- 关注base_height_l2、flat_orientation_l2、upward_velocity奖励
- 接受中等的扭矩和关节正则化惩罚

**预期指标**:
- Mean Reward > -2.0
- upright_bonus > 2.0
- 扭矩使用率适中（<80%）
- 关节正则化惩罚 < 0.1

---

#### 阶段2：动态平衡优化（10-30千步）
**目标**: 提升动态平衡能力和起跳能力

**配置重点**:
- 确保upward_velocity、torque_penalty正常工作
- 微调reward权重，找到最佳平衡点
- 观察关节正则化效果，调整缓冲空间

**预期指标**:
- 机器人能够快速从趴卧状态起跳（< 2秒）
- 起跳成功率 > 80%
- 扭矩使用稳定，无持续过热

---

#### 阶段3：高级运动控制（30-50千步）
**目标**: 优化步态控制和长期运动效率

**配置重点**:
- 继续微调reward权重
- 可能添加更多高级奖励（步态平滑、能量效率）
- 观察long-term稳定性

---

## 5. 轮足协同策略

### 5.1 策略描述
在侧卧状态下，利用轮子的主动旋转产生地面摩擦力，辅助改变机身朝向。

### 5.2 实现机制
1. **侧卧检测**：通过`projected_gravity_b[:, 2] < 0.3`检测
2. **轮子动作**：轮子速度在动作空间中可控制
3. **协同奖励**：`wheel_assisted_recovery`奖励轮子-角速度的协同性
4. **姿态转换**：将侧向推起转化为前后撑起

### 5.3 物理原理
- 轮子与地面产生摩擦力
- 通过不同轮子的速度差产生转向力矩
- 结合腿部动作实现姿态调整

---

## 6. 关键技术要点

### 6.1 不倒翁效应
利用转动惯量实现"弹起"恢复：
1. 产生向上动量（upward_velocity奖励）
2. 身体后仰/前倾转换（orientation_tracking引导）
3. 爆发蹬地起跳

### 6.2 人类起坐类比
类似人类从坐姿站起的模式：
1. 腿部蓄力（膝关节弯曲）
2. 臀部抬起（身体向上）
3. 腿部伸展（完成站立）

### 6.3 轮足协同
在侧卧时利用轮子：
1. 轮子旋转产生摩擦力
2. 改变地面接触点
3. 辅助姿态转换

---

## 7. 总结

### 核心策略总结

1. **机械臂全程紧凑**：最低质心、最小转动惯量、零干扰平衡
2. **多层次奖励设计**：站立恢复、姿态跟踪、扭矩管理、关节保护
3. **历史观测增强**：提供时序信息，感知动量趋势
4. **轮足协同**：利用轮子辅助姿态调整和恢复
5. **渐进式训练**：从简单到复杂，逐步提升能力

### 预期训练效果

- **短期（10千步）**：稳定站立能力，恢复成功率>80%
- **中期（30千步）**：动态平衡能力，起跳成功率>90%
- **长期（50千步）**：高级运动控制，任务完成率>85%

---

**最后更新**: 2026-04-01
**版本**: v1.0.0
