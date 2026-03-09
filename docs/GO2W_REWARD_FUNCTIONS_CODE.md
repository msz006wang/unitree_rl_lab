# GO2W奖励函数代码实现详解

## 目录
1. [速度跟踪奖励](#速度跟踪奖励)
2. [姿态稳定性奖励](#姿态稳定性奖励)
3. [关节惩罚项](#关节惩罚项)
4. [动作平滑性奖励](#动作平滑性奖励)
5. [接触相关奖励](#接触相关奖励)
6. [扩展奖励函数](#扩展奖励函数)

---

## 速度跟踪奖励

### 1. 线速度跟踪 (track_lin_vel_xy_exp)

**源码位置**: `IsaacLab/isaaclab/envs/mdp/rewards.py`

```python
def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """使用指数核奖励跟踪XY平面线速度命令"""
    # 获取机器人对象
    asset: RigidObject = env.scene[asset_cfg.name]

    # 计算速度误差
    # command: [v_x_cmd, v_y_cmd, ω_z_cmd]
    # root_lin_vel_b: [v_x, v_y, v_z] (机体坐标系)
    lin_vel_error = torch.sum(
        torch.square(
            env.command_manager.get_command(command_name)[:, :2]  # 取前两维 [v_x_cmd, v_y_cmd]
            - asset.data.root_lin_vel_b[:, :2]                   # 取前两维 [v_x, v_y]
        ),
        dim=1,  # 沿特征维度求和: v_x_error² + v_y_error²
    )

    # 指数核: exp(-error² / std²)
    return torch.exp(-lin_vel_error / std**2)
```

**数学推导**:

```
# 误差计算
error = ||v_cmd - v_actual||²
       = (v_x_cmd - v_x)² + (v_y_cmd - v_y)²

# 指数核
reward = exp(-error / std²)

# std = 0.5 的效果
error = 0.0  → reward = 1.0     (完美跟踪)
error = 0.25 → reward = 0.607   (一个标准差)
error = 1.0  → reward = 0.135   (两个标准差)
error = 2.25 → reward = 0.018   (三个标准差)
```

**物理意义**:

1. **为什么用平方误差?**
   - 放大大误差
   - 惩罚偏离目标的行为
   - 平滑的梯度

2. **为什么用指数核?**
   - 输出范围 [0, 1]
   - 0误差 → reward=1 (最大奖励)
   - 大误差 → reward→0 (最小奖励)
   - 比线性奖励更鲁棒

3. **为什么std=0.5?**
   - 标准差控制奖励衰减速度
   - std越大，奖励曲线越平缓
   - 0.5是经验值，平衡严格性和容错性

**配置**:
```python
# velocity_env_cfg.py
track_lin_vel_xy_exp = RewTerm(
    func=mdp.track_lin_vel_xy_exp,
    weight=3.0,                          # 最高权重
    params={
        "command_name": "base_velocity",
        "std": math.sqrt(0.25)           # std = 0.5
    }
)
```

### 2. 角速度跟踪 (track_ang_vel_z_exp)

```python
def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """使用指数核奖励跟踪Z轴角速度命令"""
    asset: RigidObject = env.scene[asset_cfg.name]

    # 计算角速度误差
    # command: [v_x_cmd, v_y_cmd, ω_z_cmd]
    # root_ang_vel_b: [ω_x, ω_y, ω_z] (机体坐标系)
    ang_vel_error = torch.square(
        env.command_manager.get_command(command_name)[:, 2]  # 取第三维 ω_z_cmd
        - asset.data.root_ang_vel_b[:, 2]                    # 取第三维 ω_z
    )

    return torch.exp(-ang_vel_error / std**2)
```

**数学推导**:

```
# 误差计算 (注意这里没有求和，因为是标量)
error = (ω_z_cmd - ω_z)²

# 指数核
reward = exp(-error / std²)
```

**与线速度的区别**:
- 线速度: 2维 → 求和
- 角速度: 1维 → 单个平方

**为什么权重是1.5 (vs 3.0)?**
- 转向比直线移动次要
- 转向频率较低
- 避免过度惩罚转向

---

## 姿态稳定性奖励

### 3. Z轴线速度惩罚 (lin_vel_z_l2)

```python
def lin_vel_z_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """使用L2平方核惩罚Z轴线速度"""
    asset: RigidObject = env.scene[asset_cfg.name]

    # v_z: 垂直速度 (向上为正)
    return torch.square(asset.data.root_lin_vel_b[:, 2])
```

**物理意义**:

```
reward = -v_z²

v_z = 0.0  → reward = 0.0    (理想: 无垂直运动)
v_z = 0.5  → reward = -0.25  (轻微跳动)
v_z = 1.0  → reward = -1.0   (明显跳动)
v_z = 2.0  → reward = -4.0   (严重跳动)
```

**为什么惩罚Z轴速度?**
- 轮腿机器人应该平稳移动
- 跳动浪费能量
- 可能导致失稳
- 不符合轮式移动特性

**为什么权重-2.0?**
- 强惩罚，必须避免
- 比其他惩罚更强
- 确保平稳移动

### 4. XY轴角速度惩罚 (ang_vel_xy_l2)

```python
def ang_vel_xy_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """使用L2平方核惩罚XY轴角速度"""
    asset: RigidObject = env.scene[asset_cfg.name]

    # ω_x: 俯仰角速度
    # ω_y: 翻滚角速度
    return torch.sum(
        torch.square(asset.data.root_ang_vel_b[:, :2]),  # 取前两维 [ω_x, ω_y]
        dim=1
    )
```

**物理意义**:

```
reward = -(ω_x² + ω_y²)

# 完全水平
ω_x = 0, ω_y = 0  → reward = 0.0

# 前后摇晃
ω_x = 0.5, ω_y = 0  → reward = -0.25

# 左右摇晃
ω_x = 0, ω_y = 0.5  → reward = -0.25

# 复合摇晃
ω_x = 0.5, ω_y = 0.5 → reward = -0.5
```

**为什么权重很小 (-0.05)?**
- 允许小幅摇晃 (自然步态)
- 主要是防止过度摇晃
- 不像Z轴速度那么严格

### 5. 保持直立奖励 (upward)

```python
def upward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """奖励保持直立姿态"""
    asset: RigidObject = env.scene[asset_cfg.name]

    # projected_gravity_b: 重力在机体坐标系的投影
    # 直立时: [0, 0, -1] 或 [0, 0, 1] (取决于坐标系)
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward
```

**物理意义**:

```
# 假设g_z为重力向量Z分量
reward = (1 - g_z)²

# 完全直立
g_z = 1.0  → reward = 0.0

# 轻微倾斜
g_z = 0.9  → reward = 0.01

# 中度倾斜
g_z = 0.5  → reward = 0.25

# 严重倾斜
g_z = 0.0  → reward = 1.0

# 倒下
g_z = -1.0 → reward = 4.0
```

**为什么是正奖励 (+1.0)?**
- 鼓励保持直立
- 倾斜越大，奖励越高
- 与lin_vel_z_l2配合使用

**为什么用平方?**
- 小倾斜时惩罚小
- 大倾斜时惩罚大
- 非线性增强

---

## 关节惩罚项

### 6. 关节力矩惩罚 (joint_torques_l2)

```python
def joint_torques_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """惩罚关节力矩"""
    asset: Articulation = env.scene[asset_cfg.name]

    # applied_torque: 电机施加的力矩
    return torch.sum(
        torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]),
        dim=1
    )
```

**物理意义**:

```
reward = -Σ τ²

# 单关节示例
τ = 0 Nm    → reward = 0.0    (无力矩)
τ = 10 Nm   → reward = -100   (中等力矩)
τ = 50 Nm   → reward = -2500  (大力矩)
```

**为什么权重极小 (-2.5e-5)?**
- 力矩数量级大 (10-100 Nm)
- 避免主导其他奖励
- 鼓励能效，不牺牲性能

**为什么只惩罚腿部?**
```python
# 配置中
joint_torques_l2:
    joint_names = leg_joint_names  # 只包括12个腿部关节
    # 不包括4个轮子
```

**物理原因**:
- 腿部需要精确控制
- 轮子需要大扭矩移动
- 轮子的力矩是必要的

### 7. 关节加速度惩罚 (joint_acc_l2)

```python
def joint_acc_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """惩罚关节加速度"""
    asset: Articulation = env.scene[asset_cfg.name]

    # 计算加速度: α = Δv / Δt
    # previous_joint_vel: 上一步的关节速度
    joint_acc = (
        asset.data.joint_vel[:, asset_cfg.joint_ids]
        - asset.data.previous_joint_vel[:, asset_cfg.joint_ids]
    ) / env.step_dt

    return torch.sum(
        torch.square(joint_acc),
        dim=1
    )
```

**物理意义**:

```
# 假设 dt = 0.02s (50Hz)
v_current = 1.0 rad/s
v_prev = 0.5 rad/s
α = (1.0 - 0.5) / 0.02 = 25 rad/s²

reward = -α² = -625
```

**为什么惩罚加速度?**
- 加速度大 → 力矩大 (τ = Iα)
- 力矩大 → 能耗高
- 快速变化 → 机械磨损
- 平滑运动更自然

**为什么权重极小 (-2.5e-7)?**
- 加速度数量级很大
- 需要极小权重
- 防止主导训练

### 8. 关节位置限制惩罚 (joint_pos_limits)

```python
def joint_pos_limits(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"
) -> torch.Tensor:
    """惩罚关节位置超出限制"""
    asset: Articulation = env.scene[asset_cfg.name]

    # 获取关节位置限制
    # pos_limits: [[q_min_0, q_max_0], [q_min_1, q_max_1], ...]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    joint_pos_limits = asset.data.joint_limits[:, asset_cfg.joint_ids]

    # 计算超出限制的程度
    # 超出上界: max(0, q - q_max)²
    # 超出下界: max(0, q_min - q)²
    out_of_limits = (
        torch.square(torch.clamp(joint_pos - joint_pos_limits[:, :, 1], min=0.0))
        + torch.square(torch.clamp(joint_pos_limits[:, :, 0] - joint_pos, min=0.0))
    )

    return torch.sum(out_of_limits, dim=1)
```

**物理意义**:

```
# 假设关节限制 [-1.0, 1.0] rad
q = 0.5   → reward = 0.0    (在限制内)
q = 1.0   → reward = 0.0    (刚好在上界)
q = 1.2   → reward = 0.04   (超出0.2)
q = 1.5   → reward = 0.25   (超出0.5)
q = -1.2  → reward = 0.04   (超出下界0.2)
```

**为什么权重-5.0 (强惩罚)?**
- 硬约束，防止机械损坏
- 必须严格遵守
- 比其他惩罚强得多

### 9. 关节功率惩罚 (joint_power)

```python
def joint_power(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """惩罚关节功率"""
    asset: Articulation = env.scene[asset_cfg.name]

    # Power = |velocity × torque|
    # q̇: 关节速度 (rad/s)
    # τ: 关节力矩 (Nm)
    # Power: 瓦特 (W)
    return torch.sum(
        torch.abs(
            asset.data.joint_vel[:, asset_cfg.joint_ids]
            * asset.data.applied_torque[:, asset_cfg.joint_ids]
        ),
        dim=1
    )
```

**物理意义**:

```
# 单关节示例
q̇ = 1.0 rad/s
τ = 10 Nm
P = |1.0 × 10| = 10 W

# 12个关节
P_total = Σ P_i = 100 W
reward = -100
```

**为什么功率很重要?**
- 功率 = 能耗速度
- 电池续航考虑
- 电机发热问题

**为什么权重-2e-5?**
- 功率数量级大 (10-100 W)
- 鼓励节能
- 不牺牲性能

### 10. 静止站立惩罚 (stand_still)

```python
def stand_still(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """当无命令时，惩罚偏离默认站立位置"""
    asset: Articulation = env.scene[asset_cfg.name]

    # 计算关节位置偏差
    joint_pos_diff = torch.sum(
        torch.abs(
            asset.data.joint_pos[:, asset_cfg.joint_ids]
            - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
        ),
        dim=1
    )

    # 获取命令范数
    cmd_norm = torch.norm(
        env.command_manager.get_command(command_name),
        dim=1
    )

    # 只在命令接近0时惩罚
    # 使用乘法作为条件 (比where更高效)
    return joint_pos_diff * (cmd_norm < 0.1)
```

**物理意义**:

```
# 场景1: 有运动命令
cmd_norm = 1.0
joint_pos_diff = 0.5
reward = 0.5 × 0 = 0.0  # 无惩罚

# 场景2: 无命令，在站立位置
cmd_norm = 0.05
joint_pos_diff = 0.01
reward = 0.01 × 1 = 0.01  # 轻微惩罚

# 场景3: 无命令，偏离站立位置
cmd_norm = 0.05
joint_pos_diff = 1.0
reward = 1.0 × 1 = 1.0  # 强惩罚
```

**为什么用乘法而不是where?**
- PyTorch中乘法更快
- 自动微分更高效
- 代码更简洁

### 11. 关节位置偏离惩罚 (joint_pos_penalty)

```python
def joint_position_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
    velocity_threshold: float
) -> torch.Tensor:
    """惩罚关节位置偏离，停止时加重惩罚"""
    asset: Articulation = env.scene[asset_cfg.name]

    # 计算关节位置偏差
    joint_pos_diff = torch.sum(
        torch.abs(
            asset.data.joint_pos[:, asset_cfg.joint_ids]
            - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
        ),
        dim=1
    )

    # 获取命令和速度范数
    cmd_norm = torch.norm(
        env.command_manager.get_command(command_name),
        dim=1
    )
    body_vel = torch.norm(
        asset.data.root_lin_vel_b[:, :2],
        dim=1
    )

    # 确定缩放因子
    # 停止时 (命令小且速度小) → 5倍惩罚
    # 运动时 → 1倍惩罚
    scale = torch.where(
        torch.logical_or(cmd_norm > 0.1, body_vel > velocity_threshold),
        1.0,
        stand_still_scale
    )

    return joint_pos_diff * scale
```

**物理意义**:

```
# 场景1: 运动中
cmd_norm = 1.0, body_vel = 0.8
scale = 1.0
joint_pos_diff = 0.5
reward = 0.5 × 1.0 = 0.5

# 场景2: 停止
cmd_norm = 0.0, body_vel = 0.0
scale = 5.0  # stand_still_scale
joint_pos_diff = 0.5
reward = 0.5 × 5.0 = 2.5  # 5倍惩罚
```

**与stand_still的区别?**
- `stand_still`: 简单条件版本
- `joint_pos_penalty`: 考虑速度，更精确

---

## 动作平滑性奖励

### 12. 动作变化率惩罚 (action_rate_l2)

```python
def action_rate_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"
) -> torch.Tensor:
    """惩罚动作变化率"""
    asset: Articulation = env.scene[asset_cfg.name]

    # 计算动作变化
    # actions: 当前动作
    # previous_actions: 上一步动作
    action_delta = (
        asset.data.actions[:, asset_cfg.joint_ids]
        - asset.data.previous_actions[:, asset_cfg.joint_ids]
    )

    return torch.sum(
        torch.square(action_delta),
        dim=1
    )
```

**物理意义**:

```
# 单关节示例
action_t = 0.5
action_{t-1} = 0.4
delta = 0.1
reward = -0.01

# 剧烈变化
action_t = 1.0
action_{t-1} = 0.0
delta = 1.0
reward = -1.0
```

**为什么惩罚动作变化?**
- 防止动作抖动
- 平滑控制信号
- 保护电机
- 提高运动流畅性

**为什么权重-0.01?**
- 适度惩罚
- 不阻止必要的动作
- 平滑性与响应性平衡

### 13. 关节镜像对称惩罚 (joint_mirror)

```python
def joint_mirror(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    mirror_joints: list[list[str]]
) -> torch.Tensor:
    """惩罚左右不对称"""
    asset: Articulation = env.scene[asset_cfg.name]

    # 缓存关节ID对
    if not hasattr(env, "joint_mirror_joints_cache"):
        # 第一次调用时建立缓存
        env.joint_mirror_joints_cache = [
            [asset.find_joints(name) for name in pair]
            for pair in mirror_joints
        ]

    reward = torch.zeros(env.num_envs, device=env.device)

    # 对每一对关节计算差异
    for joint_pair in env.joint_mirror_joints_cache:
        # joint_pair[0][0]: 左侧关节ID
        # joint_pair[1][0]: 右侧关节ID
        reward += torch.sum(
            torch.square(
                asset.data.joint_pos[:, joint_pair[0][0]]
                - asset.data.joint_pos[:, joint_pair[1][0]]
            ),
            dim=1
        )

    return reward
```

**物理意义**:

```
# 配置的镜像对
mirror_joints = [
    ["FR_(hip|thigh|calf).*", "RL_(hip|thigh|calf).*"],  # 前右 ↔ 后左
    ["FL_(hip|thigh|calf).*", "RR_(hip|thigh|calf).*"],  # 前左 ↔ 后右
]

# 单关节示例
FR_hip_angle = 0.5
RL_hip_angle = 0.5
diff = 0.0  # 完美对称

FR_hip_angle = 0.5
RL_hip_angle = 0.7
diff = 0.2  # 不对称
reward = -0.04
```

**为什么是FR↔RL, FL↔RR?**
- 对角线对称
- 自然的步态模式
- 减少偏航

---

## 接触相关奖励

### 14. 非期望接触惩罚 (undesired_contacts)

```python
def undesired_contacts(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0
) -> torch.Tensor:
    """惩罚非脚部身体接触地面"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # net_forces_w: 世界坐标系的接触力
    # shape: (num_envs, num_bodies, 3)
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]

    # 计算Z轴力 (垂直于地面)
    forces_z = forces[:, :, 2]

    # 只考虑向下的力 (z < 0)
    # threshold = 1.0 N
    reward = torch.sum(
        torch.square(torch.clamp(forces_z, max=-threshold)),
        dim=1
    )

    return reward
```

**物理意义**:

```
# 单身体部位
force_z = -0.5 N  → reward = 0.0    (力太小，忽略)
force_z = -2.0 N  → reward = 1.0    (接触地面)
force_z = -10.0 N → reward = 81.0   (强接触)

# 多个部位
reward = Σ reward_i
```

**为什么只考虑向下力?**
- 向上力是地面的反作用力
- 向下力才是接触的证据

**配置**:
```python
undesired_contacts:
    body_names = ""  # 除了脚的所有身体
    # 包括: base, hip, thigh, calf等
    # 不包括: FR_foot, FL_foot等
```

### 15. 接触力惩罚 (contact_forces)

```python
def contact_forces(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 100.0
) -> torch.Tensor:
    """惩罚脚部过大接触力"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # net_forces_w: 接触力向量
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]

    # 计算力的大小
    force_magnitude = torch.norm(forces, dim=2)

    # 只考虑超过阈值的力
    # 鼓励平滑着地，避免冲击
    reward = torch.sum(
        torch.square(torch.clamp(force_magnitude - threshold, min=0.0)),
        dim=1
    )

    return reward
```

**物理意义**:

```
# 单脚
force = 50 N   → reward = 0.0    (小于阈值)
force = 100 N  → reward = 0.0    (等于阈值)
force = 150 N  → reward = 2500   (超出50 N)
force = 200 N  → reward = 10000  (超出100 N)
```

**为什么threshold=100 N?**
- 正常行走: ~50-80 N
- 跳跃: ~150-200 N
- 鼓励平滑着地

**为什么权重极小 (-1.5e-4)?**
- 接触力大是正常的
- 只是鼓励平滑
- 不希望主导训练

### 16. 脚部接触奖励 (feet_contact_without_cmd)

```python
def feet_contact_without_cmd(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity"
) -> torch.Tensor:
    """当无命令时，奖励脚部接触"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # current_contact_time > 0 表示接触
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    # 计算接触脚数
    num_contact = torch.sum(is_contact, dim=-1).float()

    # 获取命令范数
    cmd_norm = torch.norm(
        env.command_manager.get_command(command_name),
        dim=1
    )

    # 只在无命令时奖励
    return num_contact * (cmd_norm < 0.1)
```

**物理意义**:

```
# 场景1: 有命令
cmd_norm = 1.0
num_contact = 2
reward = 2 × 0 = 0.0  # 无奖励

# 场景2: 无命令，2脚接触
cmd_norm = 0.05
num_contact = 2
reward = 2 × 1 = 2.0  # 奖励

# 场景3: 无命令，4脚接触
cmd_norm = 0.05
num_contact = 4
reward = 4 × 1 = 4.0  # 更多奖励
```

**为什么是正奖励 (+0.1)?**
- 鼓励站立稳定
- 保持脚接触地面
- 防止抬起脚

---

## 扩展奖励函数

### 17. 动作镜像奖励 (action_mirror)

**源码位置**: `unitree_rl_lab/tasks/locomotion/mdp/extended_rewards.py`

```python
def action_mirror(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    mirror_joints: list[list[str]]
) -> torch.Tensor:
    """鼓励动作输出左右对称"""
    asset: Articulation = env.scene[asset_cfg.name]

    # 缓存关节ID对
    if not hasattr(env, "action_mirror_joints_cache"):
        env.action_mirror_joints_cache = [
            [asset.find_joints(name) for name in pair]
            for pair in mirror_joints
        ]

    reward = torch.zeros(env.num_envs, device=env.device)

    # 对每一对关节计算动作差异
    for joint_pair in env.action_mirror_joints_cache:
        reward += torch.sum(
            torch.square(
                asset.data.actions[:, joint_pair[0][0]]
                - asset.data.actions[:, joint_pair[1][0]]
            ),
            dim=1
        )

    return reward
```

**与joint_mirror的区别**:
- `joint_mirror`: 惩罚**状态** (joint_pos) 的差异
- `action_mirror`: 惩罚**动作** (actions) 的差异

**物理意义**:
```
# 状态 vs 动作
joint_mirror:     q_FR - q_RL  # 当前位置差异
action_mirror:    a_FR - a_RL  # 控制信号差异
```

**为什么初始weight=0?**
- 观察是否需要
- 根据训练情况调整
- 建议值: -0.01 到 -0.1

### 18. 动作同步奖励 (action_sync)

```python
def action_sync(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    joint_groups: list[list[str]]
) -> torch.Tensor:
    """鼓励同类关节同步运动"""
    asset: Articulation = env.scene[asset_cfg.name]

    # 缓存关节组ID
    if not hasattr(env, "action_sync_joints_cache"):
        env.action_sync_joints_cache = [
            [asset.find_joints(name) for name in group]
            for group in joint_groups
        ]

    reward = torch.zeros(env.num_envs, device=env.device)

    # 对每一组关节计算方差
    for joint_group in env.action_sync_joints_cache:
        # 提取该组的所有关节动作
        group_actions = asset.data.actions[:, joint_group[0]]

        # 计算方差
        # 方差 = E[(X - μ)²]
        group_mean = torch.mean(group_actions, dim=1, keepdim=True)
        group_variance = torch.mean(
            torch.square(group_actions - group_mean),
            dim=1
        )

        reward += group_variance

    return reward
```

**物理意义**:

```
# 髋关节组
FR_hip_action = 0.5
FL_hip_action = 0.6
RR_hip_action = 0.4
RL_hip_action = 0.5

mean = (0.5 + 0.6 + 0.4 + 0.5) / 4 = 0.5
variance = [(0.5-0.5)² + (0.6-0.5)² + (0.4-0.5)² + (0.5-0.5)²] / 4
         = [0 + 0.01 + 0.01 + 0] / 4
         = 0.005

reward = -0.005
```

**为什么鼓励同步?**
- 协调步态
- 规律运动
- 减少能耗

**配置**:
```python
action_sync:
    joint_groups = [
        ["FR_hip", "FL_hip", "RL_hip", "RR_hip"],      # 髋关节组
        ["FR_thigh", "FL_thigh", "RL_thigh", "RR_thigh"],  # 大腿组
        ["FR_calf", "FL_calf", "RL_calf", "RR_calf"],      # 小腿组
    ]
```

### 19. 轮子速度惩罚 (wheel_vel_penalty)

```python
def wheel_vel_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    velocity_threshold: float,
    command_threshold: float
) -> torch.Tensor:
    """当无命令时，惩罚轮子转动"""
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # 计算轮子速度
    wheel_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]

    # 计算脚部接触力
    contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    contact_force_z = contact_forces[:, :, 2]

    # 判断脚是否接触地面
    is_contact = torch.any(contact_force_z < -threshold, dim=1)

    # 获取命令范数
    cmd_norm = torch.norm(
        env.command_manager.get_command(command_name),
        dim=1
    )

    # 惩罚条件: 无命令 且 脚接触 且 轮子转动
    should_penalize = (
        (cmd_norm < command_threshold)
        & is_contact
        & (torch.norm(wheel_vel, dim=1) > velocity_threshold)
    )

    reward = torch.norm(wheel_vel, dim=1) * should_penalize.float()

    return reward
```

**物理意义**:

```
# 场景1: 有命令
cmd_norm = 1.0
wheel_vel = 5.0
reward = 0.0  # 不惩罚

# 场景2: 无命令，脚悬空
cmd_norm = 0.05
wheel_vel = 5.0
is_contact = False
reward = 0.0  # 不惩罚 (可能在调整)

# 场景3: 无命令，脚接触，轮子转动
cmd_norm = 0.05
wheel_vel = 5.0
is_contact = True
reward = 5.0  # 惩罚 (浪费能量)
```

**为什么惩罚?**
- 停止时轮子不应该转动
- 节省能量
- 防止打滑

**配置**:
```python
wheel_vel_penalty:
    weight = 0.0  # 初始禁用
    velocity_threshold = 0.5
    command_threshold = 0.1
```

---

## 奖励函数调用流程

### 计算顺序

```python
# 每个时间步
def compute_rewards(env):
    total_reward = 0.0

    # 1. 主要任务
    total_reward += track_lin_vel_xy_exp() * 3.0
    total_reward += track_ang_vel_z_exp() * 1.5

    # 2. 姿态稳定性
    total_reward += upward() * 1.0
    total_reward += lin_vel_z_l2() * -2.0
    total_reward += ang_vel_xy_l2() * -0.05

    # 3. 关节约束
    total_reward += joint_pos_limits() * -5.0
    total_reward += joint_torques_l2() * -2.5e-5
    total_reward += joint_acc_l2() * -2.5e-7
    total_reward += joint_power() * -2e-5

    # 4. 动作平滑
    total_reward += action_rate_l2() * -0.01
    total_reward += joint_mirror() * -0.05

    # 5. 接触相关
    total_reward += undesired_contacts() * -1.0
    total_reward += contact_forces() * -1.5e-4
    total_reward += feet_contact_without_cmd() * 0.1

    return total_reward
```

### 权重调优策略

**调优流程**:

1. **确定主要目标**
```python
# 速度跟踪是主要目标
track_lin_vel_xy_exp.weight = 3.0  # 基准
```

2. **添加辅助奖励**
```python
# 按重要性递减
track_ang_vel_z_exp.weight = 1.5   # 转向
upward.weight = 1.0                 # 姿态
```

3. **添加惩罚项**
```python
# 防止不良行为
lin_vel_z_l2.weight = -2.0          # 禁止跳动
joint_pos_limits.weight = -5.0      # 硬约束
```

4. **微调权重**
```python
# 观察训练曲线，调整
if 机器人跳动:
    lin_vel_z_l2.weight *= 2

if 能耗太高:
    joint_power.weight *= 2

if 运动不流畅:
    action_rate_l2.weight *= 2
```

### 奖励归一化

**问题**:
```python
# 不同奖励项数量级不同
track_lin_vel_xy_exp:  ~1.0
lin_vel_z_l2:          ~-1.0
joint_torques_l2:      ~-10000
```

**解决方案**:
```python
# 方法1: 调整权重
joint_torques_l2.weight = -2.5e-5  # 极小权重

# 方法2: 归一化奖励项
def normalized_lin_vel_z_l2(env):
    raw_vel = env.asset.data.root_lin_vel_b[:, 2]
    normalized_vel = torch.tanh(raw_vel / 0.5)  # 归一化到[-1, 1]
    return -torch.square(normalized_vel)
```

---

## 总结

### 奖励函数设计原则

1. **明确目标**: 速度跟踪是核心
2. **平衡约束**: 性能 vs 安全 vs 能效
3. **权重设计**: 主要任务 > 辅助任务 > 惩罚
4. **调试友好**: 可以单独禁用某项观察影响

### 常用权重范围

| 类别 | 典型权重 | 说明 |
|------|----------|------|
| 主要任务 | 1.0 - 5.0 | 高权重 |
| 辅助任务 | 0.5 - 2.0 | 中等权重 |
| 硬约束 | -5.0 - -10.0 | 强惩罚 |
| 软约束 | -0.01 - -1.0 | 轻惩罚 |
| 极小惩罚 | -1e-7 - -1e-4 | 数量级大时 |

### 代码实现技巧

1. **使用缓存**: 避免重复查找
2. **向量化**: 利用PyTorch并行
3. **条件判断**: 用乘法代替where
4. **物理意义**: 保持公式的可解释性

---

**文档版本**: 1.0
**最后更新**: 2026-03-08
**相关文档**: [GO2W训练过程全面解析](GO2W_TRAINING_PROCESS_ANALYSIS.md)
