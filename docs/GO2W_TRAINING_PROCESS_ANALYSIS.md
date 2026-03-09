# GO2W训练过程全面解析

## 目录
1. [训练流程概述](#训练流程概述)
2. [PPO算法详解](#ppo算法详解)
3. [观测空间与物理意义](#观测空间与物理意义)
4. [动作空间设计](#动作空间设计)
5. [奖励函数深度解析](#奖励函数深度解析)
6. [训练超参数物理意义](#训练超参数物理意义)
7. [课程学习机制](#课程学习机制)
8. [事件随机化策略](#事件随机化策略)
9. [训练指标监控](#训练指标监控)
10. [训练阶段与调优策略](#训练阶段与调优策略)

---

## 训练流程概述

### 整体架构

```
IsaacSim物理引擎
      ↓
ManagerBasedEnv (环境管理)
      ↓
观测向量 (41维) → 策略网络 (Actor) → 动作输出 (16维)
                         ↓
                    价值网络 (Critic)
                         ↓
                    PPO算法更新
```

### 训练循环流程

```python
# 每个迭代 (Iteration) 的流程
for iteration in range(max_iterations):
    # 1. 数据收集阶段 (Data Collection)
    for step in range(num_steps_per_env):  # 24步
        # 环境步进
        actions = policy.get_actions(observations)
        next_obs, rewards, dones, info = env.step(actions)

        # 存储到经验回放缓冲区
        buffer.add(observations, actions, rewards, dones, values)

    # 2. 优势函数计算 (Advantage Estimation)
    advantages = compute_gae(rewards, values, dones)  # GAE算法

    # 3. 策略更新阶段 (Policy Update)
    for epoch in range(num_learning_epochs):  # 5个epoch
        for mini_batch in mini_batches:  # 4个mini-batch
            # 计算新的概率和价值估计
            policy_loss, value_loss, entropy_loss = ppo_loss(mini_batch)

            # 反向传播更新网络
            optimizer.step()
            optimizer.zero_grad()

    # 4. 日志记录与保存
    if iteration % save_interval == 0:
        save_checkpoint()
    log_metrics()
```

### 时间与数据流

```
物理时间步长 (dt): 0.005s (200Hz)
控制频率: dt × decimation = 0.005s × 4 = 0.02s (50Hz)
 episodes长度: 20秒 = 1000个控制步

并行环境数: 4096
每迭代收集样本数: 4096 × 24 = 98,304 transitions
总训练时间: ~6-12小时 (取决于硬件)
```

---

## PPO算法详解

### PPO核心原理

PPO (Proximal Policy Optimization) 是一种策略梯度方法，通过限制策略更新幅度来保证训练稳定性。

### 数学原理

#### 1. 策略损失 (Policy Loss)

```
L^CLIP(θ) = Ê[t * min(r_t(θ) * Ê_t, clip(r_t(θ), 1-ε, 1+ε) * Ê_t)]
```

**物理意义**:
- `r_t(θ) = π_θ(a_t|s_t) / π_old(a_t|s_t)`: 重要性采样比率
  - 衡量新策略与旧策略的概率差异
  - `r_t > 1`: 新策略更倾向于这个动作
  - `r_t < 1`: 新策略更不倾向于这个动作

- `Ê_t`: 优势函数
  - `Ê_t > 0`: 这个动作比平均好
  - `Ê_t < 0`: 这个动作比平均差

- `clip(1-ε, 1+ε)`: 裁剪区间 (ε=0.2)
  - 限制策略更新幅度，防止灾难性遗忘
  - 如果比率超出 [0.8, 1.2]，则裁剪到边界

**直观理解**:
- 当 `Ê_t > 0` 且 `r_t > 1`: 好动作，增加概率
- 当 `Ê_t > 0` 但 `r_t < 0.8`: 不再增加，防止过度优化
- 当 `Ê_t < 0` 且 `r_t < 1`: 坏动作，减少概率
- 当 `Ê_t < 0` 但 `r_t > 1.2`: 不再减少，保持探索

#### 2. 价值损失 (Value Loss)

```
L^VF(θ) = 1/n * Σ(V_θ(s_t) - R_t)²
```

**物理意义**:
- `V_θ(s_t)`: 状态价值估计（预期未来折扣奖励）
- `R_t`: 实际回报 (Return)
- 最小化均方误差，让价值网络准确预测状态价值

**作用**:
- 提供基线 (baseline)，减少方差
- 加速训练收敛
- 评估策略好坏

#### 3. 熵损失 (Entropy Loss)

```
L^entropy(θ) = -Σ π_θ(a|s) * log π_θ(a|s)
```

**物理意义**:
- 熵衡量策略的随机性/多样性
- 高熵 = 策略分布均匀，探索性强
- 低熵 = 策略集中，利用性强

**作用**:
- 鼓励探索，防止过早收敛
- `entropy_coef = 0.01` 平衡探索与利用

#### 4. 总损失函数

```
L_total(θ) = L^CLIP(θ) + c1 * L^VF(θ) - c2 * L^entropy(θ)
```

其中 `c1 = value_loss_coef = 1.0`, `c2 = entropy_coef = 0.01`

### GO2W训练配置

```python
# 文件: source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/agents/rsl_rl_ppo_cfg.py

algorithm = RslRlPpoAlgorithmCfg(
    # 价值损失系数
    value_loss_coef = 1.0,              # 价值损失的权重

    # 裁剪参数
    use_clipped_value_loss = True,      # 对价值损失也进行裁剪
    clip_param = 0.2,                   # PPO裁剪范围 [0.8, 1.2]

    # 熵正则化
    entropy_coef = 0.01,                # 熵系数，鼓励探索

    # 优化参数
    num_learning_epochs = 5,            # 每批数据学习5轮
    num_mini_batches = 4,               # 每轮分为4个mini-batch

    # 学习率
    learning_rate = 1e-3,               # 初始学习率 0.001
    schedule = "adaptive",              # 自适应学习率调度

    # GAE参数
    gamma = 0.99,                       # 折扣因子
    lam = 0.95,                         # GAE λ参数

    # KL散度控制
    desired_kl = 0.01,                  # 目标KL散度

    # 梯度裁剪
    max_grad_norm = 1.0,                # 梯度范数上限
)
```

### GAE (Generalized Advantage Estimation)

**公式**:
```
δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
Ê_t^GAE = Σ (γλ)^{l} * δ_{t+l}
```

**物理意义**:
- `δ_t`: 时间差分误差 (TD error)
  - 正值: 实际奖励好于预期
  - 负值: 实际奖励差于预期

- `Ê_t^GAE`: 广义优势估计
  - `λ=0`: 只用当前TD误差 (高偏差，低方差)
  - `λ=1`: 完全蒙特卡洛估计 (低偏差，高方差)
  - `λ=0.95`: 偏差-方差权衡

**为什么使用GAE?**
- 平滑优势估计，减少噪声
- 平衡偏差(variance)和方差(bias)
- 加速训练收敛

---

## 观测空间与物理意义

### 观测空间组成

**Flat模式观测维度**: 41维
```
总观测 = base_ang_vel (3) + projected_gravity (3) + velocity_commands (3)
       + joint_pos_leg (12) + joint_vel_wheel (4) + actions (16)
       = 41 维
```

### 详细观测项

#### 1. 基座角速度 (base_ang_vel) - 3维

```python
base_ang_vel = [ω_x, ω_y, ω_z]
# scale = 0.25
# noise = ±0.2 rad/s
```

**物理意义**:
- `ω_x`: 俯仰角速度 (绕X轴旋转)
- `ω_y`: 翻滚角速度 (绕Y轴旋转)
- `ω_z`: 偏航角速度 (绕Z轴旋转)

**作用**:
- 检测旋转运动，保持姿态稳定
- 调整步态以响应转向命令
- 缩放0.25: 限制敏感度，防止震荡

#### 2. 投影重力向量 (projected_gravity) - 3维

```python
projected_gravity = [g_x, g_y, g_z]
# scale = 1.0
# noise = ±0.05
```

**物理意义**:
- 重力加速度在机器人坐标系中的投影
- `g_z ≈ 9.8`: 机器人直立
- `g_z < 9.8`: 机器人倾斜

**作用**:
- **关键状态信息**: 比陀螺仪更可靠
- 姿态估计: 知道是否倾斜
- 平衡控制: 保持直立姿态

#### 3. 速度命令 (velocity_commands) - 3维

```python
velocity_commands = [v_x_cmd, v_y_cmd, ω_z_cmd]
# 范围: v_x, v_y ∈ [-1.0, 1.0] m/s, ω_z ∈ [-1.0, 1.0] rad/s
# resampling_time = 10.0s
```

**物理意义**:
- `v_x_cmd`: 前进/后退命令
- `v_y_cmd`: 左右平移命令
- `ω_z_cmd`: 原地转向命令

**命令生成策略**:
```python
# 每10秒重新采样命令
rel_standing_envs = 0.02  # 2%的环境静止
rel_heading_envs = 1.0    # 100%的环境使用航向控制
```

**作用**:
- 指导机器人运动目标
- 多任务学习: 不同速度和方向
- 课程学习: 从简单到复杂命令

#### 4. 腿部关节位置 (joint_pos_leg) - 12维

```python
joint_pos_leg = [
    FR_hip, FR_thigh, FR_calf,    # 前右腿
    FL_hip, FL_thigh, FL_calf,    # 前左腿
    RR_hip, RR_thigh, RR_calf,    # 后右腿
    RL_hip, RL_thigh, RL_calf,    # 后左腿
]
# scale = 1.0
# noise = ±0.01 rad
# 相对于默认位置 (joint_pos_rel)
```

**物理意义**:
- 关节角度 (弧度)
- 相对位置: 当前角度 - 默认站立角度

**各关节作用**:
- `hip`: 髋关节，控制腿部前后摆动
- `thigh`: 大腿关节，控制腿部高度
- `calf`: 小腿关节，控制腿部伸缩

**作用**:
- 知道当前姿态
- 调整步态
- 协调四腿运动

#### 5. 轮子关节速度 (joint_vel_wheel) - 4维

```python
joint_vel_wheel = [FR_wheel, FL_wheel, RR_wheel, RL_wheel]
# scale = 0.05
# noise = ±1.5 rad/s
```

**物理意义**:
- 轮子转速 (rad/s)
- 控制机器人移动的主要执行器

**作用**:
- 知道当前轮速
- 调整轮速以跟踪速度命令
- 保持轮子同步

#### 6. 上一步动作 (last_action) - 16维

```python
last_action = [
    leg_joints_pos_action (12维),  # 腿部位置控制
    wheel_joints_vel_action (4维),  # 轮子速度控制
]
```

**物理意义**:
- 上一时间步的动作输出
- 12维腿部位置 + 4维轮子速度 = 16维

**作用**:
- 提供时间上下文
- 平滑动作序列
- 防止动作抖动

### 观测归一化

```python
# 归一化策略
base_ang_vel: scale=0.25     # 缩小范围，降低敏感度
joint_vel: scale=0.05        # 大幅缩小，速度变化小
joint_pos: scale=1.0         # 保持原始范围
```

**为什么归一化?**
- 不同观测维度物理单位不同
- 加速神经网络训练
- 防止某些维度主导训练

---

## 动作空间设计

### 混合动作空间

GO2W采用**位置+速度混合控制**:

```python
动作空间 = [
    腿部关节位置控制 (12维) + 轮子速度控制 (4维)
    = 16维
]
```

### 腿部关节位置控制

```python
joint_pos_action = mdp.JointPositionActionCfg(
    joint_names = leg_joint_names,  # 12个腿部关节
    scale = {
        ".*_hip_joint": 0.125,      # 髋关节: 小幅度
        "^(?!.*_hip_joint).*": 0.25 # 其他关节: 大幅度
    },
    use_default_offset = True,      # 增量控制
    clip = (-100.0, 100.0),         # 安全限制
)
```

**物理意义**:

#### 增量控制模式
```python
实际关节位置 = 默认位置 + (网络输出 × scale)
```

**为什么髋关节scale更小 (0.125 vs 0.25)?**
- 髋关节靠近机身，影响更大
- 小幅调整避免姿态失稳
- 大腿和小腿可以更大幅度调整步态

#### 关节分组
```
FR_hip (0.125)  FR_thigh (0.25)  FR_calf (0.25)
FL_hip (0.125)  FL_thigh (0.25)  FL_calf (0.25)
RR_hip (0.125)  RR_thigh (0.25)  RR_calf (0.25)
RL_hip (0.125)  RL_thigh (0.25)  RL_calf (0.25)
```

### 轮子速度控制

```python
joint_vel_action = mdp.JointVelocityActionCfg(
    joint_names = wheel_joint_names,  # 4个轮子
    scale = 5.0,                       # 速度增量幅度
    use_default_offset = True,         # 增量控制
    clip = (-100.0, 100.0),            # 安全限制
)
```

**物理意义**:

#### 增量速度控制
```python
实际轮速 = 默认速度 + (网络输出 × 5.0)
```

**为什么scale=5.0?**
- 轮子是主要移动执行器，需要快速响应
- 较大的scale允许更快的速度调整
- 配合速度跟踪奖励 (weight=3.0)

### 动作空间特性

#### 1. 周期性 (Cyclic)
- 关节位置周期性摆动
- 形成步态 (gait)

#### 2. 连续性 (Continuous)
- 动作输出是连续值
- 需要平滑过渡

#### 3. 约束性 (Constrained)
- 位置限制: 关节角度有物理限制
- 速度限制: 电机有最大速度

#### 4. 协调性 (Coordinated)
- 四腿需要协调
- 腿轮需要配合

---

## 奖励函数深度解析

### 奖励函数分类

```
总奖励 = 奖励项₁ × weight₁ + 奖励项₂ × weight₂ + ... + 奖励项ₙ × weightₙ
```

#### 权重分类
- **正权重** (weight > 0): 鼓励行为
- **负权重** (weight < 0): 惩罚行为
- **零权重** (weight = 0): 禁用该奖励项

### 核心奖励项

#### 1. 线速度跟踪奖励 (track_lin_vel_xy_exp)

```python
track_lin_vel_xy_exp:
    weight = 3.0          # 最高的正权重
    func = exp_kernel     # 指数核函数
    std = 0.5
```

**公式**:
```python
error = ||v_actual - v_command||
reward = exp(-error² / (2 × std²))
```

**物理意义**:
- `v_actual`: 实际速度 [v_x, v_y]
- `v_command`: 命令速度 [v_x_cmd, v_y_cmd]
- `error`: 速度误差

**特性**:
- 完美跟踪: `error=0` → `reward=1.0`
- 一个std误差: `error=0.5` → `reward=0.607`
- 两个std误差: `error=1.0` → `reward=0.135`

**为什么是指数核?**
- 平滑奖励曲线
- 鼓励接近目标，不强制完美
- 比线性奖励更鲁棒

**为什么权重最高 (3.0)?**
- **主要训练目标**: 让机器人会移动
- 其他所有奖励都是辅助

#### 2. 角速度跟踪奖励 (track_ang_vel_z_exp)

```python
track_ang_vel_z_exp:
    weight = 1.5          # 第二高权重
    func = exp_kernel
    std = 0.5
```

**公式**:
```python
error = |ω_z_actual - ω_z_command|
reward = exp(-error² / (2 × std²))
```

**物理意义**:
- `ω_z`: 绕Z轴旋转速度
- 控制转向能力

**权重为什么是线速度的一半 (1.5 vs 3.0)?**
- 直线运动比转向更重要
- 转向频率较低

#### 3. 保持直立奖励 (upward)

```python
upward:
    weight = 1.0
    func = upward
```

**公式**:
```python
reward = (1 - g_z)²
```

**物理意义**:
- `g_z`: 投影重力向量的Z分量
- 直立时: `g_z ≈ 1.0` → `reward ≈ 0`
- 倒下时: `g_z << 1.0` → `reward ≈ 1.0`

**为什么是平方?**
- 小倾斜时惩罚小
- 大倾斜时惩罚大
- 非线性增强

### 惩罚项详解

#### 4. Z轴线速度惩罚 (lin_vel_z_l2)

```python
lin_vel_z_l2:
    weight = -2.0
    func = l2_norm
```

**公式**:
```python
reward = -v_z²
```

**物理意义**:
- `v_z`: 垂直速度
- 惩罚上下跳动

**为什么惩罚?**
- 轮腿机器人应该平稳移动
- 跳动浪费能量
- 可能导致失稳

**权重 -2.0 的原因**:
- 防止"跳跃式"移动
- 比其他惩罚更强

#### 5. XY轴角速度惩罚 (ang_vel_xy_l2)

```python
ang_vel_xy_l2:
    weight = -0.05
    func = l2_norm
```

**公式**:
```python
reward = -(ω_x² + ω_y²)
```

**物理意义**:
- `ω_x, ω_y`: 俯仰和翻滚角速度
- 惩罚前后/左右摇晃

**为什么权重很小 (-0.05)?**
- 允许小幅摇晃 (自然步态)
- 主要防止过度摇晃
- 不像Z轴速度那么严格

#### 6. 关节力矩惩罚 (joint_torques_l2)

```python
joint_torques_l2:
    weight = -2.5e-5     # 非常小的负权重
    func = l2_norm
    joint_names = leg_joint_names  # 只惩罚腿部
```

**公式**:
```python
reward = -Σ τ²
```

**物理意义**:
- `τ`: 关节力矩
- 惩罚大力矩

**为什么权重这么小 (-0.000025)?**
- 力矩数量级较大 (10-100 Nm)
- 避免主导其他奖励
- 鼓励能效，不牺牲性能

**为什么只惩罚腿部?**
- 腿部需要精确控制
- 轮子需要大扭矩移动

#### 7. 关节速度惩罚 (joint_vel_l2)

```python
joint_vel_l2:
    weight = 0.0         # 禁用
    joint_names = leg_joint_names
```

**为什么禁用?**
- 腿部需要快速运动
- 速度不是主要问题
- 加速度更重要

#### 8. 关节加速度惩罚 (joint_acc_l2)

```python
joint_acc_l2:
    weight = -2.5e-7     # 极小权重
    joint_names = leg_joint_names
```

**公式**:
```python
reward = -Σ α²
```

**物理意义**:
- `α`: 关节加速度 (α = Δv/Δt)
- 惩罚快速变化

**作用**:
- 平滑运动
- 减少机械磨损
- 节省能量

#### 9. 动作变化率惩罚 (action_rate_l2)

```python
action_rate_l2:
    weight = -0.01
    func = l2_norm
```

**公式**:
```python
reward = -||action_t - action_{t-1}||²
```

**物理意义**:
- 惩罚动作突变
- 鼓励平滑控制

**作用**:
- 防止抖动
- 保护电机
- 提高运动流畅性

#### 10. 关节位置限制惩罚 (joint_pos_limits)

```python
joint_pos_limits:
    weight = -5.0        # 强惩罚
    func = joint_pos_limits
    joint_names = leg_joint_names
```

**公式**:
```python
reward = -Σ max(0, q - q_max)² + max(0, q_min - q)²
```

**物理意义**:
- `q`: 关节角度
- `q_min, q_max`: 角度限制
- 超出限制时给予大惩罚

**为什么权重 -5.0?**
- 防止机械损坏
- 硬约束，必须遵守

#### 11. 关节功率惩罚 (joint_power)

```python
joint_power:
    weight = -2e-5
    func = joint_power
    joint_names = leg_joint_names
```

**公式**:
```python
reward = -Σ |q̇ × τ|
```

**物理意义**:
- `q̇`: 关节速度
- `τ`: 关节力矩
- `Power = q̇ × τ`

**作用**:
- 鼓励节能
- 防止电机过载

#### 12. 静止站立惩罚 (stand_still)

```python
stand_still:
    weight = -2.0
    func = stand_still
    joint_names = leg_joint_names
```

**公式**:
```python
cmd_norm = ||v_command||
reward = -Σ ||q - q_default|| × (cmd_norm < 0.1)
```

**物理意义**:
- 当命令速度接近0时
- 惩罚偏离默认站立位置

**作用**:
- 鼓励停止时回到站立姿态
- 为下次移动做准备

#### 13. 关节位置偏离惩罚 (joint_pos_penalty)

```python
joint_pos_penalty:
    weight = -1.0
    func = joint_pos_penalty
    joint_names = leg_joint_names
```

**公式**:
```python
cmd_norm = ||v_command||
standing_scale = (cmd_norm < threshold) ? 5.0 : 1.0
reward = -Σ ||q - q_default|| × standing_scale
```

**物理意义**:
- 类似stand_still，但更复杂
- 停止时惩罚加重

#### 14. 关节镜像对称惩罚 (joint_mirror)

```python
joint_mirror:
    weight = -0.05
    func = joint_mirror
    mirror_joints = [
        ["FR_(hip|thigh|calf).*", "RL_(hip|thigh|calf).*"],
        ["FL_(hip|thigh|calf).*", "RR_(hip|thigh|calf).*"],
    ]
```

**公式**:
```python
reward = -Σ ||action_left - action_right||²
```

**物理意义**:
- 鼓励左右对称
- FR (前右) ↔ RL (后左)
- FL (前左) ↔ RR (后右)

**作用**:
- 自然步态
- 减少偏航
- 提高稳定性

#### 15. 非期望接触惩罚 (undesired_contacts)

```python
undesired_contacts:
    weight = -1.0
    func = undesired_contacts
    body_names = ""  # 除了脚的所有身体
```

**公式**:
```python
reward = -Σ max(0, contact_force - threshold)²
```

**物理意义**:
- 惩罚非脚部接触地面
- 防止肚子、腿、头触地

**作用**:
- 保持正确姿态
- 防止摔倒

#### 16. 接触力惩罚 (contact_forces)

```python
contact_forces:
    weight = -1.5e-4   # 极小权重
    func = contact_forces
    body_names = ".*_foot"
```

**公式**:
```python
reward = -Σ contact_force²
```

**物理意义**:
- 惩罚过大接触力
- 保护脚部和地面

**为什么权重极小?**
- 接触力大是正常的
- 只是鼓励平滑着地
- 不希望主导训练

#### 17. 脚部接触奖励 (feet_contact_without_cmd)

```python
feet_contact_without_cmd:
    weight = 0.1       # 正权重
    func = feet_contact_without_cmd
```

**物理意义**:
- 当无速度命令时
- 鼓励脚保持接触

**作用**:
- 站立稳定
- 防止抬起脚

### 新增奖励项 (基于msz006_go2w)

#### 18. 动作镜像奖励 (action_mirror)

```python
action_mirror:
    weight = 0.0       # 初始禁用
    func = action_mirror
    mirror_joints = [
        ["FR_(hip|thigh|calf).*", "RL_(hip|thigh|calf).*"],
        ["FL_(hip|thigh|calf).*", "RR_(hip|thigh|calf).*"],
    ]
```

**与joint_mirror的区别**:
- `joint_mirror`: 惩罚当前状态的差异
- `action_mirror`: 惩罚动作输出的差异

**建议调整**:
```python
# 如果步态不对称，可以启用
weight = -0.01 到 -0.1
```

#### 19. 动作同步奖励 (action_sync)

```python
action_sync:
    weight = 0.0       # 初始禁用
    func = action_sync
    joint_groups = [
        ["FR_hip", "FL_hip", "RL_hip", "RR_hip"],
        ["FR_thigh", "FL_thigh", "RL_thigh", "RR_thigh"],
        ["FR_calf", "FL_calf", "RL_calf", "RR_calf"],
    ]
```

**物理意义**:
- 鼓励同类关节同步运动
- 例如: 四个髋关节同时动作

**作用**:
- 协调步态
- 规律运动

### 奖励权重总结

| 类别 | 奖励项 | 权重 | 作用 |
|------|--------|------|------|
| **主要任务** | track_lin_vel_xy_exp | 3.0 | 线速度跟踪 |
| | track_ang_vel_z_exp | 1.5 | 角速度跟踪 |
| **姿态** | upward | 1.0 | 保持直立 |
| | lin_vel_z_l2 | -2.0 | 禁止跳动 |
| | ang_vel_xy_l2 | -0.05 | 减少摇晃 |
| **关节** | joint_pos_limits | -5.0 | 位置限制 |
| | stand_still | -2.0 | 站立姿态 |
| | joint_pos_penalty | -1.0 | 位置偏离 |
| | joint_torques_l2 | -2.5e-5 | 力矩惩罚 |
| | joint_acc_l2 | -2.5e-7 | 加速度惩罚 |
| | joint_power | -2e-5 | 功率惩罚 |
| **动作** | action_rate_l2 | -0.01 | 平滑控制 |
| | joint_mirror | -0.05 | 对称步态 |
| **接触** | undesired_contacts | -1.0 | 非期望接触 |
| | contact_forces | -1.5e-4 | 接触力 |
| | feet_contact_without_cmd | 0.1 | 脚部接触 |

---

## 训练超参数物理意义

### 学习率 (Learning Rate)

```python
learning_rate = 1e-3  # 0.001
schedule = "adaptive"  # 自适应调整
```

**物理意义**:
- 控制参数更新步长
- `Δθ = -lr × ∇L`

**为什么是0.001?**
- 太大: 训练不稳定，震荡
- 太小: 收敛慢
- 0.001是PPO的经典值

**自适应调度**:
```python
if KL_divergence < desired_kl / 2:
    learning_rate *= 1.5  # 增加学习率
elif KL_divergence > desired_kl * 2:
    learning_rate *= 0.5  # 减少学习率
```

### 折扣因子 (Gamma)

```python
gamma = 0.99
```

**物理意义**:
- 未来奖励的折扣率
- `Return = Σ γ^t × r_t`

**为什么是0.99?**
- `γ=0.99`: 100步后的奖励权重 = 0.99^100 ≈ 0.37
- `γ=1.0`: 完全长远，训练不稳定
- `γ=0.9`: 过于短视

**实际效果**:
```python
# 1秒后的奖励 (50步)
weight = 0.99^50 ≈ 0.605

# 5秒后的奖励 (250步)
weight = 0.99^250 ≈ 0.082

# 10秒后的奖励 (500步)
weight = 0.99^500 ≈ 0.007
```

**启发式**:
- `episode_length = 20s` (1000步)
- `γ=0.99`: 机器人关心前7秒左右的奖励

### GAE参数 (Lambda)

```python
lam = 0.95
```

**物理意义**:
- 平衡偏差和方差
- `λ=0`: 低偏差，高方差
- `λ=1`: 高偏差，低方差

**为什么是0.95?**
- 接近1，但不是1
- 充分利用未来奖励
- 保留一定的偏差

**对比**:
```
λ=0.90: 更关注即时奖励，训练快但策略次优
λ=0.95: 平衡，推荐
λ=0.99: 更关注长期奖励，训练慢但策略优
```

### PPO裁剪参数 (Clip Param)

```python
clip_param = 0.2  # ε
```

**物理意义**:
- 限制策略更新幅度
- `ratio ∈ [0.8, 1.2]`

**为什么是0.2?**
- 原论文推荐值
- 太小: 更新太慢
- 太大: 训练不稳定

**实际效果**:
```python
old_prob = 0.3
new_prob = 0.5
ratio = 0.5 / 0.3 = 1.67  # 超出范围
clipped_ratio = min(1.67, 1.2) = 1.2  # 裁剪
```

### 熵系数 (Entropy Coef)

```python
entropy_coef = 0.01
```

**物理意义**:
- 探索与利用的权衡
- `L_total = L_policy + c1×L_value - c2×L_entropy`

**为什么是0.01?**
- 太小: 探索不足，局部最优
- 太大: 随机策略，不收敛

**实际效果**:
```
训练初期: 高熵，多探索
训练后期: 低熵，多利用
```

### 批次参数

```python
num_steps_per_env = 24       # 每个环境收集24步
num_learning_epochs = 5      # 每批数据学习5轮
num_mini_batches = 4         # 每轮分4个mini-batch
```

**物理意义**:

#### 数据收集
```python
总样本数 = num_envs × num_steps_per_env
         = 4096 × 24
         = 98,304 transitions
```

#### 学习轮次
```python
每个样本使用次数 = num_learning_epochs × num_mini_batches
                 = 5 × 4
                 = 20 次
```

#### Mini-batch大小
```python
mini_batch_size = 总样本数 / num_mini_batches
                = 98,304 / 4
                = 24,576
```

**为什么这样设置?**
- `24步`: 平衡数据量和时间
- `5轮`: 充分利用数据，不过拟合
- `4个批次`: GPU内存利用率

### 梯度裁剪 (Max Grad Norm)

```python
max_grad_norm = 1.0
```

**物理意义**:
- 限制梯度范数
- 防止梯度爆炸

**公式**:
```python
if ||g|| > max_grad_norm:
    g = g / ||g|| × max_grad_norm
```

### KL散度目标 (Desired KL)

```python
desired_kl = 0.01
```

**物理意义**:
- 衡量策略更新幅度
- `KL(p_old || p_new) ≈ 0.01`

**作用**:
- 自适应学习率调整
- 早停机制

**公式**:
```python
if KL > desired_kl × 2:
    停止更新，策略变化太大
```

---

## 课程学习机制

### 地形课程 (Terrain Levels)

```python
curriculum.terrain_levels = CurrTerm(
    func=mdp.terrain_levels_vel
)
```

**物理意义**:
- 根据性能自动调整地形难度
- 从简单到复杂

**级别划分**:
```
Level 0: 平面
Level 1: 矮障碍
Level 2: 中等障碍
Level 3: 高障碍
Level 4: 楼梯
Level 5: 斜坡
```

**触发条件**:
```python
# 基于速度跟踪奖励
if mean_reward > threshold:
    terrain_level += 1  # 升级
else:
    terrain_level -= 1  # 降级
```

### 命令课程 (Command Levels)

```python
curriculum.command_levels = CurrTerm(
    func=mdp.command_levels_vel,
    params={
        "reward_term_name": "track_lin_vel_xy_exp",
        "range_multiplier": (0.1, 1.0),  # 从10%到100%命令
    }
)
```

**物理意义**:
- 从低速到高速
- 逐步增加任务难度

**课程阶段**:
```
阶段1: range_multiplier = 0.1
        命令范围: [-0.1, 0.1] m/s

阶段2: range_multiplier = 0.3
        命令范围: [-0.3, 0.3] m/s

...

阶段N: range_multiplier = 1.0
        命令范围: [-1.0, 1.0] m/s
```

**为什么课程学习有效?**
- 避免过早陷入困难任务
- 先学会简单任务，再迁移
- 加速训练收敛

---

## 事件随机化策略

### 启动时随机化 (Startup Events)

#### 1. 材质随机化 (Material)

```python
randomize_rigid_body_material:
    mode = "startup"
    static_friction_range = (0.3, 1.0)
    dynamic_friction_range = (0.3, 0.8)
    restitution_range = (0.0, 0.5)
```

**物理意义**:
- `static_friction`: 静摩擦系数 (0.3-1.0)
- `dynamic_friction`: 动摩擦系数 (0.3-0.8)
- `restitution`: 弹性系数 (0.0-0.5)

**作用**:
- 适应不同地面
- 提高鲁棒性
- 防止过拟合特定摩擦力

#### 2. 质量随机化 (Mass)

```python
# 基座质量
randomize_rigid_body_mass_base:
    mode = "startup"
    mass_distribution_params = (-1.0, 3.0)  # kg
    operation = "add"

# 其他部件质量
randomize_rigid_body_mass_others:
    mode = "startup"
    mass_distribution_params = (0.7, 1.3)
    operation = "scale"
    recompute_inertia = True
```

**物理意义**:
- 基座: ±1到3kg质量变化
- 其他: 70%-130%质量缩放
- 自动重新计算转动惯量

**作用**:
- 适应不同负载
- 提高鲁棒性
- 模拟实际使用场景

#### 3. 质心随机化 (COM)

```python
randomize_com_positions:
    mode = "startup"
    com_range = {
        "x": (-0.05, 0.05),
        "y": (-0.05, 0.05),
        "z": (-0.05, 0.05)
    }
```

**物理意义**:
- 质心位置偏移 ±5cm
- 影响稳定性

**作用**:
- 适应质心变化
- 提高平衡能力

#### 4. 转动惯量随机化 (Inertia)

```python
randomize_rigid_body_inertia:
    mode = "startup"
    inertia_distribution_params = (0.8, 1.2)  # 80%-120%
    operation = "scale"
    distribution = "uniform"
```

**物理意义**:
- 转动惯量缩放
- I_xx, I_yy, I_zz 独立缩放

**作用**:
- 适应不同惯性特性
- 提高转向稳定性

### 重置时随机化 (Reset Events)

#### 5. 关节随机化 (Joints)

```python
randomize_reset_joints:
    mode = "reset"
    position_range = (1.0, 1.0)   # 无随机化
    velocity_range = (0.0, 0.0)   # 零速度
```

**物理意义**:
- 重置到默认站立位置
- 零速度

#### 6. 基座随机化 (Base)

```python
randomize_reset_base:
    mode = "reset"
    pose_range = {
        "x": (-0.5, 0.5),
        "y": (-0.5, 0.5),
        "yaw": (-3.14, 3.14)  # 任意朝向
    }
    velocity_range = {
        "x": (-0.5, 0.5),
        "y": (-0.5, 0.5),
        "z": (-0.5, 0.5),
        "roll": (-0.5, 0.5),
        "pitch": (-0.5, 0.5),
        "yaw": (-0.5, 0.5)
    }
```

**物理意义**:
- 位置: ±0.5m随机偏移
- 朝向: 任意方向
- 速度: ±0.5 m/s或rad/s

**作用**:
- 防止位置过拟合
- 学习从任意状态恢复

#### 7. 执行器增益随机化 (Actuator Gains)

```python
randomize_actuator_gains:
    mode = "reset"
    stiffness_distribution_params = (0.5, 2.0)  # 刚度
    damping_distribution_params = (0.5, 2.0)    # 阻尼
    operation = "scale"
```

**物理意义**:
- PD控制器增益随机化
- 刚度: 50%-200%
- 阻尼: 50%-200%

**作用**:
- 适应不同控制器参数
- 提高鲁棒性

#### 8. 外力扰动 (External Force)

```python
randomize_apply_external_force_torque:
    mode = "reset"
    force_range = (-10.0, 10.0)    # N
    torque_range = (-10.0, 10.0)   # Nm
```

**物理意义**:
- 重置时施加随机力/力矩
- 模拟外部扰动

**作用**:
- 学习抗干扰能力
- 提高稳定性

### 间隔随机化 (Interval Events)

#### 9. 推机器人 (Push Robot)

```python
randomize_push_robot:
    mode = "interval"
    interval_range_s = (10.0, 15.0)  # 每10-15秒
    velocity_range = {
        "x": (-0.5, 0.5),
        "y": (-0.5, 0.5)
    }
```

**物理意义**:
- 每10-15秒推一下
- 设置随机速度

**作用**:
- 持续扰动
- 学习动态平衡

### 随机化总结

| 事件 | 模式 | 频率 | 作用 |
|------|------|------|------|
| 材质随机化 | startup | 每episode开始 | 适应不同地面 |
| 质量随机化 | startup | 每episode开始 | 适应不同负载 |
| 质心随机化 | startup | 每episode开始 | 适应质心变化 |
| 惯量随机化 | startup | 每episode开始 | 适应惯性特性 |
| 关节重置 | reset | 每episode | 从站立姿态开始 |
| 基座重置 | reset | 每episode | 随机位置和朝向 |
| 增益随机化 | reset | 每episode | 适应控制器参数 |
| 外力扰动 | reset | 每episode | 抗干扰能力 |
| 推机器人 | interval | 每10-15秒 | 动态平衡 |

---

## 训练指标监控

### TensorBoard关键指标

#### 1. 奖励指标

**总奖励 (Reward/total)**
```
物理意义: 所有奖励项的加权和
单位: 无量纲
范围: 通常 [-10, 10]
```

**分量奖励**:
```
Reward/track_lin_vel_xy_exp    # 线速度跟踪，应该逐渐增大
Reward/track_ang_vel_z_exp     # 角速度跟踪，应该逐渐增大
Reward/lin_vel_z_l2            # Z轴速度，应该接近0
Reward/joint_torques_l2        # 关节力矩，越小越好
```

**解读**:
```python
# 训练良好的标志
track_lin_vel_xy_exp > 2.0     # > 66% 性能
lin_vel_z_l2 > -0.5            # 轻微上下移动
total_reward > 0.0             # 正奖励为主
```

#### 2. 价值函数指标

**价值损失 (Value/value_loss)**
```
物理意义: 价值网络预测误差
单位: 无量纲
范围: [0, +∞)
目标: 越小越好
```

**解读**:
```python
# 训练收敛的标志
value_loss < 0.1    # 价值预测准确
value_loss 稳定     # 不再下降
```

#### 3. 策略指标

**策略损失 (Policy/policy_loss)**
```
物理意义: PPO裁剪目标
单位: 无量纲
范围: (-∞, 0]
目标: 逐渐增大（负值减小）
```

**解读**:
```python
# 训练正常的标志
policy_loss < 0      # 总是负值
policy_loss 增大     # 策略改进
policy_loss 稳定     # 收敛
```

#### 4. 熵指标

**策略熵 (Policy/entropy)**
```
物理意义: 策略的随机性
单位: nat (信息单位)
范围: [0, +∞)
目标: 逐渐减小（从探索到利用）
```

**解读**:
```python
# 训练阶段的标志
entropy > 1.0        # 初期，高探索
entropy ≈ 0.5       # 中期，平衡
entropy < 0.2        # 后期，确定性策略
```

**熵过小问题**:
```
entropy < 0.01:      # 策略过早收敛
→ 解决: 增大entropy_coef
```

#### 5. KL散度指标

**KL散度 (Policy/kl_divergence)**
```
物理意义: 策略更新幅度
单位: nat
范围: [0, +∞)
目标: 接近desired_kl (0.01)
```

**解读**:
```python
# 训练稳定的标志
kl ≈ 0.01           # 理想范围
kl > 0.02           # 更新太大，学习率减半
kl < 0.005          # 更新太小，学习率加倍
```

#### 6. 学习率指标

**学习率 (Info/learning_rate)**
```
物理意义: 当前学习率
单位: 浮点数
范围: [0, initial_lr]
动态: 自适应调整
```

**解读**:
```python
# 自适应调整示例
1e-3 → 5e-4         # KL太大，减半
5e-4 → 7.5e-4       # KL太小，增加1.5倍
```

#### 7. 梯度指标

**梯度范数 (Grad/grad_norm)**
```
物理意义: 梯度大小
单位: 无量纲
范围: [0, max_grad_norm]
目标: 不超过max_grad_norm
```

**解读**:
```python
# 训练稳定的标志
grad_norm < 0.1     # 平稳更新
grad_norm ≈ 1.0     # 经常裁剪
grad_norm > 1.0     # 梯度爆炸风险
```

#### 8. 时间指标

**FPS (Info/fps)**
```
物理意义: 每秒环境步数
单位: step/s
范围: [1000, 10000]
```

**解读**:
```python
# 性能评估
fps > 5000          # GPU加速良好
fps ≈ 2000          # 正常
fps < 1000          # 可能瓶颈
```

**迭代时间 (Info/iteration_time)**
```
物理意义: 每次迭代耗时
单位: 秒
```

### 监控脚本

```python
# 启动TensorBoard
tensorboard --logdir=logs/rsl_rl/unitree_go2w_velocity_flat_v0

# 浏览器访问
http://localhost:6006
```

### 关键检查点

**检查点1: 1000迭代**
```
预期:
- total_reward > 0
- track_lin_vel_xy_exp > 1.0
- value_loss < 1.0
```

**检查点2: 5000迭代**
```
预期:
- total_reward > 2.0
- track_lin_vel_xy_exp > 2.0
- value_loss < 0.5
- entropy < 0.5
```

**检查点3: 10000迭代**
```
预期:
- total_reward > 3.0
- track_lin_vel_xy_exp > 2.5
- value_loss < 0.2
- entropy < 0.3
- 策略收敛
```

---

## 训练阶段与调优策略

### 训练阶段划分

#### 阶段1: 基础学习 (0-2000 iterations)

**目标**:
- 学习基本运动
- 保持直立
- 简单速度跟踪

**预期指标**:
```
total_reward: -5 → 0
track_lin_vel_xy_exp: 0 → 1.5
entropy: 1.0 → 0.8
value_loss: 5.0 → 1.0
```

**调整策略**:
```python
# 如果训练不稳定
learning_rate = 5e-4      # 减半
clip_param = 0.1          # 更保守

# 如果训练太慢
num_learning_epochs = 8   # 增加学习轮次
```

#### 阶段2: 技能提升 (2000-5000 iterations)

**目标**:
- 提高速度跟踪精度
- 学习转向
- 减少不必要的惩罚

**预期指标**:
```
total_reward: 0 → 2.0
track_lin_vel_xy_exp: 1.5 → 2.2
lin_vel_z_l2: -1.0 → -0.3
joint_torques_l2: -0.5 → -0.2
```

**调整策略**:
```python
# 启用课程学习
curriculum.command_levels = True

# 调整奖励权重
rewards.lin_vel_z_l2.weight = -3.0  # 更严格
```

#### 阶段3: 精细优化 (5000-10000 iterations)

**目标**:
- 最大化性能
- 平滑运动
- 能效优化

**预期指标**:
```
total_reward: 2.0 → 4.0
track_lin_vel_xy_exp: 2.2 → 2.8
action_rate_l2: -0.5 → -0.1
entropy: 0.5 → 0.2
```

**调整策略**:
```python
# 减少探索
entropy_coef = 0.005

# 增加平滑性
rewards.action_rate_l2.weight = -0.02
```

#### 阶段4: 鲁棒性增强 (10000+ iterations)

**目标**:
- 适应复杂地形
- 抗干扰能力
- 泛化性

**预期指标**:
```
total_reward: 4.0+ (稳定)
track_lin_vel_xy_exp: 2.8+
所有惩罚项收敛
```

**调整策略**:
```python
# 启用更多随机化
events.randomize_push_robot.interval_range_s = (5.0, 10.0)

# 启用复杂地形
curriculum.terrain_levels.max_level = 5
```

### 常见问题与解决

#### 问题1: 训练不稳定，奖励震荡

**症状**:
```
total_reward 剧烈波动
value_loss 不收敛
kl_divergence > 0.05
```

**解决方案**:
```python
# 1. 降低学习率
learning_rate = 5e-4

# 2. 减小裁剪范围
clip_param = 0.1

# 3. 增加batch size
num_steps_per_env = 32

# 4. 减少随机化强度
events.randomize_rigid_body_mass_base.mass_distribution_params = (-0.5, 1.5)
```

#### 问题2: 策略过早收敛

**症状**:
```
entropy < 0.01 (太快)
reward 不再增长
kl_divergence ≈ 0
```

**解决方案**:
```python
# 1. 增加熵系数
entropy_coef = 0.02

# 2. 增加学习率
learning_rate = 2e-3

# 3. 减少学习轮次
num_learning_epochs = 3

# 4. 增加探索噪声
policy.init_noise_std = 1.5
```

#### 问题3: 局部最优，性能差

**症状**:
```
total_reward < 0
track_lin_vel_xy_exp < 0.5
机器人只会站立，不会移动
```

**解决方案**:
```python
# 1. 检查奖励权重
rewards.track_lin_vel_xy_exp.weight = 5.0  # 临时增加

# 2. 降低惩罚权重
rewards.lin_vel_z_l2.weight = -0.5

# 3. 简化任务
curriculum.command_levels.range_multiplier = (0.05, 0.5)

# 4. 增加训练时间
max_iterations = 20000
```

#### 问题4: 内存不足

**症状**:
```
CUDA out of memory
```

**解决方案**:
```python
# 1. 减少环境数
num_envs = 4096 → 2048

# 2. 减少收集步数
num_steps_per_env = 24 → 16

# 3. 减小网络
policy.actor_hidden_dims = [256, 128]
policy.critic_hidden_dims = [256, 128]
```

### 超参数搜索

**网格搜索示例**:
```python
# 学习率搜索
learning_rates = [5e-4, 1e-3, 2e-3, 5e-3]

# 熵系数搜索
entropy_coefs = [0.005, 0.01, 0.02, 0.05]

# 网格搜索
for lr in learning_rates:
    for ec in entropy_coefs:
        训练并评估最佳配置
```

**贝叶斯优化**:
```python
# 使用Optuna等工具
import optuna

def objective(trial):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    ec = trial.suggest_float("entropy_coef", 1e-3, 1e-1, log=True)

    # 训练并返回指标
    return final_reward

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
```

---

## 总结

### GO2W训练的核心要点

1. **算法**: PPO with GAE，平衡探索与利用
2. **状态**: 41维观测，包含机器人状态和命令
3. **动作**: 16维混合控制（位置+速度）
4. **奖励**: 多目标加权，速度跟踪是核心
5. **课程**: 从简单到复杂，逐步增加难度
6. **随机化**: 多维度扰动，提高鲁棒性

### 物理意义总结

| 概念 | 物理意义 | 典型值 |
|------|----------|--------|
| **γ (gamma)** | 时间视野 | 0.99 → 7秒 |
| **λ (lam)** | 偏差-方差权衡 | 0.95 → 平衡 |
| **ε (clip)** | 更新限制 | 0.2 → 20% |
| **learning_rate** | 学习速度 | 1e-3 → 中等 |
| **entropy_coef** | 探索程度 | 0.01 → 适度 |
| **reward scale** | 目标重要性 | 3.0 → 速度跟踪 |

### 训练成功标志

```
✅ total_reward > 3.0
✅ track_lin_vel_xy_exp > 2.5
✅ lin_vel_z_l2 > -0.5
✅ value_loss < 0.2
✅ entropy ∈ [0.1, 0.3]
✅ kl_divergence ≈ 0.01
✅ 机器人能流畅移动
```

---

**文档版本**: 1.0
**最后更新**: 2026-03-08
**作者**: GO2W训练框架分析
