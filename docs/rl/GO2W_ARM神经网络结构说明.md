# GO2W_ARM 神经网络结构详细说明

## 📊 训练配置概览

- **算法**: PPO (Proximal Policy Optimization)
- **框架**: RSL-RL (Robotics System Learning - Reinforcement Learning)
- **环境**: Unitree-Go2WArm-TwoStage-Recovery-v0
- **环境数量**: 4096 (并行训练)
- **设备**: CUDA (GPU)

---

## 🧠 神经网络架构

### 整体架构

GO2W_ARM 使用经典的 **Actor-Critic** 架构，包含两个独立的神经网络：

```
输入观测空间 (观测向量)
    ↓
    ├─→ Actor Network (策略网络) → 动作均值 + 标准差 → 动作分布 → 采样动作
    └─→ Critic Network (价值网络) → 状态价值 V(s)
```

---

## 🔢 输入/输出维度

### 1. 观测空间 (Observation Space)

#### Policy 观测组成（策略网络输入）：

| 观测项 | 维度 | 说明 | Scale |
|--------|------|------|-------|
| **base_ang_vel** | 3 | 基座角速度 (roll, pitch, yaw) | 0.25 |
| **projected_gravity** | 3 | 重力投影向量 (x, y, z) | 1.0 |
| **velocity_commands** | 3 | 速度命令 (vx, vy, wz) | 1.0 |
| **joint_pos** | 18 | 关节位置（相对位置） | - |
| **joint_vel** | 18 | 关节速度（相对速度） | - |
| **last_action** | 18 | 上一步动作 | - |

**总计**: 3 + 3 + 3 + 18 + 18 + 18 = **63 维**

#### 详细说明：

**1. base_ang_vel (3维)**
- 维度: `[roll_rate, pitch_rate, yaw_rate]`
- 物理意义: 基座相对于世界的角速度
- 缩放: 0.25
- 噪声: Uniform(-0.2, 0.2)

**2. projected_gravity (3维)**
- 维度: `[gx, gy, gz]`
- 物理意义: 重力向量在基座坐标系中的投影
  - gz ≈ 1.0 表示直立
  - gz ≈ 0.0 表示倒下
- 缩放: 1.0
- 噪声: Uniform(-0.05, 0.05)

**3. velocity_commands (3维)**
- 维度: `[cmd_vx, cmd_vy, cmd_wz]`
- 物理意义: 目标线速度 (x, y) 和角速度 (z)
- 缩放: 1.0
- 无噪声

**4. joint_pos (18维)**
- 维度: `[12个腿部关节 + 6个机械臂关节]`
- 腿部关节 (12维):
  - FR/FL/RR/RL × (hip, thigh, calf)
- 机械臂关节 (6维):
  - arm_joint1 到 arm_joint6
- 相对位置: 关节当前角度 - 默认角度
- 缩放: 无（原始值）

**5. joint_vel (18维)**
- 维度: 同 joint_pos
- 物理意义: 关节角速度
- 相对速度: 关节当前速度
- 缩放: 无（原始值）

**6. last_action (18维)**
- 维度: 同 joint_pos
- 物理意义: 上一步的动作（位置或速度命令）
- 缩放: 无（原始值）

---

### 2. 动作空间 (Action Space)

#### 动作组成：

| 动作项 | 关节数 | 控制方式 | 维度 | Scale |
|--------|--------|----------|------|-------|
| **joint_pos** | 18 | 位置控制 | 18 | 不同关节不同scale |
| **joint_vel** | 4 | 速度控制 | 4 | 5.0 |

**总计**: 18 + 4 = **22 维**

#### 详细说明：

**1. joint_pos (18维) - 关节位置控制**

关节列表 (18个):
- 腿部关节 (12个):
  - FR_hip_joint, FR_thigh_joint, FR_calf_joint
  - FL_hip_joint, FL_thigh_joint, FL_calf_joint
  - RR_hip_joint, RR_thigh_joint, RR_calf_joint
  - RL_hip_joint, RL_thigh_joint, RL_calf_joint
- 机械臂关节 (6个):
  - arm_joint1, arm_joint2, arm_joint3, arm_joint4, arm_joint5, arm_joint6

Scale 配置:
```yaml
.*_hip_joint: 0.125      # 髋关节缩放较小
arm_joint1: 0.5         # 机械臂第一个关节缩放较大
其他关节: 0.25           # 其他关节中等缩放
```

**2. joint_vel (4维) - 轮子速度控制**

关节列表 (4个):
- FR_foot_joint, FL_foot_joint
- RR_foot_joint, RL_foot_joint

Scale 配置:
```yaml
所有轮子关节: 5.0
```

---

## 🏗️ 网络结构详细说明

### Actor Network (策略网络)

**配置**:
```yaml
class_name: MLPModel
hidden_dims: [512, 256, 128]
activation: elu
obs_normalization: false
distribution_cfg:
  class_name: GaussianDistribution
  init_std: 1.0
  std_type: scalar
```

**网络结构**:

```
输入: [batch_size, 63]  (观测向量)
    ↓
Linear(in=63, out=512) + ELU
    ↓
Linear(in=512, out=256) + ELU
    ↓
Linear(in=256, out=128) + ELU
    ↓
Linear(in=128, out=22)  (动作均值 μ)
    ↓
输出动作均值 μ: [batch_size, 22]
```

**动作分布**:
- 使用 **高斯分布 (GaussianDistribution)**
- 初始标准差: 1.0
- 标准差类型: 标量 (所有动作共享同一个标准差)
- 采样: action = μ + σ × ε, where ε ~ N(0, 1)

**各层维度**:

| 层 | 输入维度 | 输出维度 | 参数量 |
|----|----------|----------|--------|
| Input | - | 63 | 0 |
| FC1 | 63 | 512 | 63×512 + 512 = 32,768 |
| FC2 | 512 | 256 | 512×256 + 256 = 131,328 |
| FC3 | 256 | 128 | 256×128 + 128 = 32,896 |
| Output | 128 | 22 | 128×22 + 22 = 2,838 |
| **总计** | - | - | **199,830** |

**实际训练时的维度**:

```
输入: [4096, 63]           # (num_envs, obs_dim)
    ↓
FC1:  [4096, 512]
    ↓
FC2:  [4096, 256]
    ↓
FC3:  [4096, 128]
    ↓
输出: [4096, 22]           # (num_envs, action_dim)
```

---

### Critic Network (价值网络)

**配置**:
```yaml
class_name: MLPModel
hidden_dims: [512, 256, 128]
activation: elu
obs_normalization: false
distribution_cfg: null
```

**网络结构**:

```
输入: [batch_size, 63]  (观测向量)
    ↓
Linear(in=63, out=512) + ELU
    ↓
Linear(in=512, out=256) + ELU
    ↓
Linear(in=256, out=128) + ELU
    ↓
Linear(in=128, out=1)   (状态价值 V(s))
    ↓
输出状态价值: [batch_size, 1]
```

**各层维度**:

| 层 | 输入维度 | 输出维度 | 参数量 |
|----|----------|----------|--------|
| Input | - | 63 | 0 |
| FC1 | 63 | 512 | 63×512 + 512 = 32,768 |
| FC2 | 512 | 256 | 512×256 + 256 = 131,328 |
| FC3 | 256 | 128 | 256×128 + 128 = 32,896 |
| Output | 128 | 1 | 128×1 + 1 = 129 |
| **总计** | - | - | **197,121** |

**实际训练时的维度**:

```
输入: [4096, 63]           # (num_envs, obs_dim)
    ↓
FC1:  [4096, 512]
    ↓
FC2:  [4096, 256]
    ↓
FC3:  [4096, 128]
    ↓
输出: [4096, 1]            # (num_envs, 1) - 每个环境的V值
```

---

## 🎯 PPO 算法参数

### 核心参数

| 参数 | 值 | 说明 |
|------|-----|------|
| **class_name** | PPO | 算法类型 |
| **num_learning_epochs** | 5 | 每次收集数据后更新网络5次 |
| **num_mini_batches** | 4 | 将数据分成4个mini-batch |
| **learning_rate** | 0.0002 | 学习率 |
| **schedule** | adaptive | 自适应学习率调度 |
| **gamma** | 0.99 | 折扣因子 |
| **lam** | 0.95 | GAE (Generalized Advantage Estimation) lambda |
| **entropy_coef** | 0.01 | 熵正则化系数 |
| **desired_kl** | 0.01 | 目标KL散度 |
| **max_grad_norm** | 1.0 | 梯度裁剪阈值 |
| **optimizer** | adam | 优化器 |
| **value_loss_coef** | 1.0 | 价值损失系数 |
| **use_clipped_value_loss** | true | 使用裁剪价值损失 |
| **clip_param** | 0.2 | PPO裁剪参数 |

### 训练参数

| 参数 | 值 | 说明 |
|------|-----|------|
| **num_steps_per_env** | 24 | 每个环境收集24步 |
| **max_iterations** | 1,000,000 | 最大训练迭代次数 |
| **save_interval** | 100 | 每100个迭代保存一次 |
| **empirical_normalization** | false | 不使用经验归一化 |
| **clip_actions** | null | 不裁剪动作 |

---

## 📊 训练流程中的Tensor维度

### 1. 数据收集阶段

```
# 初始状态
observations:    [4096, 63]     # 4096个环境，每个63维观测
actions:         [4096, 22]     # 4096个环境，每个22维动作
rewards:         [4096, 1]      # 4096个环境，每个1维奖励
dones:           [4096, 1]      # 4096个环境，每个1维终止标志
values:          [4096, 1]      # 4096个环境，每个1维价值

# 收集24步后
observations:    [24, 4096, 63]  # (steps, envs, obs_dim)
actions:         [24, 4096, 22]  # (steps, envs, action_dim)
rewards:         [24, 4096, 1]   # (steps, envs, 1)
dones:           [24, 4096, 1]   # (steps, envs, 1)
values:          [24, 4096, 1]   # (steps, envs, 1)
```

### 2. 优势估计 (GAE)

```
# 计算优势
advantages:      [24, 4096, 1]   # (steps, envs, 1)
returns:         [24, 4096, 1]   # (steps, envs, 1)
```

### 3. Mini-batch 训练

```
# 展平数据
observations_flat:  [98304, 63]   # (24×4096, obs_dim)
actions_flat:       [98304, 22]   # (24×4096, action_dim)
advantages_flat:    [98304, 1]    # (24×4096, 1)
returns_flat:       [98304, 1]    # (24×4096, 1)

# 分成4个mini-batch
mini_batch_size:    24576          # 98304 / 4
observations_batch: [24576, 63]   # (mini_batch_size, obs_dim)
actions_batch:      [24576, 22]   # (mini_batch_size, action_dim)
advantages_batch:   [24576, 1]    # (mini_batch_size, 1)
returns_batch:      [24576, 1]    # (mini_batch_size, 1)

# Actor前向传播
actor_output_mean: [24576, 22]   # (mini_batch_size, action_dim)
actor_log_prob:    [24576, 22]   # (mini_batch_size, action_dim)

# Critic前向传播
critic_output:      [24576, 1]    # (mini_batch_size, 1)

# 更新5个epoch
for epoch in range(5):
    for mini_batch in 4个mini-batch:
        # 计算损失并更新
        policy_loss:      [1]      # 标量
        value_loss:       [1]      # 标量
        entropy_loss:     [1]      # 标量
        total_loss:       [1]      # 标量
```

---

## 🎨 网络可视化

### Actor Network (策略网络)

```
┌─────────────────────────────────────────────────────────┐
│                   INPUT LAYER                           │
│  [batch_size, 63]                                       │
│  (观测向量: 角速度+重力+命令+关节位置+关节速度+上一步动作) │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                    FC1 (Linear)                         │
│  in_features: 63, out_features: 512                     │
│  Parameters: 63×512 + 512 = 32,768                     │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                    ELU Activation                       │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                    FC2 (Linear)                         │
│  in_features: 512, out_features: 256                    │
│  Parameters: 512×256 + 256 = 131,328                   │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                    ELU Activation                       │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                    FC3 (Linear)                         │
│  in_features: 256, out_features: 128                    │
│  Parameters: 256×128 + 128 = 32,896                    │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                    ELU Activation                       │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                    FC4 (Linear)                         │
│  in_features: 128, out_features: 22                     │
│  Parameters: 128×22 + 22 = 2,838                       │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                   OUTPUT LAYER                           │
│  [batch_size, 22]                                       │
│  (动作均值 μ)                                           │
│                                                         │
│  高斯分布:                                              │
│    action = μ + σ × ε, where ε ~ N(0, 1)               │
│    σ = 1.0 (初始标准差)                                 │
└─────────────────────────────────────────────────────────┘

总参数量: 199,830
```

### Critic Network (价值网络)

```
┌─────────────────────────────────────────────────────────┐
│                   INPUT LAYER                           │
│  [batch_size, 63]                                       │
│  (观测向量)                                             │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                    FC1 (Linear)                         │
│  in_features: 63, out_features: 512                     │
│  Parameters: 63×512 + 512 = 32,768                     │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                    ELU Activation                       │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                    FC2 (Linear)                         │
│  in_features: 512, out_features: 256                    │
│  Parameters: 512×256 + 256 = 131,328                   │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                    ELU Activation                       │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                    FC3 (Linear)                         │
│  in_features: 256, out_features: 128                    │
│  Parameters: 256×128 + 128 = 32,896                    │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                    ELU Activation                       │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                    FC4 (Linear)                         │
│  in_features: 128, out_features: 1                      │
│  Parameters: 128×1 + 1 = 129                           │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                   OUTPUT LAYER                           │
│  [batch_size, 1]                                        │
│  (状态价值 V(s))                                        │
└─────────────────────────────────────────────────────────┘

总参数量: 197,121
```

---

## 📈 总参数量统计

| 网络 | 参数量 | 占比 |
|------|--------|------|
| Actor Network | 199,830 | 50.4% |
| Critic Network | 197,121 | 49.6% |
| **总计** | **396,951** | 100% |

---

## 🔧 关键设计特点

### 1. 输入观测
- **丰富性**: 包含机器人状态的完整信息（位置、速度、命令）
- **归一化**: 部分观测经过缩放，便于网络学习
- **噪声**: 部分观测添加噪声，提高鲁棒性
- **历史**: 包含上一步动作，提供时间上下文

### 2. 网络结构
- **深度适中**: 3层隐藏层，平衡表达能力和训练效率
- **宽度递减**: 512 → 256 → 128，逐步压缩特征
- **激活函数**: ELU (Exponential Linear Unit)，缓解梯度消失

### 3. 输出动作
- **混合控制**: 关节位置控制 + 轮子速度控制
- **差异化缩放**: 不同关节使用不同的action scale
- **随机策略**: 使用高斯分布，保留探索性

### 4. 训练策略
- **并行训练**: 4096个环境并行，提高样本效率
- **PPO算法**: 稳定且高效
- **自适应学习率**: 根据KL散度调整学习率
- **GAE估计**: 使用GAE计算优势，减少方差

---

## 📚 参考配置文件

训练配置保存在:
- Agent配置: `logs/rsl_rl/unitree_go2warm_twostage_recovery_v0/*/params/agent.yaml`
- 环境配置: `logs/rsl_rl/unitree_go2warm_twostage_recovery_v0/*/params/env.yaml`

---

## 🔍 如何查看实际维度

在训练代码中添加调试代码:

```python
# 在 train_fixed.py 中添加
from isaaclab.utils.math import print_tensor_shapes

# 获取观测空间
obs, _ = env.reset()
print(f"Observation shape: {obs.shape}")  # 应该是 (4096, 63)

# 获取动作空间
action = env.action_space.sample()
print(f"Action shape: {action.shape}")    # 应该是 (4096, 22)

# 查看网络参数
print(f"Actor parameters: {sum(p.numel() for p in runner.algo.actor_critic.actor.parameters())}")
print(f"Critic parameters: {sum(p.numel() for p in runner.algo.actor_critic.critic.parameters())}")
```

---

## 📅 生成时间

2026-04-05
