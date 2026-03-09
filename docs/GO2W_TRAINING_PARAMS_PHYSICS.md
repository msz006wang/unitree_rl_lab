# GO2W 训练参数和Loss函数物理意义详解

## 一、PPO算法核心Loss函数

### 1.1 总体Loss函数

PPO使用的总loss函数是多个loss的加权和：

```
Total Loss = Policy Loss + Value Loss + Entropy Loss
```

#### **配置位置**
[velocity_env_cfg.py:23-36](source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2w/velocity_env_cfg.py#L23-L36)

```python
algorithm = RslRlPpoAlgorithmCfg(
    value_loss_coef=1.0,              # 值函数loss系数
    use_clipped_value_loss=True,       # 使用裁剪值函数loss
    clip_param=0.2,                    # PPO裁剪参数
    entropy_coef=0.01,                 # 熵正则化系数
    num_learning_epochs=5,             # 每次更新的学习轮数
    num_mini_batches=4,                 # 小批次数量
    learning_rate=1.0e-3,               # 学习率
    schedule="adaptive",                # 学习率调度
    gamma=0.99,                        # 折扣因子
    lam=0.95,                          # GAE参数
    desired_kl=0.01,                    # 目标KL散度
    max_grad_norm=1.0,                  # 梯度裁剪
)
```

---

## 二、核心Loss函数详解

### 2.1 Policy Loss（策略Loss）

**公式**：
```
L_CLIP(θ) = -min(ratio(θ) * A, clip(ratio(θ), 1-ε, 1+ε) * A)
```

其中：
- `ratio(θ) = π_new(a|s) / π_old(a|s)` - 新旧策略概率比
- `A` - 优势函数（Advantage）
- `ε = 0.2` - clip_param（裁剪范围）

**物理意义**：
1. **优化策略**：增加采取好动作的概率，减少坏动作的概率
2. **防止过大更新**：通过裁剪限制策略更新的幅度
3. **保守改进**：确保新策略不会比旧策略差太多

**实际效果**：
- 机器人学会选择能够获得高奖励的动作
- 避免策略崩溃（突然性能大幅下降）
- 稳定的学习过程

### 2.2 Value Loss（值函数Loss）

**公式**：
```
L_VF(θ) = (V_θ(s) - R_t)²
```

其中：
- `V_θ(s)` - 预测的状态价值
- `R_t` - 实际回报（Return）

**物理意义**：
1. **价值估计**：学习预测每个状态的未来总奖励
2. **辅助策略学习**：更准确的价值估计帮助计算更好的优势函数
3. **基线**：作为baseline，减少策略梯度的方差

**实际效果**：
- 机器人知道"我现在状态好坏"（例如：快要倒了 = 价值低）
- 价值估计帮助判断是否应该继续当前动作
- 减少训练方差，加速收敛

### 2.3 Entropy Loss（熵Loss）

**公式**：
```
L_entropy(θ) = -β * Σ π(a|s) * log(π(a|s))
```

其中：
- `π(a|s)` - 动作概率分布
- `β = 0.01` - entropy_coef（熵系数）

**物理意义**：
1. **鼓励探索**：惩罚过于确定的策略，保持动作多样性
2. **防止局部最优**：避免过早收敛到次优策略
3. **保持随机性**：允许尝试新动作，可能发现更好的策略

**实际效果**：
- 机器人不会过早固定某些动作
- 保持动作的多样性和探索性
- 训练初期更多探索，后期更多利用

---

## 三、GO2W奖励函数详解

### 3.1 主要奖励函数

#### **1. 线速度跟踪奖励** (最高优先级)
**配置**: [velocity_env_cfg.py:439-441](source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2w/velocity_env_cfg.py#L439-L441)

```python
track_lin_vel_xy_exp = RewTerm(
    func=mdp.track_lin_vel_xy_exp,
    weight=3.0,  # ⭐ 最高权重
    params={
        "command_name": "base_velocity",
        "std": math.sqrt(0.25)  # 高斯核标准差
    }
)
```

**公式**：
```
reward = exp(-||v_command - v_actual||² / std²)
```

**物理意义**：
- 鼓励机器人按照命令速度移动（xy平面）
- 3.0的权重表示这是**最重要**的任务
- 使用指数核函数，误差越小奖励越高

**实际效果**：
- 机器人学会前进、后退、左移、右移
- 速度跟踪精度直接决定主要奖励

#### **2. 角速度跟踪奖励**
**配置**: [velocity_env_cfg.py:442-444](source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2w/velocity_env_cfg.py#L442-L444)

```python
track_ang_vel_z_exp = RewTerm(
    func=mdp.track_ang_vel_z_exp,
    weight=1.5,  # 第二高权重
    params={
        "command_name": "base_velocity",
        "std": math.sqrt(0.25)
    }
)
```

**物理意义**：
- 鼓励机器人按照命令旋转（yaw轴）
- 1.5的权重表示重要性次于线速度
- 旋转能力对姿态调整很重要

**实际效果**：
- 机器人学会转向、旋转
- 配合线速度实现灵活移动

### 3.2 稳定性奖励函数

#### **1. Z轴速度惩罚**
```python
lin_vel_z_l2 = RewTerm(
    func=mdp.lin_vel_z_l2,
    weight=-2.0
)
```

**物理意义**：
- **惩罚垂直运动**（跳跃、跌落）
- 鼓励机器人保持稳定的接触地面
- -2.0权重表示这是重要的约束

**实际效果**：
- 避免不必要的跳跃
- 保持平稳运动
- 减少冲击和损坏

#### **2. 向上姿态奖励**
```python
upward = RewTerm(
    func=mdp.upward,
    weight=1.0
)
```

**物理意义**：
- 鼓励机器人保持直立姿态
- 基于重力向量投影计算
- 倒立时奖励为0，直立时奖励为1

**实际效果**：
- 防止机器人倾倒
- 自动恢复平衡
- 提高运动稳定性

#### **3. XY轴角速度惩罚**
```python
ang_vel_xy_l2 = RewTerm(
    func=mdp.ang_vel_xy_l2,
    weight=-0.05
)
```

**物理意义**：
- 惩罚翻滚运动（pitch和roll）
- 防止过度倾斜
- 权重较小，只严重惩罚

**实际效果**：
- 限制机器人的倾斜角度
- 防止侧翻
- 保持水平姿态

### 3.3 关节相关奖励

#### **1. 关节力矩惩罚**
```python
joint_torques_l2 = RewTerm(
    func=mdp.joint_torques_l2,
    weight=-2.5e-5  # 非常小的权重
)
```

**物理意义**：
- 惩罚过大的关节力矩
- 鼓励节能运动
- 防止电机过载

**公式**：
```
penalty = Σ τ²
```

**实际效果**：
- 使用平滑、高效的动作
- 延长电池寿命（真实机器人）
- 减少能量消耗

#### **2. 关节加速度惩罚**
```python
joint_acc_l2 = RewTerm(
    func=mdp.joint_acc_l2,
    weight=-2.5e-7  # 极小的权重
)
```

**物理意义**：
- 惩罚过快的加速度变化
- 鼓励平滑运动
- 保护机械结构

**实际效果**：
- 运动更平滑自然
- 减少机械冲击
- 提高运动质量

#### **3. 关节功率惩罚**
```python
joint_power = RewTerm(
    func=mdp.joint_power,
    weight=-2e-5
)
```

**物理意义**：
- 惩罚高功率消耗
- 功率 = 力矩 × 速度
- 直接考虑能量效率

**公式**：
```
penalty = Σ |τ × ω|
```

**实际效果**：
- 能量最优的运动方式
- 平滑协调的关节运动
- 节能策略

#### **4. 关节位置偏差惩罚**
```python
joint_pos_penalty = RewTerm(
    func=mdp.joint_pos_penalty,
    weight=-1.0
)
```

**物理意义**：
- 鼓励关节保持在默认位置附近
- 运动时：正常惩罚
- 静止时：5倍惩罚（强迫回到站立姿态）

**实际效果**：
- 停下时自动回到站立姿态
- 运动时允许偏离但不过度
- 自然的零位恢复行为

### 3.4 动作平滑奖励

#### **1. 动作变化率惩罚**
```python
action_rate_l2 = RewTerm(
    func=mdp.action_rate_l2,
    weight=-0.01
)
```

**物理意义**：
- 惩罚动作的剧烈变化
- 鼓励平滑的控制信号
- 防止抖动

**公式**：
```
penalty = ||a_t - a_{t-1}||²
```

**实际效果**：
- 机器人运动更流畅
- 减少机械震动
- 提高控制品质

#### **2. 关节镜像对称奖励**（新增）
```python
action_mirror = RewTerm(
    func=mdp.action_mirror,
    weight=0.0  # 初始禁用，可启用
)
```

**物理意义**：
- 鼓励左右对称关节采取相似动作
- 产生自然、协调的步态
- 简化策略学习难度

**实际效果**：
- 对称的运动模式
- 更自然的步态
- 减少不必要的扭转

#### **3. 动作同步奖励**（新增）
```python
action_sync = RewTerm(
    func=mdp.action_sync,
    weight=0.0  # 初始禁用，可启用
)
```

**物理意义**：
- 鼓励同类型关节同步运动
- 提高运动协调性
- 增强稳定性

**实际效果**：
- 四条腿协调运动
- 一致的步态节奏
- 更稳定的运动

### 3.5 接触相关奖励

#### **1. 不当接触惩罚**
```python
undesired_contacts = RewTerm(
    func=mdp.undesired_contacts,
    weight=-1.0
)
```

**物理意义**：
- 惩罚非足部身体接触地面
- 例如：膝盖、手臂、身体躯干
- 防止不自然的姿态

**实际效果**：
- 只用脚接触地面
- 避免趴在地上
- 保持直立行走

#### **2. 接触力惩罚**
```python
contact_forces = RewTerm(
    func=mdp.contact_forces,
    weight=-1.5e-4  # 很小的权重
)
```

**物理意义**：
- 惩罚过大的足部接触力
- 保护足部结构和地面
- 鼓励轻盈的步态

**实际效果**：
- 减少冲击力
- 更柔和的落地
- 延长硬件寿命

### 3.6 其他奖励

#### **1. 足部接触奖励**
```python
feet_contact_without_cmd = RewTerm(
    func=mdp.feet_contact_without_cmd,
    weight=0.1
)
```

**物理意义**：
- 停止命令时，鼓励所有足部接触地面
- 提供稳定的支撑
- 自然地站立姿态

**实际效果**：
- 停下时稳定站立
- 防止单腿站立
- 平稳的静止状态

---

## 四、训练超参数物理意义

### 4.1 基础训练参数

| 参数 | 值 | 物理意义 |
|------|-----|----------|
| `num_envs` | 4096 | 并行环境数量 |
| | | - 更多环境 → 更多样本 → 更快学习 |
| | | - 4096是常用平衡点（速度 vs 内存） |
| `episode_length_s` | 20.0 | 每个回合时长（秒） |
| | | - 太短：无法完成复杂任务 |
| | | - 太长：credit assignment困难 |
| `decimation` | 4 | 控制频率倍数 |
| | | - 物理更新：200Hz，控制：50Hz |
| | | - 平衡实时性和计算成本 |
| `dt` | 0.005 | 物理时间步长（秒） |
| | | - 0.005s = 200Hz物理频率 |
| | | - 足小越精确但计算慢 |
| `max_iterations` | 10000 | 最大训练迭代次数 |
| | | - 10000次约需8-12小时（取决于硬件） |
| | | - 通常5000-15000次收敛 |

### 4.2 PPO算法参数

| 参数 | 值 | 物理意义 |
|------|-----|----------|
| `learning_rate` | 1.0e-3 | 学习率 |
| | | - 控制参数更新幅度 |
| | | - 太大：不稳定；太小：学习慢 |
| | | - 1e-3是常用起点，自适应调整 |
| `clip_param` | 0.2 | PPO裁剪范围 |
| | | - 限制策略更新幅度 |
| | | - 确保守改进，防止崩溃 |
| | | - 0.2表示最多偏离20% |
| `gamma` | 0.99 | 折扣因子 |
| | | - 未来奖励的重要性 |
| | | - 0.99重视长远奖励 |
| | | - γ^t ≈ 0.99^100 = 0.37（100步后） |
| `lam` (λ) | 0.95 | GAE（广义优势估计）参数 |
| | | - 平衡偏差和方差 |
| | | - 接近1：低偏差高方差 |
| | | - 接近0：高偏差低方差 |
| | | - 0.95是良好平衡 |
| `entropy_coef` | 0.01 | 熵正则化系数 |
| | | - 探索强度 |
| | | - 太大：纯随机；太小：不探索 |
| | | - 通常随训练逐渐减小 |
| `num_learning_epochs` | 5 | 每次更新的学习轮数 |
| | | - 使用每批数据更新5次 |
| | | - 充分利用收集的数据 |
| `num_mini_batches` | 4 | 小批次数量 |
| | | - 将4096个环境分成4批更新 |
| | | - 每批1024个环境 |
| `desired_kl` | 0.01 | 目标KL散度 |
| | | - 策略更新幅度指标 |
| | | - 自适应调整学习率 |
| | | - KL > 0.01：降低学习率 |
| `max_grad_norm` | 1.0 | 梯度裁剪 |
| | | - 防止梯度爆炸 |
| | | - 稳定训练过程 |

### 4.3 网络结构参数

```python
policy = RslRlPpoActorCriticCfg(
    init_noise_std=1.0,        # 初始化噪声标准差
    actor_hidden_dims=[512, 256, 128],  # 策略网络隐藏层
    critic_hidden_dims=[512, 256, 128], # 价值网络隐藏层
    activation="elu",           # ELU激活函数
)
```

**物理意义**：
- **512-256-128三层结构**：
  - 第一层（512）：提取高级特征
  - 第二层（256）：中层抽象
  - 第三层（128）：低级控制信号
- **ELU激活**：
  - 比ReLU更平滑
  - 有负值输出（某些任务有用）
  - 计算效率高
- **初始化噪声（1.0）**：
  - 随机初始化网络权重
  - 打破对称性，促进探索
  - 标准差1.0表示相当大的随机性

---

## 五、观测空间物理意义

### 5.1 策略网络观测（16维）

```python
# 1. 基座角速度 (3维)
base_ang_vel = [ω_x, ω_y, ω_z]
# 物理意义：机器人旋转速度
# 单位：rad/s
```

```python
# 2. 投影重力 (3维)
projected_gravity = [g_x, g_y, g_z]
# 物理意义：重力向量在机体坐标系中的投影
# 直立时 ≈ [0, 0, -1]，倒下时会偏离
# 用于姿态估计和平衡控制
```

```python
# 3. 速度命令 (3维)
velocity_commands = [v_x_cmd, v_y_cmd, ω_z_cmd]
# 物理意义：期望的运动速度
# 范围：[-1.0, 1.0] m/s (线速度)，[-1.0, 1.0] rad/s (角速度)
```

```python
# 4. 腿关节位置 (12维)
joint_pos = [θ_FR_hip, θ_FR_thigh, θ_FR_calf,
           θ_FL_hip, θ_FL_thigh, θ_FL_calf,
           θ_RR_hip, θ_RR_thigh, θ_RR_calf,
           θ_RL_hip, θ_RL_thigh, θ_RL_calf]
# 物理意义：腿部关节相对于默认位置的偏差
# 单位：rad（弧度）
```

```python
# 5. 轮关节速度 (4维)
joint_vel = [ω_FR, ω_FL, ω_RR, ω_RL]
# 物理理意义：四个轮子的角速度
# 单位：rad/s
```

```python
# 6. 上一步动作 (16维)
last_action = [a_legs_12, a_wheels_4]
# 物理意义：上一时刻的控制信号
# 用于动作平滑和时间相关性
```

**总维度**: 3+3+3+12+4+16 = **41维**（展开后）

### 5.2 为什么没有base_lin_vel？

**设计决策**：
```python
self.observations.policy.base_lin_vel = None  # 已禁用
```

**物理意义**：
1. **鲁棒性**：不依赖线速度测量，防止传感器噪声
2. **Sim-to-Real**：真实机器人速度传感器可能有漂移
3. **泛化能力**：强迫策略从其他状态推断速度
4. **简化**：减少观测维度，加快学习

**替代方案**：
- 通过关节位置变化推断速度
- 通过角速度和命令推断
- 学习内在的速度表示

---

## 六、动作空间物理意义

### 6.1 混合动作空间（16维）

#### **腿关节：位置控制** (12维)
```python
joint_pos = mdp.JointPositionActionCfg(
    scale={
        ".*_hip_joint": 0.125,    # 髋关节：小范围（±0.125 rad）
        "^(?!.*_hip_joint).*": 0.25  # 其他关节：大范围（±0.25 rad）
    },
    use_default_offset=True  # 增量控制
)
```

**物理意义**：
- **位置控制**：直接控制关节目标角度
- **增量模式**：动作 = 默认位置 + (网络输出 × scale)
- **髋关节小范围**：髋关节是主要的承重关节，需要精确控制
- **其他关节大范围**：大腿和小腿可以有更大的运动范围

**实际效果**：
- 精确控制腿部姿态
- 保持自然的关节角度
- 避免极限位置

#### **轮子：速度控制** (4维)
```python
joint_vel = mdp.JointVelocityActionCfg(
    scale=5.0  # 速度缩放因子
)
```

**物理意义**：
- **速度控制**：控制轮子转动速度
- **增量模式**：速度 = 默认速度 + (网络输出 × 5.0)
- **scale=5.0**：允许±5 rad/s的速度变化

**实际效果**：
- 快速响应的速度控制
- 适应不同地形
- 高效的推进方式

### 6.2 混合控制的优势

| 控制方式 | 腿关节（位置） | 轮子（速度） |
|---------|---------------|-------------|
| **目的** | 姿态控制 | 推进控制 |
| **响应速度** | 较慢（物理约束） | 快（直接驱动） |
| **精度** | 高（位置精确） | 中（速度积分） |
| **能耗** | 中（保持姿态） | 低（直接驱动） |
| **适用** | 复杂地形、精确控制 | 平地、高效推进 |

**配合策略**：
- 腿：负责姿态调整、平衡、跨越障碍
- 轮：负责主要推进、速度控制

---

## 七、课程学习物理意义

### 7.1 地形课程学习

**配置**: [velocity_env_cfg.py:543](source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2w/velocity_env_cfg.py#L543)

```python
terrain_levels = CurrTerm(
    func=mdp.terrain_levels_vel
)
```

**工作机制**：
1. 初始：所有环境在最简单地形（level 0）
2. 条件：机器人行走距离 > 地形半宽 → 升级
3. 条件：行走距离 < 命令距离的一半 → 降级
4. 结果：逐渐适应更难地形

**物理意义**：
- **循序渐进**：从简单到复杂
- **自适应难度**：根据表现调整
- **防止挫败**：太难时自动降低难度
- **最大化学习**：始终在合适的难度

### 7.2 命令速度课程学习

**配置**: [velocity_env_cfg.py:544-550](source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2w/velocity_env_cfg.py#L544-L550)

```python
command_levels = CurrTerm(
    func=mdp.command_levels_vel,
    params={
        "reward_term_name": "track_lin_vel_xy_exp",
        "range_multiplier": (0.1, 1.0),
    },
)
```

**工作机制**：
1. 初始：10%速度范围（[-0.1, 0.1] m/s）
2. 条件：跟踪奖励 > 80% → 增加0.1 m/s
3. 上限：100%速度范围（[-1.0, 1.0] m/s
4. 结果：逐渐学会更快速度

**物理意义**：
- **速度分级学习**：慢速→中速→快速
- **稳定性优先**：在当前速度学好后再提速
- **防止过早失败**：太快时控制困难会失败
- **最大化性能**：最终达到全速度范围

---

## 八、事件随机化物理意义

### 8.1 启动时随机化

#### **质量随机化**
```python
randomize_rigid_body_mass_base:
    mass_distribution_params=(-1.0, 3.0)  # -1kg到+3kg
    operation="add"
```
**物理意义**：
- 模拟不同负载（例如：携带物品）
- 增强策略对质量变化的鲁棒性
- 测试在不同质量下的平衡能力

#### **质心随机化**
```python
randomize_com_positions:
    com_range={"x": (-0.05, 0.05), ...}  # ±5cm
```
**物理意义**：
- 模拟质心偏移（例如：安装了设备）
- 测试平衡适应能力
- 增强姿态控制鲁棒性

#### **转动惯量随机化**（新增）
```python
randomize_rigid_body_inertia:
    inertia_distribution_params=(0.8, 1.2)  # 80%-120%
    operation="scale"
```
**物理意义**：
- 模拟不同转动惯量
- 测试旋转适应能力
- 增强Sim-to-Real迁移能力

### 8.2 重置时随机化

#### **外部扰动**
```python
randomize_apply_external_force_torque:
    force_range=(-10.0, 10.0) N    # ±10N力
    torque_range=(-10.0, 10.0) N·m  # ±10N·m力矩
```
**物理意义**：
- 模拟外部冲击
- 测试平衡恢复能力
- 增强抗干扰能力

---

## 九、训练监控指标

### 9.1 TensorBoard关键指标

```bash
tensorboard --logdir logs/rsl_rl/
```

#### **主要指标**

| 指标 | 物理意义 | 期望趋势 |
|------|---------|---------|
| `ep_rew_mean` | 平均回合奖励 | ↗️ 上升 |
| `policy_loss` | 策略Loss | 先↓后稳定 |
| `value_loss` | 值函数Loss | ↓↓ 下降 |
| `ratio` | PPO概率比 | 接近1.0 |
| `kl` | KL散度 | 0.01左右 |

#### **辅助指标**

| 指标 | 物理意义 |
|------|---------|
| `track_lin_vel_xy_exp` | 线速度跟踪奖励 |
| `track_ang_vel_z_exp` | 角速度跟踪奖励 |
| `upward` | 向上姿态奖励 |
| `lin_vel_z_l2` | Z轴速度惩罚（越小越好）|
| `joint_torques_l2` | 力矩惩罚 |
| `action_rate_l2` | 动作平滑度 |

---

## 十、实际训练建议

### 10.1 训练阶段划分

**阶段1：基础学习** (0-2000 iterations)
- 目标：学习基本运动技能
- 重点关注：`track_lin_vel_xy_exp`, `track_ang_vel_z_exp`
- 预期：奖励快速上升

**阶段2：精细调整** (2000-5000 iterations)
- 目标：提高稳定性和效率
- 重点关注：`upward`, `joint_torques_l2`, `action_rate_l2`
- 预期：奖励缓慢上升，趋于稳定

**阶段3：优化提升** (5000-10000 iterations)
- 目标：最大化性能
- 重点关注：所有指标的综合优化
- 预期：奖励达到平台期

### 10.2 常见问题诊断

#### **奖励不上升**
- 检查：速度跟踪奖励权重
- 检查：网络结构、学习率
- 尝试：增加num_envs

#### **策略崩溃**
- 表现：奖励突然大幅下降
- 原因：更新幅度过大
- 解决：降低learning_rate，增加clip_param

#### **运动不稳定**
- 表现：机器人抖动、翻倒
- 检查：`action_rate_l2`权重
- 检查：力矩和加速度惩罚

#### **能耗过高**
- 表现：策略成功但能耗大
- 调整：增加`joint_torques_l2`, `joint_power`权重
- 调整：降低运动速度

---

## 总结

GO2W训练系统是一个精心设计的多目标优化问题：

1. **核心目标**：速度跟踪（weight=3.0+1.5）
2. **约束条件**：稳定性、能量效率、平滑性
3. **优化方法**：PPO算法（裁剪+熵正则化）
4. **学习策略**：课程学习（地形+速度）
5. **鲁棒性**：事件随机化（质量、质心、惯量）

所有参数都有明确的物理意义和实际效果，通过调整这些参数可以塑造机器人的行为特性。
