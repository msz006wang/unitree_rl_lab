# GO2W训练快速参考指南

## 📋 训练命令

### Flat模式 (平地训练)
```bash
./scripts/quick_start_training.sh flat
```

### Rough模式 (粗糙地形训练)
```bash
./scripts/quick_start_training.sh rough
```

### 指定GPU
```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/quick_start_training.sh flat
```

### 恢复训练
```bash
./scripts/quick_start_training.sh flat --resume --load_run recent
```

---

## 🎯 核心训练指标

### 关键指标 (TensorBoard)

| 指标 | 良好范围 | 说明 |
|------|----------|------|
| **Reward/total** | > 3.0 | 总奖励，越高越好 |
| **Reward/track_lin_vel_xy_exp** | > 2.5 | 线速度跟踪 |
| **Reward/track_ang_vel_z_exp** | > 1.2 | 角速度跟踪 |
| **Reward/lin_vel_z_l2** | > -0.5 | Z轴速度 (接近0) |
| **Value/value_loss** | < 0.2 | 价值损失，越低越好 |
| **Policy/entropy** | 0.1-0.3 | 策略熵 (探索度) |
| **Policy/kl_divergence** | ≈ 0.01 | 策略更新幅度 |

### 训练阶段检查点

**1000 iters**: 基础学习阶段
- total_reward: -5 → 0
- 机器人学会基本移动

**5000 iters**: 技能提升阶段
- total_reward: 0 → 2.0
- 速度跟踪精度提升

**10000 iters**: 精细优化阶段
- total_reward: 2.0 → 4.0
- 运动平滑，能耗降低

**20000+ iters**: 鲁棒性增强
- total_reward: 4.0+ (稳定)
- 适应复杂地形

---

## 🔧 训练超参数

### PPO算法参数

```python
# 学习率
learning_rate = 1e-3          # 初始学习率
schedule = "adaptive"         # 自适应调整

# PPO裁剪
clip_param = 0.2              # 策略更新限制 [0.8, 1.2]

# 损失系数
value_loss_coef = 1.0         # 价值损失权重
entropy_coef = 0.01           # 熵系数 (探索)

# GAE参数
gamma = 0.99                  # 折扣因子 (时间视野)
lam = 0.95                    # GAE λ (偏差-方差权衡)

# 训练批次
num_learning_epochs = 5       # 每批数据学习轮次
num_mini_batches = 4          # 每轮mini-batch数
num_steps_per_env = 24        # 每环境收集步数

# KL控制
desired_kl = 0.01             # 目标KL散度
max_grad_norm = 1.0           # 梯度裁剪
```

### 环境参数

```python
# 场景
num_envs = 4096               # 并行环境数
env_spacing = 2.5             # 环境间距 (米)

# 物理仿真
sim_dt = 0.005                # 物理时间步 (秒) - 200Hz
decimation = 4                # 控制降频
episode_length_s = 20.0       # Episode长度 (秒)

# 控制频率 = 200Hz / 4 = 50Hz (每步0.02秒)
```

---

## 📊 观测空间 (41维)

### 观测组成

| 观测项 | 维度 | 物理意义 |
|--------|------|----------|
| **base_ang_vel** | 3 | 基座角速度 [ω_x, ω_y, ω_z] |
| **projected_gravity** | 3 | 重力投影 [g_x, g_y, g_z] |
| **velocity_commands** | 3 | 速度命令 [v_x, v_y, ω_z] |
| **joint_pos_leg** | 12 | 腿部关节位置 (12个腿关节) |
| **joint_vel_wheel** | 4 | 轮子速度 (4个轮子) |
| **last_action** | 16 | 上一步动作 (12腿+4轮) |

### 观测缩放

```python
base_ang_vel.scale = 0.25     # 降低敏感度
joint_vel.scale = 0.05        # 速度变化小
joint_pos.scale = 1.0         # 保持原范围
```

---

## 🎮 动作空间 (16维)

### 混合控制

```python
动作 = [
    腿部位置控制 (12维) + 轮子速度控制 (4维)
]

# 腿部: 位置增量控制
scale = {
    ".*_hip_joint": 0.125,    # 髋关节小幅
    "^(?!.*_hip_joint).*": 0.25  # 其他大幅
}
实际位置 = 默认位置 + (输出 × scale)

# 轮子: 速度增量控制
scale = 5.0
实际速度 = 默认速度 + (输出 × 5.0)
```

---

## 💰 奖励函数速查

### 主要奖励 (正权重)

| 奖励项 | 权重 | 公式 | 目标 |
|--------|------|------|------|
| **track_lin_vel_xy_exp** | 3.0 | exp(-error²/0.25) | 线速度跟踪 |
| **track_ang_vel_z_exp** | 1.5 | exp(-error²/0.25) | 角速度跟踪 |
| **upward** | 1.0 | (1-g_z)² | 保持直立 |
| **feet_contact_without_cmd** | 0.1 | num_contact × 没有命令 | 停止稳定 |

### 主要惩罚 (负权重)

| 奖励项 | 权重 | 公式 | 目的 |
|--------|------|------|------|
| **joint_pos_limits** | -5.0 | Σ(超限)² | 硬约束 |
| **lin_vel_z_l2** | -2.0 | -v_z² | 禁止跳动 |
| **stand_still** | -2.0 | -偏差×停止 | 站立姿态 |
| **joint_pos_penalty** | -1.0 | -偏差×条件 | 位置偏离 |
| **undesired_contacts** | -1.0 | -Σ力² | 非期望接触 |
| **action_rate_l2** | -0.01 | -Δaction² | 平滑控制 |
| **joint_mirror** | -0.05 | -Σ差² | 对称步态 |

### 次要惩罚 (极小权重)

| 奖励项 | 权重 | 说明 |
|--------|------|------|
| **joint_torques_l2** | -2.5e-5 | 力矩惩罚 (数量级大) |
| **joint_acc_l2** | -2.5e-7 | 加速度惩罚 (数量级极大) |
| **joint_power** | -2e-5 | 功率惩罚 (数量级大) |
| **contact_forces** | -1.5e-4 | 接触力惩罚 (数量级大) |

---

## 🎓 课程学习

### 地形课程 (自动调整)

```
Level 0: 平面
Level 1: 矮障碍
Level 2: 中等障碍
Level 3: 高障碍
Level 4: 楼梯
Level 5: 斜坡

触发: 基于速度跟踪奖励
```

### 命令课程 (渐进式)

```
阶段1: range_multiplier = 0.1  → [-0.1, 0.1] m/s
阶段2: range_multiplier = 0.3  → [-0.3, 0.3] m/s
阶段3: range_multiplier = 0.5  → [-0.5, 0.5] m/s
...
阶段N: range_multiplier = 1.0  → [-1.0, 1.0] m/s
```

---

## 🎲 随机化策略

### 启动时 (每Episode)

| 事件 | 参数 | 作用 |
|------|------|------|
| **材质随机化** | μ∈[0.3,1.0] | 适应不同地面 |
| **质量随机化** | 基座±1kg, 其他±30% | 适应不同负载 |
| **质心随机化** | ±5cm | 适应质心变化 |
| **惯量随机化** | 80%-120% | 适应惯性特性 |

### 重置时 (每Episode)

| 事件 | 参数 | 作用 |
|------|------|------|
| **基座位置** | ±0.5m | 防止位置过拟合 |
| **基座朝向** | 任意yaw | 学习任意朝向 |
| **基座速度** | ±0.5m/s | 学习动态恢复 |
| **执行器增益** | 50%-200% | 适应控制器参数 |

### 间隔 (运行时)

| 事件 | 间隔 | 作用 |
|------|------|------|
| **推机器人** | 10-15秒 | 持续扰动，动态平衡 |

---

## 🐛 故障排查

### 问题1: 训练不稳定

**症状**: 奖励剧烈波动

```python
# 解决方案
learning_rate = 5e-4        # 降低学习率
clip_param = 0.1            # 更保守裁剪
num_steps_per_env = 32      # 增加batch
```

### 问题2: 策略过早收敛

**症状**: entropy < 0.01 (太快)

```python
# 解决方案
entropy_coef = 0.02         # 增加熵系数
learning_rate = 2e-3        # 增加学习率
num_learning_epochs = 3     # 减少学习轮次
```

### 问题3: 局部最优

**症状**: 只会站立，不会移动

```python
# 解决方案
rewards.track_lin_vel_xy_exp.weight = 5.0  # 临时增加
rewards.lin_vel_z_l2.weight = -0.5        # 降低惩罚
curriculum.command_levels.range_multiplier = (0.05, 0.5)  # 简化任务
```

### 问题4: 内存不足

**症状**: CUDA out of memory

```python
# 解决方案
num_envs = 2048             # 减少环境数
num_steps_per_env = 16      # 减少步数
policy.actor_hidden_dims = [256, 128]  # 减小网络
```

---

## 📈 TensorBoard使用

### 启动TensorBoard

```bash
tensorboard --logdir=logs/rsl_rl/unitree_go2w_velocity_flat_v0
```

### 关键图表

**训练概览**:
- `Reward/total`: 总奖励趋势
- `Value/value_loss`: 价值收敛情况
- `Policy/entropy`: 探索度变化

**详细奖励**:
- `Reward/track_lin_vel_xy_exp`: 速度跟踪
- `Reward/lin_vel_z_l2`: 垂直运动
- `Reward/joint_torques_l2`: 力矩使用

**训练信息**:
- `Info/fps`: 训练速度
- `Info/learning_rate`: 学习率调整
- `Info/iteration_time`: 每迭代时间

---

## 🔍 监控命令

### 查看训练日志

```bash
# 实时查看
tail -f logs/rsl_rl/*/logs/stdout.txt

# 查看最近错误
grep -i "error\|warning" logs/rsl_rl/*/logs/stdout.txt | tail -20
```

### 检查训练进度

```bash
# 查看TensorBoard事件文件
ls -lh logs/rsl_rl/unitree_go2w_velocity_flat_v0/*/events.out.tfevents.*

# 查看保存的模型
ls -lh logs/rsl_rl/unitree_go2w_velocity_flat_v0/*/model_*.pt
```

### 监控GPU使用

```bash
# 实时监控
watch -n 1 nvidia-smi

# 查看特定GPU
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv
```

---

## 📦 文件结构

### 训练相关

```
logs/rsl_rl/unitree_go2w_velocity_flat_v0/
├── 2026-03-08_22-06-14/          # 训练运行
│   ├── params/                    # 配置文件
│   │   ├── env.yaml              # 环境配置
│   │   ├── agent.yaml            # PPO配置
│   │   └── velocity_env_cfg.py   # 原始配置
│   ├── model_*.pt                # 模型检查点
│   ├── events.out.tfevents.*     # TensorBoard日志
│   └── logs/
│       └── stdout.txt            # 训练输出
```

### 配置文件

```
source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2w/
├── __init__.py                   # 任务注册
└── velocity_env_cfg.py           # 环境配置
    ├── RobotSceneCfg             # 场景配置
    ├── EventCfg                  # 事件配置
    ├── ObservationsCfg           # 观测配置
    ├── ActionsCfg                # 动作配置
    ├── RewardsCfg                # 奖励配置
    ├── TerminationsCfg           # 终止配置
    ├── CurriculumCfg             # 课程配置
    ├── RobotEnvCfg               # 粗糙地形
    └── RobotFlatEnvCfg           # 平地
```

---

## ⚙️ 常用修改

### 调整训练时长

```python
# rsl_rl_ppo_cfg.py
max_iterations = 10000  # 默认50000
```

### 调整并行环境数

```bash
# 命令行
./scripts/quick_start_training.sh flat --num_envs 2048
```

### 调整保存频率

```python
# rsl_rl_ppo_cfg.py
save_interval = 50  # 默认100，更频繁保存
```

### 调整Episode长度

```python
# velocity_env_cfg.py
episode_length_s = 30.0  # 默认20.0秒
```

---

## 📚 相关文档

- **[GO2W训练过程全面解析](GO2W_TRAINING_PROCESS_ANALYSIS.md)**: 详细算法和参数解释
- **[GO2W奖励函数代码详解](GO2W_REWARD_FUNCTIONS_CODE.md)**: 奖励函数源码解析
- **[GO2W训练参数物理意义](GO2W_TRAINING_PARAMS_PHYSICS.md)**: PPO算法和损失函数

---

## 🚀 快速开始

### 1. 验证配置

```bash
python scripts/verify_flat_config.py
```

### 2. 短期测试 (100 iters)

```bash
./scripts/quick_start_training.sh flat --max_iterations 100
```

### 3. 完整训练

```bash
./scripts/quick_start_training.sh flat
```

### 4. 监控训练

```bash
# 终端1: 启动TensorBoard
tensorboard --logdir=logs/rsl_rl

# 终端2: 监控日志
tail -f logs/rsl_rl/*/logs/stdout.txt

# 终端3: 监控GPU
watch -n 1 nvidia-smi
```

---

**版本**: 1.0
**更新**: 2026-03-08
**快速查阅**: 训练GO2W机器人的必备参考
