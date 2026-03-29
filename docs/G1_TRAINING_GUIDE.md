# G1机器人训练指南
# G1 Robot Training Guide

## 目录

1. [项目概述](#项目概述)
2. [环境要求](#环境要求)
3. [安装说明](#安装说明)
4. [快速开始](#快速开始)
5. [训练配置](#训练配置)
6. [奖励函数说明](#奖励函数说明)
7. [监控和调试](#监控和调试)
8. [常见问题](#常见问题)

---

## 项目概述

本项目基于 Isaac Lab 框架，使用 RSL-RL (Rugged Scenery Learning - Reinforcement Learning) 库对 Unitree G1 双足机器人进行强化学习训练。G1 是一个具有29个自由度的高性能人形机器人。

### 主要特性

- **基于 Isaac Lab**: 使用 NVIDIA Isaac Lab 物理仿真环境
- **RSL-RL 算法**: 使用 PPO (Proximal Policy Optimization) 算法
- **多环境并行训练**: 支持高达 4096 个并行环境
- **多种训练模式**: 支持16级渐进式地形和平地训练
- **速度跟踪任务**: 训练机器人跟踪指定的线速度和角速度命令
- **摔倒恢复能力**: 改进配置支持摔倒后自动恢复

### 任务类型

- **原始配置 (Original)**: 标准16级渐进式地形训练
- **改进配置 (Improved)**: 包含长时间行走和摔倒恢复功能的训练
- **平地模式 (Flat)**: 在简单平地上进行基础训练
- **渐进式地形 (Progressive)**: 在16级不同难度地形上进行高级训练

---

## 环境要求

### 硬件要求

- **GPU**: NVIDIA GPU (推荐 RTX 3090 或更高)
- **显存**: 至少 24GB (用于 4096 环境)
- **内存**: 至少 32GB RAM
- **存储**: 至少 10GB 可用空间

### 软件要求

- **操作系统**: Ubuntu 20.04+ / Windows 10+
- **Python**: 3.10+
- **CUDA**: 11.8+
- **Isaac Sim**: 5.1.0+
- **Isaac Lab**: 2.3.0+
- **RSL-RL**: 5.0.1+

---

## 快速开始

### 方法1：使用训练脚本 (推荐)

#### 基础训练（16级渐进式地形）

```bash
# 原始配置训练
./scripts/train_g1.sh original

# 改进配置训练
./scripts/train_g1.sh improved
```

#### 平地训练（简单地形）

```bash
# 原始配置的平地训练
./scripts/train_g1.sh flat-original

# 改进配置的平地训练 (推荐)
./scripts/train_g1.sh flat-improved --num_envs 512
```

### 方法2：使用Python训练脚本

```bash
# 使用原始配置
python scripts/rsl_rl/train.py --task Unitree-G1-29dof-Velocity --num_envs 4096

# 使用改进配置
python scripts/rsl_rl/train.py --task Unitree-G1-29dof-Velocity-Improved --num_envs 4096

# 使用平地改进配置
python scripts/rsl_rl/train.py --task Unitree-G1-29dof-Velocity-Flat-Improved --num_envs 512
```

### 方法3：快速测试命令

```bash
# 快速平地测试 (512环境，约30分钟)
./scripts/train_g1.sh flat-improved --num_envs 512

# 大规模平地训练 (4096环境，约4-8小时)
./scripts/train_g1.sh flat-improved --num_envs 4096 --headless

# 可视化训练
./scripts/train_g1.sh flat-improved --gui --video --video_interval 2000
```

---

## 训练配置

### 原始配置 vs 改进配置对比

| 特性 | 原始配置 | 改进配置 |
|------|-----------|-----------|
| **地形类型** | 16级渐进式 | 16级渐进式 + 平地 |
| **Episode长度** | 20.0s | 25.0s |
| **Action Scale** | 0.3 | 0.35 (平衡稳定性和灵活性) |
| **初始高度** | 0.8m | 0.65m (更稳定) |
| **基础奖励** | 标准任务跟踪 | 标准任务跟踪 |
| **扩展奖励** | 无 | 生存、距离、能量效率、速度一致性 |
| **摔倒恢复** | 无 | 摔倒检测、恢复奖励、站起进度 |
| **终止条件** | 标准 | 适度收紧 (更稳定) |

### 奖励权重配置

#### 原始配置关键权重

```python
# 任务相关奖励
track_lin_vel_xy = 1.0        # 线速度跟踪
track_ang_vel_z = 0.5         # 角速度跟踪

# 基础保持奖励
alive = 0.1                   # 存活奖励

# 姿态和稳定性惩罚
base_linear_velocity = -2.0      # Z轴速度惩罚 (防跳跃)
base_angular_velocity = -0.05    # 角速度惩罚
flat_orientation_l2 = -5.0      # 姿态惩罚
base_height = -10.0              # 高度惩罚
```

#### 改进配置关键权重

```python
# 任务相关奖励 (优化后)
track_lin_vel_xy = 1.0        # 线速度跟踪 (从2.0降低)
track_ang_vel_z = 0.5         # 角速度跟踪 (从1.0降低)

# 长时间行走奖励
survival = 0.5                 # 生存奖励 (每个时间步)
distance_traveled = 0.3         # 行走距离奖励
energy_efficiency = 0.1         # 能量效率奖励
consistent_velocity = 0.2         # 速度一致性奖励

# 摔倒恢复奖励 (修复后权重)
fall_recovery = 0.5             # 摔倒恢复 (从5.0大幅降低)
stand_up_progress = 0.3         # 站起进度 (从2.0大幅降低)
upright_orientation = 0.5        # 直立姿态奖励

# 基础保持奖励
alive = 0.1                   # 存活奖励

# 姿态和稳定性惩罚 (修复后)
base_linear_velocity = -2.0      # Z轴速度惩罚
base_angular_velocity = -0.05    # 角速度惩罚
flat_orientation_l2 = -3.0      # 姿态惩罚 (从-5.0提高)
base_height = -8.0              # 高度惩罚 (从-10.0提高)
```

### 环境配置

#### 平地地形配置

```python
# 地形类型
terrain_type = "plane"  # 简单平地

# 地形物理材质
friction = 1.0               # 摩擦系数
restitution = 0.0          # 弹性系数

# 其他配置
env_spacing = 2.5             # 环境间距
decimation = 4                 # 动作间隔
sim_dt = 0.005               # 物理时间步长
```

#### 16级渐进式地形配置

```python
# 地形类型
terrain_type = "generator"      # 程序生成地形

# 地形难度等级
num_rows = 16                # 16个难度等级
num_cols = 21                # 地形网格列数

# 子地形组成 (按比例)
flat: 25%                   # 平地 (基础训练)
rough_terrain: 30%          # 随机粗糙地形
gentle_slopes: 20%          # 温和斜坡
stairs: 15%                  # 楼梯
obstacles: 10%               # 障碍物

# 地形参数
horizontal_scale = 0.1         # 水平缩放
vertical_scale = 0.005          # 垂直缩放
slope_threshold = 0.75         # 斜坡阈值
difficulty_range = (0.0, 1.0)  # 难度范围
curriculum = True              # 启用课程学习
```

---

## 奖励函数说明

### 基础任务奖励

1. **`track_lin_vel_xy`** - 线速度跟踪
   - 目标: 跟踪期望的线速度命令
   - 权重: 1.0 (原始/改进)
   - 测量: 速度跟踪误差的指数衰减

2. **`track_ang_vel_z`** - 角速度跟踪
   - 目标: 跟踪期望的角速度命令
   - 权重: 0.5
   - 测量: 角速度跟踪误差的指数衰减

### 长时间行走奖励 (仅改进配置)

3. **`survival`** - 生存奖励
   - 目标: 鼓励机器人保持存活
   - 权重: 0.5
   - 计算: 每个时间步给予固定奖励

4. **`distance_traveled`** - 行走距离奖励
   - 目标: 鼓励机器人前进更远
   - 权重: 0.3
   - 计算: 累计前进距离

5. **`energy_efficiency`** - 能量效率奖励
   - 目标: 鼓励机器人高效运动
   - 权重: 0.1
   - 计算: 奖励低能量消耗

6. **`consistent_velocity`** - 速度一致性奖励
   - 目标: 减少速度波动
   - 权重: 0.2
   - 计算: 惩罚速度变化过大

### 摔倒恢复奖励 (仅改进配置)

7. **`fall_recovery`** - 摔倒恢复奖励
   - 目标: 奖励从摔倒状态恢复
   - 权重: 0.5 (修复后，避免诱导摔倒)
   - 触发条件: 检测到摔倒后的恢复行为

8. **`stand_up_progress`** - 站起进度奖励
   - 目标: 奖励重新站立的过程
   - 权重: 0.3 (修复后)
   - 计算: 评估站起进度

9. **`upright_orientation`** - 直立姿态奖励
   - 目标: 奖励保持直立姿态
   - 权重: 0.5
   - 计算: 评估姿态直立程度

---

## 监控和调试

### 训练监控

#### TensorBoard监控

```bash
# 启动TensorBoard
tensorboard --logdir logs/rsl_rl/ --port 6006

# 访问地址
# http://localhost:6006
```

#### 关键监控指标

**总体指标:**
- `Train/mean_episode_reward` - 平均episode奖励
- `Train/mean_reward_per_time_step` - 平均每步奖励
- `Train/mean_episode_length` - 平均episode长度

**任务相关奖励:**
- `Train/mean_reward/track_lin_vel_xy` - 线速度跟踪奖励
- `Train/mean_reward/track_ang_vel_z` - 角速度跟踪奖励

**扩展奖励 (改进配置):**
- `Train/mean_reward/survival` - 生存奖励
- `Train/mean_reward/distance_traveled` - 行走距离奖励
- `Train/mean_reward/fall_recovery` - 摔倒恢复奖励

**训练损失:**
- `Train/loss/policy_loss` - 策略损失
- `Train/loss/value_loss` - 价值损失
- `Train/mean_std/mean_std` - 动作标准差

**终止统计:**
- `Train/termination/time_out` - 超时终止 (好)
- `Train/termination/base_height` - 高度过低终止 (坏)
- `Train/termination/bad_orientation` - 姿态异常终止 (坏)

### 调试工具

#### 配置验证

```bash
# 验证配置文件
./scripts/validate_improved_config.sh

# 测试Python导入
python3 test_g1_flat_import.py
```

#### 可视化工具

```bash
# 地形可视化 (16级渐进式)
python scripts/rsl_rl/visualize_terrains.py --task Unitree-G1-29dof-Velocity-Improved --num_envs 16 --real-time

# 回放训练好的策略
./scripts/train_g1.sh play-improved --load_run recent
```

---

## 常见问题

### Q1: 训练在1分钟内就结束了怎么办？

**可能原因:**
1. 初始高度设置过高 (已修复: 0.8m → 0.65m)
2. 奖励权重配置不当 (已优化)
3. Action scale过大 (已调整: 0.5 → 0.35)

**解决方案:**
1. 确保使用修复后的配置
2. 使用平地模式进行初步测试
3. 监控TensorBoard中的episode长度

```bash
# 使用平地改进配置重新测试
./scripts/train_g1.sh flat-improved --num_envs 512
```

### Q2: 机器人频繁摔倒怎么办？

**可能原因:**
1. 物理参数不稳定
2. 终止条件过松
3. 地形过于困难

**解决方案:**
1. 使用平地模式训练基础步态
2. 检查终止条件设置
3. 减少环境数量以加快训练

```bash
# 使用平地原始配置
./scripts/train_g1.sh flat-original --num_envs 512
```

### Q3: 训练曲线不收敛怎么办？

**可能原因:**
1. 奖励函数冲突
2. 学习率过高
3. Episode长度过短

**解决方案:**
1. 调整学习率 (修改rsl_rl_ppo_cfg.py)
2. 简化奖励函数
3. 增加训练时间

### Q4: TensorBoard无法访问怎么办？

**解决方案:**
```bash
# 1. 确认TensorBoard进程正在运行
ps aux | grep tensorboard

# 2. 检查日志目录
ls -la logs/rsl_rl/

# 3. 使用不同的端口
tensorboard --logdir logs/rsl_rl/ --port 6007
```

---

## 训练模式说明

| 模式 | 任务名称 | 地形类型 | 用途 |
|------|----------|----------|------|
| `original` | Unitree-G1-29dof-Velocity | 16级渐进式 | 完整训练 |
| `improved` | Unitree-G1-29dof-Velocity-Improved | 16级渐进式 | 改进功能训练 |
| `flat-original` | Unitree-G1-29dof-Velocity-Flat | 平地 | 基础步态训练 |
| `flat-improved` | Unitree-G1-29dof-Velocity-Flat-Improved | 平地 | 改进步态训练 ⭐ |

---

## 推荐训练流程

### 阶段1：快速验证 (30分钟)

```bash
# 平地快速测试
./scripts/train_g1.sh flat-improved --num_envs 512
```

**验证要点:**
- 训练能持续进行，不提前结束
- Episode长度达到10步以上
- TensorBoard显示正常训练曲线

### 阶段2：基础训练 (2-4小时)

```bash
# 平地完整训练
./scripts/train_g1.sh flat-improved --num_envs 4096
```

### 阶段3：高级训练 (8-12小时)

```bash
# 16级渐进式地形训练
./scripts/train_g1.sh improved --num_envs 8192
```

### 阶段4：性能优化

```bash
# 带视频录制的大规模训练
./scripts/train_g1.sh improved --num_envs 16384 --video --video_interval 2000
```

---

## 训练参数说明

### 基本参数

```bash
--num_envs N        # 环境数量 (推荐: 512, 2048, 4096, 8192, 16384)
--headless          # 无GUI模式 (训练更快)
--gui               # 强制启用GUI可视化
--device D          # 设备选择 (cuda:0, cuda:1)
--iterations N      # 最大训练迭代数 (默认: 10000)
--seed N            # 随机种子 (默认: 42)
--video             # 启用视频录制
--resume            # 从最新检查点恢复训练
```

### 高级参数

```bash
# 在Python训练脚本中直接指定
--task TASK_NAME                    # 自定义任务名称
--max_epochs MAX_EPOCHS              # 最大训练轮数
--learning_rate LEARNING_RATE         # 自定义学习率
--gamma GAMMA                       # 自定义折扣因子
```

---

## 文件结构说明

```
scripts/
├── train_g1.sh                    ⭐ G1训练主脚本
├── rsl_rl/
│   └── train.py                  # 通用训练入口
├── validate_improved_config.sh     # 配置验证脚本
├── compare_configs.py              # 配置对比工具
└── test_g1_flat_import.py         # 导入测试脚本

source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/
├── velocity_env_cfg.py              # 原始配置
├── velocity_env_cfg_improved.py    # 改进配置 (16级地形)
├── velocity_env_cfg_flat.py          # 原始平地配置
├── velocity_env_cfg_flat_improved.py # 改进平地配置 ⭐
├── actions_cfg.py                   # 动作配置
└── __init__.py                       # 任务注册
```

---

## 总结

**主要改进:**
1. ✅ 降低了机器人初始高度 (0.8m → 0.65m)
2. ✅ 优化了奖励权重配置 (避免过高权重)
3. ✅ 调整了Action scale (0.5 → 0.35)
4. ✅ 收紧了终止条件 (提高稳定性)
5. ✅ 支持平地训练模式 (简化环境)
6. ✅ 完整的训练脚本和文档

**推荐训练命令:**
```bash
# 最简单的平地训练 (推荐开始)
./scripts/train_g1.sh flat-improved --num_envs 512
```

**准备训练了吗?**
- ✅ 检查环境: `conda activate env_isaaclab`
- ✅ 验证配置: `./scripts/validate_improved_config.sh`
- ✅ 开始训练: `./scripts/train_g1.sh flat-improved`

**开始训练吧!** 🚀
