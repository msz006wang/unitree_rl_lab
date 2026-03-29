# G1 机器人快速参考手册
# G1 Robot Quick Reference Manual

快速参考G1机器人训练、配置和故障排除。

## 📋 目录

1. [训练模式](#训练模式)
2. [配置模式](#配置模式)
3. [奖励函数](#奖励函数)
4. [关键参数](#关键参数)
5. [训练命令](#训练命令)
6. [监控指标](#监控指标)
7. [故障排除](#故障排除)

---

## 训练模式

### 原始配置模式

| 模式 | 命令 | 说明 |
|------|------|------|
| Original 16-level | `./scripts/train_g1.sh original` | 标准16级渐进式地形训练 |
| Flat Original | `./scripts/train_g1.sh flat-original` | 平地原始配置训练 |

### 改进配置模式

| 模式 | 命令 | 说明 |
|------|------|------|
| Improved 16-level | `./scripts/train_g1.sh improved` | 改进16级渐进式地形训练 |
| Flat Improved | `./scripts/train_g1.sh flat-improved` | 平地改进配置训练 ⭐ |

### 回放模式

| 模式 | 命令 | 说明 |
|------|------|------|
| Play Original | `./scripts/train_g1.sh play-original` | 回放原始策略 |
| Play Improved | `./scripts/train_g1.sh play-improved` | 回放改进策略 |

---

## 配置模式

### 原始配置特性

**文件**: `velocity_env_cfg.py`

**主要特点:**
- 16级渐进式地形难度
- 标准奖励权重
- Action scale: 0.3
- Episode长度: 20.0s
- 初始高度: 0.8m
- 标准终止条件

**适用场景:**
- 完整训练
- 复杂地形适应
- 研究和开发

### 改进配置特性

**文件**: `velocity_env_cfg_improved.py`

**主要特点:**
- 16级渐进式地形难度
- 扩展奖励函数（生存、距离、能量效率、摔倒恢复）
- 优化的奖励权重
- Action scale: 0.35 (平衡稳定性和灵活性)
- Episode长度: 25.0s
- 初始高度: 0.65m (修复后，更稳定)
- 适度收紧的终止条件

**适用场景:**
- 高级训练
- 摔倒恢复能力训练
- 长时间行走任务

### 平地配置特性

**文件**: `velocity_env_cfg_flat.py` (原始) 和 `velocity_env_cfg_flat_improved.py` (改进)

**主要特点:**
- 简单平地地形（plane）
- 基础步态训练
- 快速训练和测试
- 无课程学习

**适用场景:**
- 基础步态验证
- 算法开发
- 快速迭代测试
- 与GO2W对比实验

---

## 奖励函数

### 基础奖励函数

```python
# 任务跟踪奖励
track_lin_vel_xy    # 线速度跟踪 (权重: 1.0)
track_ang_vel_z      # 角速度跟踪 (权重: 0.5)

# 基础保持奖励
alive              # 存活奖励 (权重: 0.1)

# 稳定性惩罚
base_linear_velocity     # Z轴速度惩罚 (权重: -2.0)
base_angular_velocity     # 角速度惩罚 (权重: -0.05)
flat_orientation_l2     # 姿态惩罚 (权重: -5.0 / -3.0)
base_height             # 高度惩罚 (权重: -10.0 / -8.0)
```

### 扩展奖励函数 (仅改进配置)

```python
# 长时间行走奖励
survival                 # 生存奖励 (权重: 0.5)
distance_traveled         # 行走距离奖励 (权重: 0.3)
energy_efficiency          # 能量效率奖励 (权重: 0.1)
consistent_velocity        # 速度一致性奖励 (权重: 0.2)

# 摔倒恢复奖励
fall_recovery             # 摔倒恢复奖励 (权重: 0.5 - 已修复)
stand_up_progress         # 站起进度奖励 (权重: 0.3 - 已修复)
upright_orientation        # 直立姿态奖励 (权重: 0.5)
```

---

## 关键参数

### 训练参数

| 参数 | 默认值 | 说明 |
|------|----------|------|
| --num_envs | 4096 | 环境数量 |
| --headless | 无 | 无GUI模式 |
| --iterations | 10000 | 最大训练迭代数 |
| --seed | 42 | 随机种子 |
| --video | 无 | 视频录制 |

### 环境参数

| 参数 | 原始值 | 改进值 | 平地值 |
|------|----------|----------|----------|
| episode_length_s | 20.0 | 25.0 | 20.0 |
| action_scale | 0.3 | 0.35 | 0.3 |
| decimation | 4 | 4 | 4 |
| sim_dt | 0.005 | 0.005 | 0.005 |

### 终止条件

| 条件 | 原始值 | 改进值 | 说明 |
|------|----------|----------|------|
| min_height | 0.15m | 0.12m | 0.15m | 最小允许高度 |
| max_tilt | 1.0rad | 1.1rad | 1.0rad | 最大允许倾斜角 |

---

## 训练命令

### 推荐训练命令

#### 1. 平地快速测试（推荐首次使用）

```bash
# 使用改进配置，512环境，约30分钟
./scripts/train_g1.sh flat-improved --num_envs 512
```

#### 2. 平地完整训练

```bash
# 使用改进配置，4096环境，约4-8小时
./scripts/train_g1.sh flat-improved --num_envs 4096 --headless
```

#### 3. 16级渐进式地形训练

```bash
# 使用改进配置，8192环境，约8-12小时
./scripts/train_g1.sh improved --num_envs 8192 --headless
```

#### 4. 可视化训练

```bash
# 使用改进配置，带GUI和视频录制
./scripts/train_g1.sh flat-improved --gui --video --video_interval 2000
```

### Python训练命令

#### 基础训练

```bash
# 原始配置训练
python scripts/rsl_rl/train.py --task Unitree-G1-29dof-Velocity --num_envs 4096

# 改进配置训练
python scripts/rsl_rl/train.py --task Unitree-G1-29dof-Velocity-Improved --num_envs 4096

# 平地改进配置训练
python scripts/rsl_rl/train.py --task Unitree-G1-29dof-Velocity-Flat-Improved --num_envs 512
```

#### 高级训练

```bash
# 自定义迭代数
python scripts/rsl_rl/train.py --task Unitree-G1-29dof-Velocity-Improved --max_iterations 50000 --num_envs 8192

# 视频录制
python scripts/rsl_rl/train.py --task Unitree-G1-29dof-Velocity-Improved --video --video_interval 500
```

---

## 监控指标

### TensorBoard关键指标

#### 总体训练指标

- **Train/mean_episode_reward**: 平均episode总奖励
- **Train/mean_reward_per_time_step**: 平均每步奖励
- **Train/mean_episode_length**: 平均episode长度
- **Train/loss/policy_loss**: 策略网络损失
- **Train/loss/value_loss**: 价值网络损失
- **Train/mean_std/mean_std**: 动作标准差

#### 任务相关奖励指标

- **Train/mean_reward/track_lin_vel_xy**: 线速度跟踪奖励
- **Train/mean_reward/track_ang_vel_z**: 角速度跟踪奖励

#### 扩展奖励指标 (改进配置)

- **Train/mean_reward/survival**: 生存奖励
- **Train/mean_reward/distance_traveled**: 行走距离奖励
- **Train/mean_reward/fall_recovery**: 摔倒恢复奖励
- **Train/mean_reward/stand_up_progress**: 站起进度奖励

#### 终止统计

- **Train/termination/time_out**: 正常超时终止 (应该占大多数)
- **Train/termination/base_height**: 高度过低终止 (应该很少)
- **Train/termination/bad_orientation**: 姿态异常终止 (应该很少)

### 成功训练指标

- **Episode长度**: 应该逐步增加，从2步到10+步
- **超时终止比例**: 应该>90%的episode超时终止
- **损失曲线**: 应该平稳下降，不剧烈波动
- **检查点数量**: 应该定期增加

---

## 故障排除

### 问题1: 训练在25个iteration内结束

**症状:**
- 训练启动后立即结束
- 只有3个模型检查点
- TensorBoard显示很短的训练曲线

**可能原因:**
1. 初始高度过高 (已修复: 0.8m → 0.65m)
2. 奖励权重配置不当 (已优化)
3. Action scale过大 (已调整: 0.5 → 0.35)
4. 终止条件过松 (已收紧)

**解决方案:**
```bash
# 使用平地配置重新测试
./scripts/train_g1.sh flat-improved --num_envs 512

# 监控TensorBoard
tensorboard --logdir logs/rsl_rl/
```

### 问题2: 机器人在2-3步内摔倒

**症状:**
- Episode长度稳定在2-3步
- 频繁的高度过低终止
- 训练无法收敛

**可能原因:**
1. 地形过于困难
2. 物理参数不稳定
3. 初始姿态不当

**解决方案:**
```bash
# 1. 使用平地模式训练基础步态
./scripts/train_g1.sh flat-original --num_envs 512

# 2. 减少环境数量以加快调试
./scripts/train_g1.sh flat-improved --num_envs 256

# 3. 启用GUI可视化
./scripts/train_g1.sh flat-improved --gui
```

### 问题3: TensorBoard无法访问

**症状:**
- 浏览器显示"Unable to connect"
- 端口冲突

**解决方案:**
```bash
# 1. 检查TensorBoard进程
ps aux | grep tensorboard

# 2. 停止现有进程
pkill -f tensorboard

# 3. 使用不同端口启动
tensorboard --logdir logs/rsl_rl/ --port 6007

# 4. 清理TensorBoard缓存
rm -rf ~/.tensorboard-info/
```

### 问题4: 训练曲线不收敛

**症状:**
- Episode奖励剧烈波动
- Loss曲线上升不下降
- 策略表现持续退化

**可能原因:**
1. 学习率过高
2. 奖励权重冲突
3. 批大小不合适

**解决方案:**
```bash
# 1. 降低学习率
# 编辑rsl_rl_ppo_cfg.py，修改学习率
learning_rate = 0.0005  # 从0.001降低

# 2. 重新平衡奖励权重
# 编辑velocity_env_cfg_improved.py
# 增加任务跟踪奖励，减少冲突奖励

# 3. 调整批大小
python scripts/rsl_rl/train.py --task Unitree-G1-29dof-Velocity-Flat-Improved --num_envs 512
```

### 问题5: 导入错误 "No module named 'pxr'"

**症状:**
- 配置验证失败
- 训练启动时出现导入错误

**原因:**
- Isaac Lab依赖问题，不影响训练执行

**解决方案:**
```bash
# 直接启动训练（跳过验证）
python scripts/rsl_rl/train.py --task Unitree-G1-29dof-Velocity-Flat-Improved --num_envs 512

# 训练应该可以正常进行
```

---

## 配置对比：G1 vs GO2W

### 主要差异

| 特性 | G1 (改进配置) | GO2W (flat) |
|------|-------------------|------------|
| 地形 | 16级渐进式 + 平地 | 平地 |
| Episode长度 | 25.0s | 20.0s |
| Action scale | 0.35 | 0.3 |
| 初始高度 | 0.65m | 0.45m |
| 摔倒恢复 | 支持 | 不支持 |
| 长时间行走 | 支持 | 基础 |
| 奖励函数 | 基础 + 扩展 | 基础 |

### 选择建议

**使用G1的情况:**
- 需要摔倒恢复能力 → 使用改进配置
- 需要复杂地形适应 → 使用16级渐进式配置
- 快速测试对比 → 使用平地模式
- 最稳定基础步态 → 使用平地原始配置

**使用GO2W的情况:**
- 简单轮足机器人训练 → 使用GO2W
- 平地基础训练 → 使用GO2W flat模式
- 性能对比基准 → 使用GO2W作为参考

---

## 快速命令参考

### 验证配置

```bash
# 验证改进配置
./scripts/validate_improved_config.sh

# 测试导入
python test_g1_flat_import.py
```

### 监控训练

```bash
# 启动TensorBoard
tensorboard --logdir logs/rsl_rl/ --port 6006

# 在另一个终端查看具体G1训练
tensorboard --logdir logs/rsl_rl/unitree_g1_29dof_velocity_improved/ --port 6007
```

### 测试训练好的策略

```bash
# 回放改进配置策略
./scripts/train_g1.sh play-improved --load_run recent

# 可视化模式
./scripts/train_g1.sh play-improved --gui
```

### 清理训练数据

```bash
# 查看训练日志
ls -lht logs/rsl_rl/unitree_g1_29dof_velocity_improved/

# 查看最新的训练运行
ls -lht logs/rsl_rl/

# 查看TensorBoard事件文件
ls logs/rsl_rl/unitree_g1_29dof_velocity_improved/*/events.out.*
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

**推荐开始命令:**
```bash
./scripts/train_g1.sh flat-improved --num_envs 512
```

**开始训练吧!** 🚀
