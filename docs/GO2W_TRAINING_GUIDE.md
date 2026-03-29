# GO2W 机器人训练指南

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

本项目基于 Isaac Lab 框架，使用 RSL-RL (Rugged Scenery Learning - Reinforcement Learning) 库对 Unitree GO2W 四足机器人进行强化学习训练。GO2W 是一个轮式四足机器人，具有强大的运动能力和适应性。

### 主要特性

- **基于 Isaac Lab**: 使用 NVIDIA Isaac Lab 物理仿真环境
- **RSL-RL 算法**: 使用 PPO (Proximal Policy Optimization) 算法
- **多环境并行训练**: 支持高达 4096 个并行环境
- **两种地形模式**: Flat (平地) 和 Rough (复杂地形)
- **速度跟踪任务**: 训练机器人跟踪指定的线速度和角速度命令

### 任务类型

- **Flat Terrain**: 在平地上进行速度跟踪训练
- **Rough Terrain**: 在程序生成的复杂地形上进行速度跟踪训练

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

## 安装说明

### 1. 克隆项目

```bash
git clone https://github.com/unitreerobotics/unitree-rl-lab.git
cd unitree-rl-lab
```

### 2. 安装依赖

```bash
# 安装 Isaac Lab 依赖
./isaaclab.sh --install

# 安装项目特定依赖
pip install -r requirements.txt
```

### 3. 验证安装

```bash
# 检查 Isaac Lab
python -c "import isaaclab; print('Isaac Lab version:', isaaclab.__version__)"

# 检查 RSL-RL
python -c "import importlib.metadata; print('RSL-RL version:', importlib.metadata.version('rsl-rl-lib'))"

# 检查项目
python -c "import unitree_rl_lab; print('Unitree RL Lab loaded successfully')"
```

---

## 快速开始

### 使用快速启动脚本

```bash
# 平地训练
./scripts/quick_start_training.sh flat

# 复杂地形训练
./scripts/quick_start_training.sh rough
```

### 使用完整训练脚本

```bash
# 平地训练（默认设置）
./scripts/train_go2w.sh flat

# 复杂地形训练
./scripts/train_go2w.sh rough

# 自定义环境数量
./scripts/train_go2w.sh flat --num_envs 8192

# 启用 GUI 可视化
./scripts/train_go2w.sh flat --gui

# 启用视频录制
./scripts/train_go2w.sh flat --video

# 从检查点恢复训练
./scripts/train_go2w.sh flat --resume
```

### 直接使用 Python 训练

```bash
# 平地训练
python scripts/rsl_rl/train.py \
    --task Unitree-Go2W-Velocity-Flat-v0 \
    --num_envs 4096 \
    --device cuda:0 \
    --max_iterations 10000 \
    --headless

# 复杂地形训练
python scripts/rsl_rl/train.py \
    --task Unitree-Go2W-Velocity-Rough-v0 \
    --num_envs 4096 \
    --device cuda:0 \
    --max_iterations 10000 \
    --headless
```

---

## 训练配置

### 命令行参数

| 参数 | 说明 | 默认值 | 推荐范围 |
|------|------|----------|-----------|
| `--task` | 任务名称 | - | `Unitree-Go2W-Velocity-Flat-v0` 或 `Unitree-Go2W-Velocity-Rough-v0` |
| `--num_envs` | 并行环境数量 | 4096 | 2048-8192 |
| `--device` | 计算设备 | cuda:0 | cuda:0, cuda:1, ... |
| `--max_iterations` | 最大训练迭代次数 | 10000 | 10000-50000 |
| `--seed` | 随机种子 | 42 | 任意整数 |
| `--headless` | 无头模式（无 GUI） | True | True/False |
| `--gui` | 启用 GUI | False | True/False |
| `--video` | 启用视频录制 | False | True/False |
| `--video_interval` | 视频录制间隔 | 2000 | 1000-5000 |
| `--resume` | 从检查点恢复 | False | True/False |

### 环境配置

主要配置文件位于：
- `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/agents/rsl_rl_ppo_cfg.py`
- `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/velocity_env_cfg.py`

#### 网络架构

**Actor 网络 (策略网络):**
```python
actor = RslRlMLPModelCfg(
    hidden_dims=[512, 256, 128],  # 隐藏层维度
    activation="elu",               # 激活函数
    obs_normalization=False,          # 观察归一化
    distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
        init_std=1.0               # 初始标准差
    ),
)
```

**Critic 网络 (价值网络):**
```python
critic = RslRlMLPModelCfg(
    hidden_dims=[512, 256, 128],  # 隐藏层维度
    activation="elu",               # 激活函数
    obs_normalization=False,          # 观察归一化
)
```

#### PPO 算法参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `value_loss_coef` | 1.0 | 价值损失系数 |
| `use_clipped_value_loss` | True | 使用裁剪价值损失 |
| `clip_param` | 0.2 | PPO 裁剪参数 |
| `entropy_coef` | 0.01 | 熵系数（鼓励探索） |
| `num_learning_epochs` | 5 | 每次更新的学习轮数 |
| `num_mini_batches` | 4 | 每次更新的 mini-batch 数量 |
| `learning_rate` | 1.0e-3 | 学习率 |
| `schedule` | "adaptive" | 学习率调度策略 |
| `gamma` | 0.99 | 折扣因子 |
| `lam` | 0.95 | GAE lambda 参数 |
| `desired_kl` | 0.01 | 期望 KL 散度 |
| `max_grad_norm` | 1.0 | 最大梯度范数 |

### 仿真参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `decimation` | 4 | 动作降采样率 |
| `episode_length_s` | 20.0 | 每回合最大时长（秒） |
| `sim.dt` | 0.005 | 物理时间步长 |
| `num_steps_per_env` | 24 | 每个环境的步数 |

### 命令配置

**速度命令范围（Flat 模式）:**
```python
ranges = {
    "lin_vel_x": (-0.1, 0.1),   # X 方向线速度
    "lin_vel_y": (-0.1, 0.1),   # Y 方向线速度
    "ang_vel_z": (-1, 1),        # Z 轴角速度
}
```

**速度命令限制:**
```python
limit_ranges = {
    "lin_vel_x": (-1.0, 1.0),   # X 方向线速度限制
    "lin_vel_y": (-0.4, 0.4),   # Y 方向线速度限制
    "ang_vel_z": (-1.0, 1.0),   # Z 轴角速度限制
}
```

---

## 奖励函数说明

### 主要奖励项

| 奖励项 | 权重 | 说明 |
|---------|-------|------|
| `track_lin_vel_xy` | 1.5 | 跟踪 XY 平面线速度（指数形式） |
| `track_ang_vel_z` | 0.75 | 跟踪 Z 轴角速度（指数形式） |
| `joint_pos` | -0.7 | 关节位置惩罚（鼓励站立姿态） |

### 惩罚项

| 惩罚项 | 权重 | 说明 |
|---------|-------|------|
| `base_linear_velocity` | -2.0 | 基础 Z 轴线速度惩罚（防止跳跃） |
| `base_angular_velocity` | -0.05 | 基础 XY 平面角速度惩罚（防止翻滚） |
| `joint_vel` | -0.001 | 关节速度惩罚（节能） |
| `joint_acc` | -2.5e-7 | 关节加速度惩罚（平滑动作） |
| `joint_torques` | -2e-4 | 关节扭矩惩罚（节能） |
| `action_rate` | -0.1 | 动作变化率惩罚（平滑动作） |
| `dof_pos_limits` | -10.0 | 关节位置限制惩罚 |
| `energy` | -2e-5 | 能量惩罚（节能） |
| `flat_orientation` | -2.5 | 姿态平坦度惩罚（防止倾斜） |
| `feet_air_time` | 0.1 | 足部空中时间奖励（鼓励步态） |
| `air_time_variance` | -1.0 | 空中时间变化惩罚（鼓励接触） |
| `feet_slide` | -0.1 | 足部滑动惩罚 |
| `undesired_contacts` | -1.0 | 不期望接触惩罚（头部、大腿、小腿等） |

### 奖励计算公式

**线速度跟踪（指数形式）:**
```python
reward = exp(-((vel_error)^2 / (2 * std^2)) * weight
```

**角速度跟踪（指数形式）:**
```python
reward = exp(-((ang_vel_error)^2 / (2 * std^2)) * weight
```

**关节位置惩罚（站立姿态）:**
```python
penalty = scale * sum(|joint_pos - default_pos|^2)
```

### 终止条件

| 条件 | 说明 |
|-------|------|
| `time_out` | 超过最大回合时间（20秒） |
| `base_contact` | 基础与地面接触 |
| `bad_orientation` | 姿态角度超过阈值（0.8弧度） |

---

## 监控和调试

### TensorBoard 监控

```bash
# 启动 TensorBoard
tensorboard --logdir logs/rsl_rl/

# 在浏览器中打开
# http://localhost:6006
```

### TensorBoard 指标说明

**训练指标:**
- `Train/mean_reward`: 平均奖励
- `Train/mean_episode_length`: 平均回合长度
- `Train/mean_success_rate`: 平均成功率

**策略指标:**
- `Policy/mean_action_mean`: 平均动作
- `Policy/mean_action_std`: 动作标准差
- `Policy/mean_value_mean`: 平均价值估计

**损失指标:**
- `Loss/value_function`: 价值损失
- `Loss/surrogate`: 策略替代损失
- `Loss/entropy`: 策略熵

### 日志文件位置

训练日志保存在：
```
logs/rsl_rl/{experiment_name}/{timestamp}/
```

包含：
- `params/agent.yaml`: 智能体配置
- `params/env.yaml`: 环境配置
- `model_*.pt`: 模型检查点
- `git/`: Git 差异信息

### 视频录制

启用视频录制后，视频保存在：
```
logs/rsl_rl/{experiment_name}/{timestamp}/videos/train/
```

### 调试技巧

**1. 减少环境数量:**
```bash
--num_envs 512  # 减少显存使用
```

**2. 启用 GUI 可视化:**
```bash
--gui  # 观察仿真过程
```

**3. 检查配置:**
```bash
# 验证配置文件
python scripts/verify_config.py --task Unitree-Go2W-Velocity-Flat-v0
```

**4. 逐步训练:**
```bash
--max_iterations 1000  # 先训练少量迭代验证
```

**5. 启用详细日志:**
```bash
export YDRA_FULL_ERROR=1  # 完整错误日志
```

---

## 常见问题

### 1. 内存不足

**问题:** `RuntimeError: CUDA out of memory`

**解决方案:**
- 减少环境数量：`--num_envs 2048`
- 减少 batch size：修改配置文件中的 `num_mini_batches`
- 关闭不必要应用程序释放显存

### 2. Isaac Sim 崩溃

**问题:** Isaac Sim 仿真器意外退出

**解决方案:**
- 确保驱动程序更新：`nvidia-smi`
- 检查 Isaac Sim 版本兼容性
- 使用 `--headless` 模式
- 重启 Isaac Sim 进程

### 3. 训练不收敛

**问题:** 奖励曲线不上升或震荡

**解决方案:**
- 检查奖励函数权重是否合理
- 调整学习率：`learning_rate=5e-4`
- 增加 entropy_coef：`entropy_coef=0.02`
- 检查观察和动作空间是否正确
- 查看 TensorBoard 中的损失曲线

### 4. 机器人倒伏频繁

**问题:** 机器人经常倒伏或翻滚

**解决方案:**
- 增加 `flat_orientation_l2` 奖励权重
- 调整 `base_angular_velocity` 惩罚权重
- 检查命令速度范围是否合理
- 增加 `bad_orientation` 终止阈值

### 5. 动作过于抖动

**问题:** 动作连续变化，机器人抖动

**解决方案:**
- 增加 `action_rate_l2` 惩罚权重：`-0.2`
- 增加 `joint_acc_l2` 惩罚权重
- 减少 `clip_param`：`0.15`
- 调整动作缩放：修改 `ActionsCfg` 中的 `scale`

### 6. 配置文件错误

**问题:** `TypeError: MLPModel.__init__() got an unexpected keyword argument 'stochastic'`

**原因:** RSL-RL 5.0+ 不再支持 `stochastic` 参数

**解决方案:**
- 已在配置文件中使用 `distribution_cfg` 替代
- 确保 `actor` 配置包含 `distribution_cfg` 参数
- 示例见 `rsl_rl_ppo_cfg.py`

### 7. 环境初始化失败

**问题:** `AttributeError: 'NoneType' object has no attribute 'log_prob'`

**原因:** Actor 没有配置 distribution

**解决方案:**
- 确保配置文件中 actor 包含 `distribution_cfg`
- 检查 `handle_deprecated_rsl_rl_cfg` 是否正确调用
- 见：[scripts/rsl_rl/train.py:107](scripts/rsl_rl/train.py#L107) 和 [:131](scripts/rsl_rl/train.py#L131)

### 8. TensorBoard 无数据

**问题:** TensorBoard 显示空图表

**解决方案:**
- 检查日志目录路径：`logs/rsl_rl/`
- 确认训练进程正常运行
- 重新加载 TensorBoard：`Ctrl+C` 然后重新启动

---

## 高级配置

### 自定义地形

修改 `velocity_env_cfg.py` 中的 `COBBLESTONE_ROAD_CFG`:

```python
COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),           # 地形大小
    num_rows=10,                  # 行数
    num_cols=20,                  # 列数
    horizontal_scale=0.1,          # 水平缩放
    vertical_scale=0.005,            # 垂直缩放
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.1),
        # 添加更多地形类型
    },
)
```

### 自定义奖励

修改 `velocity_env_cfg.py` 中的 `RewardsCfg`:

```python
rewards = RewardsCfg(
    # 添加自定义奖励项
    custom_reward = RewTerm(
        func=mdp.custom_function,
        weight=1.0,
        params={...}
    )
)
```

### 自定义观察

修改 `velocity_env_cfg.py` 中的 `ObservationsCfg`:

```python
observations = ObservationsCfg(
    policy = ObservationsCfg.PolicyCfg(
        # 添加自定义观察项
        custom_obs = ObsTerm(
            func=mdp.custom_observation,
            scale=1.0,
            noise=Unoise(n_min=-0.01, n_max=0.01)
        )
    )
)
```

### 分布式训练

```bash
# 多 GPU 训练
./scripts/train_go2w.sh flat \
    --distributed \
    --num_envs 8192

# 使用特定 GPU
CUDA_VISIBLE_DEVICES=0,1 ./scripts/train_go2w.sh flat --distributed
```

---

## 部署和推理

### 保存部署配置

训练完成后，部署配置自动保存到：
```
logs/rsl_rl/{experiment_name}/{timestamp}/deploy/
```

### 加载和运行

```bash
# 使用训练脚本运行
./scripts/train_go2w.sh play-flat --load_run {timestamp}

# 或直接使用 Python
python scripts/rsl_rl/play.py \
    --task Unitree-Go2W-Velocity-Flat-v0 \
    --checkpoint logs/rsl_rl/{experiment_name}/{timestamp}/model_{iteration}.pt
```

### 导出 ONNX 模型

```python
from isaaclab_rl.rsl_rl.exporter import export_policy_with_onnx

export_policy_with_onnx(
    checkpoint_path="logs/rsl_rl/.../model_10000.pt",
    onnx_path="policy.onnx",
    obs_shape=(41,),  # 观察维度
    act_shape=(16,),  # 动作维度
)
)
```

---

## 性能优化建议

### 训练速度优化

1. **增加环境数量** (受限于 GPU 显存)
   - 4096 环境适合 24GB 显存
   - 8192 环境适合 40GB+ 显存

2. **调整 decimation**
   - 减少物理更新：`decimation=2` (更精确但更慢)
   - 增加物理更新：`decimation=8` (更快但可能不稳定)

3. **使用多 GPU**
   - 启用分布式训练：`--distributed`
   - 注意：需要足够的通信带宽

### 仿真质量优化

1. **提高物理精度**
   ```python
   self.sim.dt = 0.002  # 更小时间步长
   self.decimation = 2     # 相应减少 decimation
   ```

2. **增加接触传感器频率**
   ```python
   self.scene.contact_forces.update_period = self.sim.dt  # 每步更新
   ```

3. **改进地形质量**
   ```python
   horizontal_scale=0.05,   # 更精细的纹理
   vertical_scale=0.002,     # 更平滑的高度变化
   ```

---

## 参考资源

### 官方文档

- [Isaac Lab 文档](https://isaac-sim.github.io/IsaacLab/main/index.html)
- [RSL-RL GitHub](https://github.com/leggedrobotics/rsl_rl)
- [Unitree 官方网站](https://www.unitree.com/)

### 社区资源

- [Isaac Lab 论坛](https://forum.omniverse.nvidia.com/c/isaac-lab)
- [RSL-RL Issues](https://github.com/leggedrobotics/rsl_rl/issues)

### 论文引用

如果在研究中使用本框架，请引用：

```bibtex
@article{rsl_rl_2024,
  title={RSL-RL: A Minimalist Library for Legged Robot Learning},
  author={Rudin et al.},
  journal={...},
  year={2024}
}
```

---

## 许可证

本项目遵循 BSD-3-Clause 许可证。详见 [LICENSE](LICENSE) 文件。

---

## 更新日志

### v1.0.0 (2026-03-26)
- 初始版本
- 支持 GO2W 机器人 Flat 和 Rough 地形训练
- 集成 RSL-RL 5.0.1
- 修复 distribution 配置兼容性问题
