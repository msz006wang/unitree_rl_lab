# GO2W ARM 快速索引

## 🚀 快速开始

### 验证代码
```bash
python scripts/verify_code_syntax.py
```

### 开始训练
```bash
python scripts/train.py --task Robot-v0
```

### 监控训练
```bash
python scripts/start_tensorboard.sh
```

---

## 📋 关键修改摘要

### 1️⃣ 机械臂策略
- **夹紧状态**：arm_joint2-6完全折叠（0.0）
- **根部旋转**：arm_joint1可调整，辅助平衡
- **动作空间**：腿部 + arm_joint1

### 2️⃣ 新增奖励（6个）

| 奖励函数 | 权重 | 作用 |
|---------|-------|------|
| upward_velocity | 2.0 | 鼓励Z轴向上速度 |
| orientation_tracking | 1.5 | 奖励直立姿态 |
| torque_penalty | -0.01 | 惩罚持续高扭矩 |
| joint_regularization | -0.5 | 避免关节限位 |
| contact_management | -0.3 | 惩罚非足端接触 |
| wheel_assisted_recovery | 0.5 | 轮足协同奖励 |

### 3️⃣ 历史观测（2个）
- **joint_pos_history**：10帧关节位置
- **body_vel_history**：10帧身体速度

### 4️⃣ 动作缩放
- **髋关节**：0.125
- **arm_joint1**：0.1（小范围）
- **其他关节**：0.25

---

## 🔧 调试检查清单

### ✅ 训练前
- [ ] 代码语法验证通过
- [ ] 机械臂初始姿态为折叠状态
- [ ] arm_joint1在动作空间中
- [ ] arm_joint2-6不在动作空间中
- [ ] 历史观测已配置

### ✅ 训练中（TensorBoard）
- [ ] upward_velocity > 0（有向上速度）
- [ ] orientation_tracking > 0.8（姿态较直）
- [ ] torque_penalty 较小（不持续过载）
- [ ] contact_management 接近0（无非足端接触）
- [ ] episode_length 增加（稳定性提升）

### ⚠️ 常见问题

**机器人无法站立？**
→ 增加upward_velocity权重（2.0→3.0）
→ 增加orientation_tracking权重（1.5→2.5）

**机械臂摆动干扰？**
→ 检查arm_joint2-6是否在动作空间
→ 减小arm_joint1的scale（0.1→0.05）

**扭矩过高？**
→ 增加sustained_window（2.0→3.0）
→ 增加burst_threshold（1.5→2.0）

**轮子不协同？**
→ 增加wheel_assisted_recovery权重（0.5→1.0）
→ 检查轮子是否在动作空间

---

## 📊 训练曲线目标

### 前期（0-1M steps）
- ✅ 奖励平稳上升
- ✅ episode_length增加
- ✅ 无NaN或崩溃

### 中期（1-3M steps）
- ✅ upward_velocity激活频率增加
- ✅ orientation_tracking接近1.0
- ✅ 恢复成功率>50%

### 后期（3-5M steps）
- ✅ 平均episode_length>15秒
- ✅ 恢复成功率>80%
- ✅ 能耗优化

---

## 📁 重要文件位置

### 代码文件
```
source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/
├── extended_rewards.py       # 新增6个奖励函数
├── observations.py          # 新增3个观测函数
└── __init__.py            # 导出新函数

source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2w_arm/
└── velocity_env_cfg.py     # 配置文件修改
```

### 文档文件
```
docs/
├── GO2W_ARM_算法策略.md              # 完整算法策略说明
├── GO2W_ARM_参数优化.md              # 完整参数优化说明
└── GO2W_ARM_快速索引.md              # 本文档 - 快速参考
```

### 脚本文件
```
scripts/
├── verify_code_syntax.py                    # 语法验证
├── verify_comprehensive_optimization.py       # 完整验证（需依赖）
├── train_go2w_arm.sh                      # 训练脚本
└── start_tensorboard_latest.sh              # TensorBoard监控
```

---

## 🎯 核心策略说明

### 不倒翁效应
利用转动惯量实现"弹起"恢复：
1. 产生向上动量（upward_velocity奖励）
2. 身体后仰/前倾转换（orientation_tracking引导）
3. 爆发蹬地起跳

### 人类起坐类比
类似人类从坐姿站起的模式：
1. 腿部蓄力（膝关节弯曲）
2. 臀部抬起（身体向上）
3. 腿部伸展（完成站立）

### 轮足协同
在侧卧时利用轮子：
1. 轮子旋转产生摩擦力
2. 改变地面接触点
3. 辅助姿态转换

---

## 🔑 关键参数说明

### 扭矩惩罚参数
- **sustained_window=2.0**：持续2秒以上才惩罚
- **burst_threshold=1.5**：允许1.5倍额定扭矩
- **decay_rate=0.9**：历史扭矩的衰减率
- **rated_torque=23.5**：电机额定扭矩

### 关节正则化参数
- **soft_ratio=0.95**：距离限位95%时开始惩罚
- 预留5%缓冲空间

### 历史观测参数
- **buffer_length=10**：存储10帧历史
- **循环缓冲**：高效内存使用

---

## 📞 问题排查流程

```
问题发生
    ↓
检查TensorBoard日志
    ↓
定位问题类型
    ↓
参考调优建议
    ↓
修改配置参数
    ↓
重新训练验证
```

---

## 🎓 学习资源

### 理论基础
- **不倒翁原理**：转动惯量和重心位置
- **动量守恒**：向上动量利用
- **摩擦力**：轮子与地面摩擦

### 实现细节
- **历史缓冲**：循环队列实现
- **指数移动平均**：平滑扭矩检测
- **姿态表示**：投影重力向量

### 调试技巧
- **TensorBoard**：实时监控
- **可视化**：启用debug_vis
- **日志分析**：检查奖励权重

---

## 📊 性能指标参考

### 短期目标（5千步）

| 指标 | 目标值 | 验证方法 |
|------|----------|----------|
| Mean Reward | > -3 | tensorboard监控 |
| upright_bonus | > 2.0 | tensorboard监控 |
| 站立恢复成功率 | > 80% | 行为观察 |
| 翻倒率 | < 10% | episode统计 |

### 中期目标（20千步）

| 指标 | 目标值 | 验证方法 |
|------|----------|----------|
| Mean Reward | > 0 | tensorboard监控 |
| track_lin_vel | > 0.88 | tensorboard监控 |
| 静态站立时间 | > 30s | 行为观察 |
| 机械臂全程保持折叠 | 是 | TensorBoard验证 |

### 长期目标（50千步）

| 指标 | 目标值 | 验证方法 |
|------|----------|----------|
| Mean Reward | > 5 | tensorboard监控 |
| track_lin_vel | > 0.92 | tensorboard监控 |
| 翻倒率 | < 2% | episode统计 |
| 完美的平衡控制 | 是 | 行为观察 |

---

## 🎛️ 配置验证清单

### 初始状态配置
- [ ] 位置范围：x/y: (-0.2, 0.2), z: (0.35, 0.5)
- [ ] 姿态角度：roll/pitch: (-0.3, 0.3), yaw: (-3.14, 3.14)
- [ ] 初始速度：线速度±0.1 m/s, 角速度±0.1 rad/s

### 机械臂配置
- [ ] arm_joint1: 0.0 (在动作空间中)
- [ ] arm_joint2-6: 0.0 (不在动作空间中)
- [ ] 机械臂完全折叠状态

### 扭矩配置
- [ ] 腿部关节：35.0 N·m
- [ ] 轮子关节：35.0 N·m
- [ ] 机械臂1-3：25.0 N·m
- [ ] 机械臂4-6：15.0 N·m

### 奖励配置
- [ ] upright_velocity: 2.0
- [ ] orientation_tracking: 1.5
- [ ] torque_penalty: -0.01
- [ ] joint_regularization: -0.5

---

## 🔍 TensorBoard监控指南

### 关键奖励指标

#### 运动奖励
- `rewards/track_lin_vel_xy_exp` - 线速度追踪（目标 > 2.5）
- `rewards/track_ang_vel_z_exp` - 角速度追踪（目标 > 1.2）
- `rewards/upward_velocity` - 向上速度（目标 > 1.0）

#### 姿态奖励
- `rewards/orientation_tracking` - 姿态稳定性（目标 > 0.8）
- `rewards/base_height_l2` - 高度控制（目标接近目标值）

#### 稳定性奖励
- `rewards/torque_penalty` - 扭矩使用情况（目标 < -0.005）
- `rewards/joint_regularization` - 关节限位保护（目标 < 0.1）
- `rewards/contact_management` - 非足端接触（目标接近0）

### 关键性能指标

- `episode_length` - 回合长度（目标 > 1000）
- `mean_reward` - 总体奖励（目标 > 0）
- `success_rate` - 任务完成率（目标 > 80%）

---

## 🚀 训练启动步骤

### 步骤1：验证配置
```bash
# 检查配置文件导入
python -c "
import sys
import os
os.chdir('/home/jay/unitree_rl_lab/source')
from unitree_rl_lab.tasks.locomotion.robots.go2w_arm.velocity_env_cfg import RobotFlatEnvCfg
cfg = RobotFlatEnvCfg()
print('✅ 配置导入成功')
print(f'upward_velocity权重: {cfg.rewards.upward_velocity.weight}')
print(f'torque_penalty权重: {cfg.rewards.torque_penalty.weight}')
print(f'joint_regularization权重: {cfg.rewards.joint_regularization.weight}')
" 2>&1 | head -20
```

### 步骤2：启动训练
```bash
# 启动训练
./scripts/train_go2w_arm.sh arx5_flat
```

### 步骤3：监控训练
```bash
# 启动TensorBoard
./scripts/start_tensorboard_latest.sh

# 或监控最新训练
./scripts/start_tensorboard.sh
```

### 步骤4：查看实时日志
```bash
# 查看训练日志
tail -f logs/rsl_rl/Unitree-Go2WArm-Velocity-Flat-v0/log.txt
```

---

## 📝 调优参考表

### 奖励权重调优

#### 如果机器人过度静止
| 奖励项 | 原始值 | 建议值 | 调优理由 |
|---------|--------|----------|----------|
| upward_velocity | 2.0 | 3.0 | 增强向上激励 |
| track_lin_vel_xy_exp | 4.5 | 5.5 | 提高运动积极性 |

#### 如果机器人控制振荡
| 奖励项 | 原始值 | 建议值 | 调优理由 |
|---------|--------|----------|----------|
| action_rate_l2 | -0.001 | -0.0005 | 降低惩罚强度 |
| joint_acc_l2 | -1.0e-7 | -0.5e-7 | 允许更快的加速度 |

#### 如果扭矩使用过高
| 参数 | 原始值 | 建议值 | 调优理由 |
|------|--------|----------|----------|
| sustained_window | 2.0 | 3.0 | 延长允许时间 |
| burst_threshold | 1.5 | 1.3 | 降低瞬发阈值 |

### 算法参数调优

#### 如果学习过慢
| 参数 | 原始值 | 建议值 | 调优理由 |
|------|--------|----------|----------|
| learning_rate | 2.0e-4 | 5.0e-4 | 加快学习速度 |

#### 如果探索过度
| 参数 | 原始值 | 建议值 | 调优理由 |
|------|--------|----------|----------|
| entropy_coef | 0.005 | 0.003 | 减少随机探索 |

---

## 🏃 快速恢复策略

### 从不同初始状态恢复

#### 从轻微倾斜（±17°）
- **策略**：使用upward_velocity快速蹬地
- **预期**：恢复时间 < 0.5秒
- **成功率**：> 95%

#### 从中度倾斜（±30°）
- **策略**：先调整姿态，再蹬地起跳
- **预期**：恢复时间 < 1.0秒
- **成功率**：> 80%

#### 从侧卧状态
- **策略**：利用wheel_assisted_recovery + 腿部蹬地
- **预期**：恢复时间 < 2.0秒
- **成功率**：> 60%

---

## 📚 相关文档

### 核心文档
- **GO2W_ARM_算法策略.md** - 完整的算法和策略设计
- **GO2W_ARM_参数优化.md** - 详细的参数调整记录
- **GO2W_ARM_快速索引.md** - 本文档

### 辅助文档
- **TENSORBOARD_GUIDE.md** - TensorBoard使用指南
- **TRAINING_SCRIPT_GUIDE.md** - 训练脚本使用指南
- **TRAINING_ERROR_ANALYSIS.md** - 训练错误分析

---

## 💡 常用命令

### 训练相关
```bash
# 启动训练
./scripts/train_go2w_arm.sh arx5_flat

# 监控最新训练
./scripts/start_tensorboard_latest.sh

# 检查训练进程
ps aux | grep train
```

### 调试相关
```bash
# 验证配置语法
python -m py_compile velocity_env_cfg.py

# 检查初始状态
python scripts/check_initial_state.py

# 验证完整优化
python scripts/verify_comprehensive_optimization.py
```

### 查看结果
```bash
# 查看TensorBoard
tensorboard --logdir logs/rsl_rl/

# 查看训练日志
tail -f logs/rsl_rl/Unitree-Go2WArm-Velocity-Flat-v0/log.txt

# 检查checkpoint
ls -lh logs/rsl_rl/Unitree-Go2WArm-Velocity-Flat-v0/
```

---

## 🎯 核心指标监控

### 每日检查
- Mean Reward趋势
- Episode Length变化
- 各奖励项贡献

### 每周检查
- 站立恢复成功率
- 速度追踪精度
- 机械臂稳定性

### 每月检查
- 整体训练效果
- 参数调优效果
- 鲁棒性测试结果

---

## ⚙️ 环境配置

### 硬件要求
- GPU: 至少8GB显存
- CPU: 至少8核
- RAM: 至少32GB
- 磁盘: 至少100GB可用空间

### 软件要求
- Isaac Lab: 最新稳定版
- PyTorch: 2.0+
- Python: 3.8+

---

## 📞 支持与帮助

### 问题排查顺序
1. 查看本文档的常见问题部分
2. 检查TensorBoard日志
3. 参考参数调优表
4. 查看核心文档详细说明

### 获取帮助
- 检查训练日志中的错误信息
- 使用验证脚本诊断问题
- 参考相关文档查找解决方案

---

**最后更新**: 2026-04-01
**版本**: v1.0.0
**状态**: ✅ 完成并验证
