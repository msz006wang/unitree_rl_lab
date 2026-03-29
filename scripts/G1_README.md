# G1 Training Scripts and Documentation

G1机器人训练脚本和文档。

## 📋 目录

1. [训练脚本](#训练脚本)
2. [快速开始](#快速开始)
3. [配置文件](#配置文件)
4. [故障排除](#故障排除)

---

## 训练脚本

### 主要训练脚本

**[train_g1.sh](train_g1.sh)** ⭐ **主要训练脚本**
- 功能最全面，支持所有训练模式
- 支持平地和16级渐进式地形
- 支持原始和改进配置
- 命令行参数解析

### 训练模式

| 模式 | 命令 | 说明 |
|------|------|------|
| 原始配置 | `./scripts/train_g1.sh original` | 16级渐进式地形训练 |
| 改进配置 | `./scripts/train_g1.sh improved` | 16级渐进式地形 + 摔倒恢复 |
| 平地原始 | `./scripts/train_g1.sh flat-original` | 平地原始配置训练 |
| 平地改进 | `./scripts/train_g1.sh flat-improved` | 平地改进配置训练 ⭐ |
| 回放原始 | `./scripts/train_g1.sh play-original` | 回放原始策略 |
| 回放改进 | `./scripts/train_g1.sh play-improved` | 回放改进策略 |

---

## 快速开始

### 推荐训练命令（30分钟测试）

```bash
# 平地改进配置训练（推荐）
./scripts/train_g1.sh flat-improved --num_envs 512
```

### 其他常用选项

```bash
# 原始配置完整训练
./scripts/train_g1.sh original --num_envs 4096 --headless

# 改进配置完整训练
./scripts/train_g1.sh improved --num_envs 4096 --headless

# 带GUI可视化训练
./scripts/train_g1.sh flat-improved --gui --video

# 恢复训练
./scripts/train_g1.sh improved --resume

# 使用自定义设备
./scripts/train_g1.sh flat-improved --device cuda:1 --num_envs 512
```

---

## 配置文件

### G1配置文件结构

```
source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/
├── velocity_env_cfg.py              # 原始16级渐进式配置
├── velocity_env_cfg_improved.py    # 改进16级渐进式配置 ⭐
├── velocity_env_cfg_flat.py          # 原始平地配置
├── velocity_env_cfg_flat_improved.py # 改进平地配置 ⭐
├── actions_cfg.py                   # 动作配置
└── __init__.py                      # 任务注册
```

### 配置模式对比

| 配置类型 | 地形 | Episode长度 | Action Scale | 奖励函数 |
|----------|------|-------------|-------------|-----------|
| 原始配置 | 16级渐进式 | 20.0s | 0.3 | 基础 |
| 改进配置 | 16级渐进式 | 25.0s | 0.35 | 基础+扩展 |
| 平地原始 | 平地 | 20.0s | 0.3 | 基础 |
| 平地改进 | 平地 | 25.0s | 0.35 | 基础+扩展 |

---

## 故障排除

### 常见问题

#### Q: 训练在25个iteration内结束？
**A:** 使用平地配置重新测试，已修复初始高度和奖励权重

#### Q: 机器人频繁摔倒？
**A:** 检查终止条件设置，考虑使用更严格的参数

#### Q: TensorBoard无法访问？
**A:** 尝试不同端口 (6006, 6007)，清理TensorBoard缓存

#### Q: 导入错误 "No module named 'pxr'"?
**A:** 这是Isaac Lab依赖问题，训练应该可以正常进行，直接使用训练命令

### 验证工具

```bash
# 验证配置文件
./scripts/validate_improved_config.sh

# 测试导入
python test_g1_flat_import.py
```

### 监控工具

```bash
# TensorBoard监控
tensorboard --logdir logs/rsl_rl/

# 查看G1训练
tensorboard --logdir logs/rsl_rl/unitree_g1_29dof_velocity_improved/
```

---

## 文档参考

- **[完整训练指南](../docs/G1_TRAINING_GUIDE.md)** - 详细的训练说明
- **[快速参考手册](../docs/G1_QUICK_REFERENCE.md)** - 快速命令和故障排除

---

## 配置建议

### 推荐训练流程

#### 步骤1: 平地快速测试
```bash
./scripts/train_g1.sh flat-improved --num_envs 512
```

#### 步骤2: 完整平地训练
```bash
./scripts/train_g1.sh flat-improved --num_envs 4096
```

#### 步骤3: 16级渐进式训练
```bash
./scripts/train_g1.sh improved --num_envs 8192
```

### 配置选择建议

| 目标 | 推荐配置 | 原因 |
|------|-----------|--------|
| 基础步态验证 | flat-original | 简单环境，专注步态 |
| 稳定性测试 | flat-improved | 优化权重，更稳定 |
| 摔倒恢复训练 | improved | 包含摔倒恢复功能 |
| 复杂地形适应 | improved | 16级渐进式地形 |
| 长期训练 | improved | 最大训练容量 |

---

## 快速命令参考

### 最常用命令

```bash
# 快速平地测试（推荐）
./scripts/train_g1.sh flat-improved --num_envs 512

# 平地完整训练
./scripts/train_g1.sh flat-improved --num_envs 4096

# 原始配置训练
./scripts/train_g1.sh original

# 改进配置训练
./scripts/train_g1.sh improved

# 回放训练好的策略
./scripts/train_g1.sh play-improved --load_run recent
```

---

## 总结

**主要特性:**
- ✅ 完整的G1训练脚本系统
- ✅ 支持平地和16级渐进式地形
- ✅ 原始和改进配置选项
- ✅ 优化的奖励权重和参数
- ✅ 完整的文档和故障排除指南

**开始训练:**
```bash
./scripts/train_g1.sh flat-improved --num_envs 512
```

**推荐快速开始:**
- 查看 [G1_TRAINING_GUIDE.md](../docs/G1_TRAINING_GUIDE.md) 了解详细说明
- 查看 [G1_QUICK_REFERENCE.md](../docs/G1_QUICK_REFERENCE.md) 获取快速参考
- 使用 `flat-improved` 模式进行初始训练

**准备训练了吗?** ✅ 开始吧！
