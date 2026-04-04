# 环境ID错误修复说明

## 问题描述

执行训练脚本时出现以下错误：

```
gymnasium.error.NameNotFound: Environment `go2w_arm_two_stage_recovery` doesn't exist.
```

## 根本原因

训练脚本使用了错误的环境ID `go2w_arm_two_stage_recovery`，但该环境在 Gymnasium 中注册的ID是 `Unitree-Go2WArm-TwoStage-Recovery-v0`。

## 正确的环境ID

根据 `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2w_arm/__init__.py` 文件中的注册信息，可用的环境ID如下：

1. **Unitree-Go2WArm-Velocity-Flat-v0** - 平地速度控制训练
2. **Unitree-Go2WArm-Velocity-Rough-v0** - 复杂地形速度控制训练
3. **Unitree-Go2WArm-Velocity** - 速度控制训练（向后兼容）
4. **Unitree-Go2WArm-TwoStage-Recovery-v0** - 两段式恢复训练 🎯

## 解决方案

已修复以下脚本文件，将错误的环境ID更正为 `Unitree-Go2WArm-TwoStage-Recovery-v0`：

### 修复的文件列表

- [x] scripts/train_go2w_arm_two_stage.sh
- [x] scripts/train_fixed.py
- [x] scripts/train_direct.sh
- [x] scripts/train_working.sh
- [x] scripts/train_minimal.sh
- [x] scripts/diagnose_isaac_sim.sh
- [x] scripts/train_test.sh

### 修复内容

所有脚本中的环境ID从：
```bash
TASK_NAME="go2w_arm_two_stage_recovery"
```

更正为：
```bash
TASK_NAME="Unitree-Go2WArm-TwoStage-Recovery-v0"
```

## 使用方法

### 方法1：使用训练脚本（GUI 模式）

```bash
./scripts/train_go2w_arm_two_stage.sh
```

### 方法1.5：使用训练脚本（无头模式）

```bash
./scripts/train_go2w_arm_two_stage.sh --headless
```

### 方法2：直接使用 Python 脚本（GUI 模式）

```bash
python3 scripts/train_fixed.py \
  --task Unitree-Go2WArm-TwoStage-Recovery-v0 \
  --num_envs 4096
```

### 方法2.5：直接使用 Python 脚本（无头模式）

```bash
python3 scripts/train_fixed.py \
  --task Unitree-Go2WArm-TwoStage-Recovery-v0 \
  --headless \
  --num_envs 4096
```

### 方法3：使用测试脚本（少量环境）

```bash
./scripts/train_test.sh
```

## 验证环境ID

运行以下脚本查看所有已注册的环境：

```bash
python3 scripts/verify_env_id.py
```

输出示例：

```
============================================================
GO2W ARM 环境注册信息
============================================================

✅ 已注册的环境ID:
  1.    Unitree-Go2WArm-Velocity-Flat-v0
  2.    Unitree-Go2WArm-Velocity-Rough-v0
  3.    Unitree-Go2WArm-Velocity
  4. 🎯 Unitree-Go2WArm-TwoStage-Recovery-v0
```

## 常见问题

### Q: 为什么环境ID要遵循这种命名规范？

A: Gymnasium 环境注册遵循 `NameSpace-Task-Type-Version` 的格式，这是标准的命名约定，便于区分不同的环境和版本。

### Q: 如果忘记正确的环境ID怎么办？

A: 运行 `python3 scripts/verify_env_id.py` 查看所有可用的环境ID。

### Q: 旧的环境ID还能用吗？

A: 不能。必须使用注册表中存在的环境ID，否则会抛出 `NameNotFound` 错误。

## 相关文件

- 环境注册文件：[go2w_arm/__init__.py](../source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2w_arm/__init__.py)
- 环境配置文件：[two_stage_recovery_env_cfg.py](../source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2w_arm/two_stage_recovery_env_cfg.py)
