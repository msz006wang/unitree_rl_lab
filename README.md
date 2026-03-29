# Unitree RL Lab

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.3.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![License](https://img.shields.io/badge/License-Apache-2.0-yellow.svg)](https://opensource.org/licenses/Apache-2.0)
[![Discord](https://img.shields.io/badge/Discord/5865F2?style=flat&logo=Discord&logoColor=white)](https://discord.gg/ZwcVwxv5rq)


## Overview

This project provides a set of reinforcement learning environments for Unitree robots, built on top of [IsaacLab](https://github.com/isaac-sim/IsaacLab).

Currently supports Unitree **Go2**, **H1** and **G1-29dof** robots.

<div align="center">

| <div align="center"> Isaac Lab </div> | <div align="center"> Mujoco </div> | <div align="center"> Physical </div> |
|--- | --- | --- |
| [<img src="https://oss-global-cdn.unitree.com/static/d879adac250648c587de3681e90658b49_480x397.gif" width="240px">](g1_sim.gif) | [<img src="https://oss-global-cdn.unitree.com/static/3c88e045ab124c3ab9c761a99cb5e71f_480x397.gif" width="240px">](g1_mujoco.gif) | [<img src="https://oss-global-cdn.unitree.com/static/6c17c6cf52ec4e26bbfab1fbf591adb2_480x270.gif" width="240px">](g1_real.gif) |

</div>


## G1 Robot Training

**📖 [Complete Training Guide](docs/G1_TRAINING_GUIDE.md)** - 详细的G1训练说明
**📖 [Quick Reference](docs/G1_QUICK_REFERENCE.md)** - G1快速参考手册
**📖 [G1 Scripts README](scripts/G1_README.md)** - G1训练脚本文档

### Quick Start

**Recommended Training Command:**
```bash
# Flat terrain with improved configuration (recommended)
./scripts/train_g1.sh flat-improved --num_envs 512
```

**Available Training Modes:**
| Mode | Task Name | Terrain Type | Description |
|------|----------|-------------|-----------|
| original | Unitree-G1-29dof-Velocity | 16-level progressive | Standard 16-level progressive terrain training |
| improved | Unitree-G1-29dof-Velocity-Improved | 16-level progressive | Enhanced training with fall recovery |
| flat-original | Unitree-G1-29dof-Velocity-Flat | Plane | Original config on flat terrain |
| flat-improved | Unitree-G1-29dof-Velocity-Flat-Improved | Plane | Enhanced config on flat terrain ⭐ |

### Configuration Features

| Feature | Original | Improved |
|---------|-----------|-----------|
| Episode Length | 20.0s | 25.0s |
| Action Scale | 0.3 | 0.35 (optimized for stability) |
| Init Height | 0.8m | 0.65m (more stable) |
| Fall Recovery | No | Yes (with rewards) |
| Extended Rewards | No | Yes (survival, distance, energy, etc.) |
| Terrain | 16-level progressive | 16-level progressive + flat options |

### Key Improvements

1. **Stability Fixes**
   - ✅ Reduced initial height (0.8m → 0.65m)
   - ✅ Optimized reward weights (avoid conflicting objectives)
   - ✅ Adjusted action scale (0.5 → 0.35)
   - ✅ Tightened termination conditions for better stability

2. **New Training Modes**
   - ✅ Flat terrain options for basic gait training
   - ✅ Simplified configuration files for easier debugging
   - ✅ Complete documentation system

3. **Documentation**
   - ✅ Comprehensive training guide
   - ✅ Quick reference manual
   - ✅ Troubleshooting guide

### Quick Reference

**Training Commands:**
```bash
# Quick test (recommended)
./scripts/train_g1.sh flat-improved --num_envs 512

# Complete training
./scripts/train_g1.sh flat-improved --num_envs 4096

# With GUI
./scripts/train_g1.sh flat-improved --gui --video
```

### Key Differences: G1 vs GO2W

| Feature | G1 | GO2W | Impact |
|------|-------|-----|--------|
| Robot Type | Humanoid | Wheeled | G1 is more complex |
| Terrain | 16-level progressive | Flat + Rough | G1 supports more complex terrain |
| Training Stability | Base stability | High | G1 requires more tuning |
| Recovery | No | No | G1 supports fall recovery |

---

## Installation

Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
Install Unitree RL IsaacLab standalone environments.

- Clone or copy this repository separately from Isaac Lab installation:
    ```bash
    git clone https://github.com/unitreerobotics/unitree_rl_lab.git
    ```

- Use a python interpreter that has Isaac Lab installed, install library in editable mode using:
    ```bash
    conda activate env_isaaclab
    ./unitree_rl_lab.sh -i
    # restart your shell to activate environment changes.
    ```

- Download unitree robot description files from [G1 Training Guide](docs/G1_TRAINING_GUIDE.md).

---

## Getting Started

**Recommended First Training Command:**
```bash
./scripts/train_g1.sh flat-improved --num_envs 512
```

**Ready to train?**
- ✅ Check environment: `conda activate env_isaaclab`
- ✅ Validate configuration: `./scripts/validate_improved_config.sh`
- ✅ Start training: `./scripts/train_g1.sh flat-improved --num_envs 512`

**Start training!** 🚀
