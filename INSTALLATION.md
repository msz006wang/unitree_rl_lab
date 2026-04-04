# GO2W Wheel-Legged Robot RL Project - Installation Guide
# Updated based on conda environment env_isaaclab installation
# Last Updated: 2026-04-03

## 📋 Dependencies Overview

### 🔧 Environment Status

| Environment | Python Version | Isaac Sim Version | Status |
|------------|---------------|------------------|--------|
| env_isaaclab | 3.11 | 5.1.0-rc.19+release.26219.9c81211b.gl (~5.1.0) | ✅ **就绪** |

### 📦 Core Dependencies (from env_isaaclab)

| Dependency | Installed Version | Installation Method | Purpose | Status |
|-----------|------------------|-----------|--------|
| torch | 2.7.0+cu128 | pip | Deep Learning Framework (CUDA 12.x) | ✅ |
| gymnasium | 1.2.1 | pip | RL Environment API | ✅ |
| isaaclab | 0.54.3 | pip (editable) | Isaac Lab Core Framework | ✅ |
| isaaclab_assets | 0.2.4 | pip (editable) | Isaac Lab Assets | ✅ |
| isaaclab_tasks | 0.11.14 | pip (editable) | Isaac Lab Tasks | ✅ |
| isaaclab_rl | 0.5.0 | pip (editable) | Isaac Lab RL Algorithms | ✅ |

**Pre-installed Isaac Sim Components**：
- numpy, matplotlib, warp-lang (scientific computing)
- Isaac Sim core modules, extensions, toolkits

**Note about Isaac Lab packages**：
- isaaclab packages are editable, meaning you can modify them
- Project location: /home/jay/IsaacLab/source/isaaclab

### 📂 System Utilities

| Dependency | Installed Version | Installation Method | Purpose | Status |
|-----------|------------------|-----------|--------|
| psutil | 5.9.8 | pip | System Monitoring | ✅ |
| argcomplete | 3.6.3 | pip | Command-line completion | ✅ |

### 📊 Isaac Sim Information

- **Version**: 5.1.0-rc.19+release.26219.9c81211b.gl (~5.1.0)
- **Python Version**: >= 3.10
- **CUDA Support**: CUDA 12.x (via torch==2.7.0+cu128)
- **RAM Recommendation**: 16GB minimum, 32GB preferred for full training

### 📂 Requirements Files

| File | Description | Use Case |
|------|-------------|----------|
| `requirements.txt` | Complete dependencies list (based on actual installed versions) | Production/Development |
| `requirements-dev.txt` | Development and testing tools only | Development |
| `requirements-minimal.txt` | Core training dependencies only | Isaac Sim training |

## 🚀 Installation Methods

### Method 1: Isaac Sim Python Environment (Recommended)

Isaac Sim 5.1.0 comes with a pre-configured Python environment that includes most dependencies.

```bash
# Activate Isaac Sim environment
source /home/jay/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

# Verify environment
python3 -c "import torch; import gymnasium; import isaaclab; print('Environment ready')"
```

**Pros:**
- ✅ No need for additional installation
- ✅ Most compatible version combination
- ✅ Works out of the box

**Cons:**
- ❌ Dependent on Isaac Sim specific version
- ❌ Less flexibility for environment customization

### Method 2: Standard pip Installation (Alternative)

```bash
# Install core dependencies
pip install torch==2.7.0+cu128 gymnasium==1.2.1 psutil==5.9.8 argcomplete==3.6.3

# Verify installation
python3 -c "import torch; import gymnasium; import isaaclab; print('All dependencies installed')"
```

### 🎯 Quick Start

#### Test Training (Minimal Setup)
```bash
# Activate environment
conda activate env_isaaclab

# Run minimal training script
./scripts/train_direct.sh
```

## 🔍 Troubleshooting

### Common Issues and Solutions

#### 1. Isaac Sim Crashes (Segmentation Fault)

**Symptoms:** Segmentation fault + core dumped + C++ template symbol errors

**Solution:**
```bash
# Use minimal training script with reduced environment count
./scripts/train_direct.sh
```
**Environment Variables:**
```bash
export _GLIBCXX_ASSERTIONS=0
export MALLOC_CHECK_=0
export MALLOC_PERTURB_=0
export CARB_APP_DISABLE_FILE_WATCHING=1
export MALLOC_TRIM_THRESHOLD_=131072
```

#### 2. Parameter Conflicts

**Symptoms:** ValueError: field 'headless' already exists

**Solution:** Use fixed training scripts that avoid conflicts

#### 3. Module Import Errors

**Symptoms:** ModuleNotFoundError: No module named 'omni.timeline'

**Cause:** Importing Isaac Lab modules before SimulationApp instantiation

**Solution:** Training scripts are configured correctly. Use provided scripts.

## 📊 Environment Requirements

- **Python Version:** >= 3.10
- **Isaac Sim Version:** 5.1.0-rc.19+release.26219.9c81211b.gl (~5.1.0)
- **CUDA Support:** CUDA 12.x (via torch==2.7.0+cu128)
- **RAM Recommendation:** 16GB minimum, 32GB preferred for full training
- **Disk Space:** 10GB+ available for cache and logs

## 📁 Project Structure

```
unitree_rl_lab/
├── source/unitree_rl_lab/
│   ├── unitree_rl_lab/               # Main package
│   ├── tasks/                        # RL tasks and environments
│   │   ├── locomotion/             # Locomotion tasks
│   │   │   ├── mdp/               # MDP functions
│   │   │   ├── actions.py      # Action spaces
│   │   │   ├── rewards.py       # Reward functions
│   │   │   ├── observations.py # Observation spaces
│   │   │   └── terminations.py # Termination conditions
│   │   └── robots/               # Robot configurations
│   │       └── go2w_arm/         # GO2W ARM with arm
│   │           └── two_stage_recovery_env_cfg.py
├── scripts/                          # Training and utility scripts
│   ├── train_go2w.py            # Main training script
│   ├── train_direct.sh           # Direct training (minimal)
│   ├── train_fixed.py            # Fixed training script
│   ├── train_working.sh           # Working script
│   └── diagnose_isaac_sim.sh   # Diagnostic script
├── requirements.txt                  # Complete dependencies list
├── requirements-dev.txt               # Development dependencies
├── requirements-minimal.txt           # Minimal dependencies
└── INSTALLATION.md                  # Installation guide
```

## 📞 Support and Resources

- **Isaac Lab Documentation:** https://isaac-sim.github.io/IsaacLab/main
- **Isaac Sim Documentation:** https://docs.omniverse.nvidia.com/
- **Unitree Documentation:** Check robot manufacturer documentation
- **Issue Reporting:** Report problems to project repository

## ⚠️ Important Notes

1. **Always activate correct conda environment:** `conda activate env_isaaclab`
2. **Use provided training scripts:** They are configured to handle common issues
3. **Monitor system resources:** During training, especially with large environment counts
4. **Check for Isaac Sim updates:** If experiencing compatibility issues
5. **Isaac Lab packages are editable:** isaaclab project can be modified in /home/jay/IsaacLab/source/isaaclab
6. **CUDA compatibility:** Ensure CUDA driver matches PyTorch version

---

**Generated for GO2W RL Lab Project**
**Last Updated:** 2026-04-03  
**Based on environment:** env_isaaclab (Isaac Sim 5.1.0, Python 3.11, PyTorch 2.7.0+cu128)