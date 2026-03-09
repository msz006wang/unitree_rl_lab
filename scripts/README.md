# GO2W Training Scripts and Documentation

This directory contains comprehensive training scripts and documentation for the Unitree GO2W wheel-legged robot.

## 📚 Documentation Files

### [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
**Complete training guide** - Start here for detailed instructions on training GO2W.

Contents:
- Quick start examples
- Detailed command-line options
- Training workflow recommendations
- Monitoring and debugging
- Hardware requirements
- Troubleshooting tips

### [ENVIRONMENT_COMPARISON.md](ENVIRONMENT_COMPARISON.md)
**Flat vs Rough terrain comparison** - Understand the differences between training environments.

Contents:
- Feature comparison table
- Terrain configuration details
- Observation space differences
- Reward function variations
- Performance metrics
- Use case recommendations

## 🚀 Training Scripts

### 1. [train_go2w.sh](train_go2w.sh) ⭐ **Recommended for Beginners**
Bash script with simple interface for common training scenarios.

**Features:**
- Easy-to-use command syntax
- Color-coded output
- Automatic error handling
- Quick start options

**Usage:**
```bash
# Train on flat terrain
./train_go2w.sh flat

# Train on rough terrain
./train_go2w.sh rough

# With custom settings
./train_go2w.sh flat --num_envs 8192 --gui
```

### 2. [train_go2w.py](train_go2w.py) 🔧 **Advanced Users**
Python script with maximum flexibility and customization options.

**Features:**
- Comprehensive argument parsing
- Advanced configuration options
- Distributed training support
- Video recording capabilities

**Usage:**
```bash
# Basic usage
python train_go2w.py --mode flat

# Advanced configuration
python train_go2w.py --mode rough \
    --num_envs 8192 \
    --max_iterations 20000 \
    --video \
    --video_interval 2000
```

### 3. [quick_start_training.sh](quick_start_training.sh) ⚡ **Fastest Way to Start**
Minimal script for immediate training start with default settings.

**Usage:**
```bash
# Start training on flat terrain
./quick_start_training.sh flat

# Start training on rough terrain
./quick_start_training.sh rough
```

## 🎯 Quick Start

### Option 1: Quick Start (Recommended for First-Time Users)

```bash
# Train on flat terrain with default settings (GUI enabled)
./quick_start_training.sh flat
```

### Option 2: Custom Training

```bash
# Train on rough terrain with custom settings
./train_go2w.sh rough --num_envs 8192 --max_iterations 15000
```

### Option 3: Play with Trained Policy

```bash
# Interactive play on rough terrain
./train_go2w.sh play-rough --gui
```

## 📊 Training Workflow

### Complete Training Pipeline

```bash
# Step 1: Initial training on flat terrain (5000 iterations)
./train_go2w.sh flat --max_iterations 5000

# Step 2: Transfer to rough terrain (10000 iterations)
./train_go2w.sh rough --resume --max_iterations 10000

# Step 3: Fine-tune on rough terrain (20000 iterations total)
./train_go2w.sh rough --resume --max_iterations 20000

# Step 4: Test the final policy
./train_go2w.sh play-rough --gui
```

### Monitoring Training

```bash
# View training progress
tensorboard --logdir logs/rsl_rl/
```

## 🔧 Common Use Cases

### Fast Iteration (Testing)
```bash
# Few environments, short training
./train_go2w.sh flat --num_envs 512 --max_iterations 1000
```

### High-Quality Training
```bash
# Many environments, long training
./train_go2w.sh rough --num_envs 8192 --max_iterations 20000
```

### Production Training
```bash
# Maximum environments, full training
./train_go2w.sh rough --num_envs 16384 --max_iterations 30000
```

### Debugging
```bash
# GUI enabled, few environments
./train_go2w.sh flat --num_envs 128 --gui
```

### Recording Videos
```bash
# Record training videos
./train_go2w.sh rough --video --video_interval 2000
```

## 📈 Training Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `flat` | Train on flat terrain | Initial training, quick iteration |
| `rough` | Train on rough terrain | Advanced training, real-world prep |
| `play-flat` | Play on flat terrain | Test flat terrain policy |
| `play-rough` | Play on rough terrain | Test rough terrain policy |

## 🎮 Keyboard Controls (Play Mode)

When in play mode:
- **W/S**: Forward/Backward velocity
- **A/D**: Left/Right velocity
- **Q/E**: Turn left/right
- **Space**: Reset robot

## 💡 Tips

1. **Start Simple**: Begin with flat terrain training
2. **Monitor Progress**: Use TensorBoard to track metrics
3. **Save Checkpoints**: Training saves automatically
4. **Be Patient**: Training takes time (10k+ iterations)
5. **Experiment**: Adjust reward weights in config files

## 🛠️ Troubleshooting

### Out of Memory
```bash
# Reduce number of environments
./train_go2w.sh flat --num_envs 2048
```

### Training Not Converging
1. Check reward weights in `velocity_env_cfg.py`
2. Try training longer
3. Increase number of environments

### GUI Performance Issues
```bash
# Run in headless mode
./train_go2w.sh flat --headless
```

## 📖 Additional Resources

- [IsaacLab Documentation](https://isaac-sim.github.io/IsaacLab/)
- [RSL-RL Documentation](https://leggedrobotics.github.io/rsl_rl/)
- [Unitree GO2W Hardware](https://www.unitree.com/go2w/)

## 📁 File Structure

```
scripts/
├── TRAINING_GUIDE.md              # Complete training guide
├── ENVIRONMENT_COMPARISON.md      # Flat vs Rough comparison
├── README.md                       # This file
├── train_go2w.sh                   # Bash training script ⭐
├── train_go2w.py                   # Python training script 🔧
└── quick_start_training.sh         # Quick start script ⚡
```

## 🤝 Support

For issues or questions:
1. Check [TRAINING_GUIDE.md](TRAINING_GUIDE.md) troubleshooting section
2. Review [ENVIRONMENT_COMPARISON.md](ENVIRONMENT_COMPARISON.md) for environment details
3. Check training logs in `logs/rsl_rl/`
4. Open an issue on the project repository

## 📝 Version History

- **v1.0** (2025-03-07): Initial release
  - Flat and Rough terrain support
  - Comprehensive training scripts
  - Complete documentation

---

**Ready to train?** Start with [quick_start_training.sh](quick_start_training.sh) or see [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for detailed instructions!
