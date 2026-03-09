# GO2W Training Guide

This guide explains how to train the Unitree GO2W wheel-legged robot using the provided training scripts.

## Overview

The GO2W robot can be trained on two types of terrain:

1. **Flat Terrain**: Simple plane terrain for basic locomotion training
2. **Rough Terrain**: Complex generated terrain with various obstacles for advanced training

## Quick Start

### Method 1: Using the Bash Script (Recommended for Beginners)

The bash script provides a simple interface for common training scenarios:

```bash
# Train on flat terrain with default settings (GUI enabled by default)
./scripts/train_go2w.sh flat

# Train on rough terrain with default settings (GUI enabled by default)
./scripts/train_go2w.sh rough

# Train without GUI (headless mode for faster training)
./scripts/train_go2w.sh flat --headless

# Train with custom number of environments
./scripts/train_go2w.sh flat --num_envs 8192

# Resume training from last checkpoint
./scripts/train_go2w.sh flat --resume

# Play with trained policy
./scripts/train_go2w.sh play-flat
```

### Method 2: Using the Python Script (More Flexible)

The Python script provides more customization options:

```bash
# Basic usage
python scripts/train_go2w.py --mode flat

# With custom parameters
python scripts/train_go2w.py --mode rough --num_envs 8192 --max_iterations 20000

# With video recording
python scripts/train_go2w.py --mode flat --video --video_interval 2000

# Resume from checkpoint
python scripts/train_go2w.py --mode flat --resume --load_run recent
```

## Training Modes

### Flat Terrain Training

Flat terrain uses a simple plane, ideal for:
- Initial policy training
- Basic locomotion skills
- Faster training iterations
- Debugging and testing

```bash
./scripts/train_go2w.sh flat
```

**Configuration:**
- Terrain: Infinite plane
- No terrain curriculum
- No height scanning
- Simpler observation space

### Rough Terrain Training

Rough terrain uses generated terrain with various obstacles:
- Stairs, slopes, and uneven surfaces
- Terrain curriculum learning
- Height scanning for terrain awareness
- More complex observation space

```bash
./scripts/train_go2w.sh rough
```

**Configuration:**
- Terrain: Generated with multiple difficulty levels
- Terrain curriculum enabled
- Height scanning active
- Full observation space

## Command-Line Options

### Bash Script Options

```bash
./scripts/train_go2w.sh [MODE] [OPTIONS]

Modes:
  flat          Train on flat terrain
  rough         Train on rough terrain
  play-flat     Play with policy on flat terrain
  play-rough    Play with policy on rough terrain

Options:
  --num_envs N       Number of environments (default: 4096)
  --headless         Run without GUI (default: enabled)
  --gui              Enable GUI visualization
  --device D         Device to use (default: cuda:0)
  --iterations N     Max training iterations (default: 10000)
  --seed N           Random seed (default: 42)
  --video            Record videos during training
  --resume           Resume from checkpoint
```

### Python Script Options

```bash
python scripts/train_go2w.py --mode [MODE] [OPTIONS]

Required:
  --mode {flat,rough,play-flat,play-rough}    Training mode

Environment:
  --num_envs INT              Number of parallel environments (default: 4096)
  --seed INT                  Random seed (default: 42)

Training:
  --max_iterations INT        Maximum training iterations (default: 10000)
  --resume                    Resume from checkpoint
  --load_run STR              Run name to load (default: recent)
  --load_checkpoint STR       Checkpoint name to load

Visualization:
  --headless                  Run without GUI (default: enabled)
  --gui                       Enable GUI visualization
  --video                     Record videos
  --video_interval INT        Video interval in steps (default: 2000)
  --video_length INT          Video length in steps (default: 200)

Hardware:
  --device STR                Device to use (default: cuda:0)
  --distributed               Enable multi-GPU training
```

## Training Workflow

### 1. Initial Training on Flat Terrain

Start with flat terrain to learn basic locomotion:

```bash
./scripts/train_go2w.sh flat --num_envs 4096
```

Monitor training progress in TensorBoard:
```bash
tensorboard --logdir logs/rsl_rl/
```

### 2. Transition to Rough Terrain

Once the policy learns basic locomotion on flat terrain (typically after 2000-5000 iterations), transfer to rough terrain:

```bash
./scripts/train_go2w.sh rough --resume
```

The training will automatically load the latest checkpoint from flat terrain training.

### 3. Fine-tuning on Rough Terrain

Continue training on rough terrain to adapt to complex terrain:

```bash
./scripts/train_go2w.sh rough --max_iterations 20000
```

## Monitoring Training

### TensorBoard

View training metrics:

```bash
tensorboard --logdir logs/rsl_rl/
```

Key metrics to monitor:
- **ep_rew_mean**: Mean episode reward (should increase)
- **value_loss**: Value function loss (should decrease)
- **policy_loss**: Policy loss (should stabilize)
- **ratio**: PPO ratio (should stay near 1.0)

### Checkpoints

Checkpoints are saved automatically in:
```
logs/rsl_rl/[experiment_name]/[timestamp]/
├── model_*.pt              # Model checkpoints
├── params/
│   ├── env.yaml           # Environment configuration
│   └── agent.yaml         # Agent configuration
└── videos/                # Training videos (if enabled)
```

### Resume Training

To resume from a specific checkpoint:

```bash
# Resume from latest checkpoint
./scripts/train_go2w.sh flat --resume

# Resume from specific run
./scripts/train_go2w.sh flat --resume --load_run 2025-01-15_10-30-00_my_run
```

## Playing with Trained Policies

### Interactive Play

```bash
# Play on flat terrain
./scripts/train_go2w.sh play-flat

# Play on rough terrain
./scripts/train_go2w.sh play-rough
```

### Keyboard Controls

When in play mode, use keyboard to control the robot:
- **W/S**: Forward/Backward velocity
- **A/D**: Left/Right velocity
- **Q/E**: Turn left/right
- **Space**: Reset robot

## Hardware Recommendations

### GPU Requirements

| Environments | Recommended GPU | VRAM |
|--------------|-----------------|------|
| 512          | GTX 1660        | 6GB  |
| 2048         | RTX 3070        | 8GB  |
| 4096         | RTX 3080        | 10GB |
| 8192         | RTX 3090        | 24GB |
| 16384        | A100            | 40GB |

### Training Speed

Approximate training speed on RTX 3090:
- Flat terrain: ~1500-2000 iterations/hour
- Rough terrain: ~1000-1500 iterations/hour

## Troubleshooting

### Out of Memory

Reduce number of environments:
```bash
./scripts/train_go2w.sh flat --num_envs 2048
```

### Training Not Converging

1. Check reward weights in configuration
2. Reduce learning rate
3. Increase number of environments
4. Train longer (more iterations)

### Simulation Instability

1. Reduce action scale in configuration
2. Increase decimation
3. Check physics materials
4. Reduce time step

### GUI Performance Issues

Run in headless mode:
```bash
./scripts/train_go2w.sh flat --headless
```

## Advanced Configuration

### Custom Reward Weights

Edit the reward weights in `velocity_env_cfg.py`:

```python
@configclass
class RewardsCfg:
    # Adjust these weights to modify behavior
    track_lin_vel_xy = RewTerm(func=mdp.track_lin_vel_xy_exp, weight=1.5)
    track_ang_vel_z = RewTerm(func=mdp.track_ang_vel_z_exp, weight=0.75)
    # ... more rewards
```

### Custom Terrain Generation

Modify terrain configuration in `velocity_env_cfg.py`:

```python
@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,  # Customize this
        # ... more options
    )
```

## Tips for Successful Training

1. **Start Simple**: Begin with flat terrain to establish basic locomotion
2. **Monitor Progress**: Use TensorBoard to track training metrics
3. **Save Checkpoints**: Regularly save and test policies
4. **Curriculum Learning**: Let the terrain curriculum progress naturally
5. **Hyperparameter Tuning**: Adjust reward weights based on observed behavior
6. **Patience**: Training may take 10,000+ iterations for good performance

## Examples

### Complete Training Pipeline

```bash
# Step 1: Train on flat terrain (5000 iterations)
./scripts/train_go2w.sh flat --max_iterations 5000 --num_envs 4096

# Step 2: Transfer to rough terrain (continue training)
./scripts/train_go2w.sh rough --resume --max_iterations 10000

# Step 3: Fine-tune on rough terrain (additional 5000 iterations)
./scripts/train_go2w.sh rough --resume --max_iterations 15000

# Step 4: Test the final policy
./scripts/train_go2w.sh play-rough --gui
```

### Fast Training (Less Environments)

```bash
# For faster iteration testing with fewer environments
./scripts/train_go2w.sh flat --num_envs 512 --max_iterations 2000
```

### High-Quality Training (More Environments)

```bash
# For better quality with more environments
./scripts/train_go2w.sh rough --num_envs 8192 --max_iterations 20000
```

## References

- [IsaacLab Documentation](https://isaac-sim.github.io/IsaacLab/)
- [RSL-RL Documentation](https://leggedrobotics.github.io/rsl_rl/)
- [Unitree GO2W Hardware Documentation](https://www.unitree.com/go2w/)

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review training logs in `logs/rsl_rl/`
3. Open an issue on the project repository
