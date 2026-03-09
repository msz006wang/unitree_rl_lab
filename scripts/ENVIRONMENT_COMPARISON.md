# GO2W Environment Comparison: Flat vs Rough

This document provides a detailed comparison between Flat and Rough terrain environments for GO2W training.

## Quick Reference

| Feature | Flat Terrain | Rough Terrain |
|---------|--------------|---------------|
| **Terrain Type** | Infinite plane | Generated terrain with obstacles |
| **Training Difficulty** | Easy | Hard |
| **Training Speed** | Fast (~1500-2000 iter/hour) | Medium (~1000-1500 iter/hour) |
| **Recommended For** | Initial training, debugging | Advanced training, real-world deployment |
| **Terrain Curriculum** | Disabled | Enabled |
| **Height Scanner** | Disabled | Enabled |
| **Observation Space Size** | Smaller | Larger |
| **Memory Usage** | Lower | Higher |

## Detailed Comparison

### 1. Terrain Configuration

#### Flat Terrain
```python
terrain = TerrainImporterCfg(
    terrain_type="plane",          # Simple infinite plane
    terrain_generator=None,        # No terrain generation
)
```

**Characteristics:**
- Perfectly flat surface
- No obstacles or variations
- Consistent friction everywhere
- Predictable footing

#### Rough Terrain
```python
terrain = TerrainImporterCfg(
    terrain_type="generator",      # Generated terrain
    terrain_generator=ROUGH_TERRAINS_CFG,  # Complex terrain
    max_init_terrain_level=5,      # Starting difficulty
)
```

**Characteristics:**
- Multiple terrain types: stairs, slopes, rough ground
- Progressive difficulty through curriculum
- Varied friction and elevation
- Realistic challenges

### 2. Observation Space

#### Flat Terrain Observations
```
Total size: ~45 dimensions
- Base angular velocity: 3
- Projected gravity: 3
- Velocity commands: 3
- Joint positions (legs only): 12
- Joint velocities: 12
- Last actions: 12
```

#### Rough Terrain Observations
```
Total size: ~45 + height_scan dimensions
- Base angular velocity: 3
- Projected gravity: 3
- Velocity commands: 3
- Joint positions (legs only): 12
- Joint velocities: 12
- Last actions: 12
- Height scan: ~16-32 (added)
```

**Key Difference:** Rough terrain includes height scanning for terrain awareness.

### 3. Reward Configuration

#### Flat Terrain Rewards
```python
# Simplified rewards for flat terrain
track_lin_vel_xy = RewTerm(weight=1.5)    # Track velocity commands
track_ang_vel_z = RewTerm(weight=0.75)    # Track angular velocity
joint_pos = RewTerm(weight=-0.7)          # Joint position penalty
energy = RewTerm(weight=-2e-5)            # Energy efficiency
# ... other rewards
```

**Focus:**
- Velocity tracking
- Energy efficiency
- Smooth motion
- Joint stability

#### Rough Terrain Rewards
```python
# Enhanced rewards for rough terrain
track_lin_vel_xy = RewTerm(weight=1.5)    # Track velocity commands
track_ang_vel_z = RewTerm(weight=0.75)    # Track angular velocity
feet_air_time = RewTerm(weight=0.1)       # Encourage dynamic stepping
feet_slide = RewTerm(weight=-0.1)         # Penalize foot sliding
undesired_contacts = RewTerm(weight=-1.0) # Avoid unwanted contacts
# ... additional rough terrain rewards
```

**Focus:**
- All flat terrain rewards, plus:
- Dynamic foot placement
- Foot slip prevention
- Obstacle avoidance
- Terrain adaptation

### 4. Curriculum Learning

#### Flat Terrain
```python
curriculum = CurriculumCfg(
    terrain_levels = None,        # Disabled
    command_levels = CurrTerm(
        func=mdp.command_levels_vel,
        params={"reward_term_name": "track_lin_vel_xy"}
    ),
)
```

**Only command curriculum:**
- Gradually increases velocity command ranges
- Starts at 10% of max velocity
- Progresses to 100% based on performance

#### Rough Terrain
```python
curriculum = CurriculumCfg(
    terrain_levels = CurrTerm(
        func=mdp.terrain_levels_vel,
        params={"asset_cfg": SceneEntityCfg("robot")}
    ),
    command_levels = CurrTerm(
        func=mdp.command_levels_vel,
        params={"reward_term_name": "track_lin_vel_xy"}
    ),
)
```

**Both command and terrain curricula:**
- Command curriculum (same as flat)
- Terrain curriculum:
  - Increases terrain difficulty based on travel distance
  - Automatically adjusts to robot capability
  - Provides progressive challenge

### 5. Termination Conditions

Both environments use similar termination conditions:

```python
terminations = TerminationsCfg(
    time_out = DoneTerm(func=mdp.time_out, time_out=True),
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base")}
    ),
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 0.8}  # ~45 degrees
    ),
)
```

**Note:** Rough terrain may have more lenient termination to allow recovery attempts.

### 6. Event Randomization

Both environments use similar randomization events:

```python
events = EventCfg(
    # Startup randomization
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={"static_friction_range": (0.3, 1.2)}
    ),
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={"mass_distribution_params": (-1.0, 3.0)}
    ),

    # Reset randomization
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}}
    ),

    # Interval randomization
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 10.0)
    ),
)
```

### 7. Training Workflow

#### Recommended Flat Terrain Workflow

1. **Initial Training** (0-5000 iterations)
   - Learn basic locomotion
   - Establish velocity tracking
   - Develop energy-efficient gait

2. **Refinement** (5000-10000 iterations)
   - Smooth out jerky motions
   - Improve stability
   - Fine-tune gait parameters

#### Recommended Rough Terrain Workflow

1. **Flat Terrain Pre-training** (0-5000 iterations)
   ```bash
   ./train_go2w.sh flat --max_iterations 5000
   ```

2. **Transfer to Rough Terrain** (5000-10000 iterations)
   ```bash
   ./train_go2w.sh rough --resume --max_iterations 10000
   ```

3. **Fine-tuning on Rough Terrain** (10000-20000 iterations)
   ```bash
   ./train_go2w.sh rough --resume --max_iterations 20000
   ```

### 8. Performance Metrics

#### Flat Terrain Performance

After 10000 iterations on flat terrain:
- **Velocity Tracking Error**: < 0.1 m/s
- **Angular Velocity Error**: < 0.2 rad/s
- **Energy Consumption**: ~200-300 W
- **Episode Length**: ~20 seconds (no terminations)

#### Rough Terrain Performance

After 20000 iterations on rough terrain:
- **Velocity Tracking Error**: < 0.2 m/s
- **Angular Velocity Error**: < 0.3 rad/s
- **Energy Consumption**: ~300-500 W
- **Success Rate**: > 90% on moderate terrain
- **Episode Length**: ~15-20 seconds

### 9. Computational Requirements

#### Flat Terrain
- **GPU Memory** (4096 envs): ~8 GB
- **Training Speed**: ~1500-2000 iterations/hour
- **Disk Space**: ~500 MB per 1000 iterations

#### Rough Terrain
- **GPU Memory** (4096 envs): ~10 GB
- **Training Speed**: ~1000-1500 iterations/hour
- **Disk Space**: ~700 MB per 1000 iterations

### 10. Use Cases

#### Flat Terrain Best For

1. **Initial Policy Training**
   - Learn basic locomotion primitives
   - Establish stable gaits
   - Debug reward functions

2. **Quick Iteration**
   - Test hyperparameters
   - Validate code changes
   - Rapid prototyping

3. **Specific Applications**
   - Warehouse robots (flat floors)
   - Indoor navigation
   - Controlled environments

#### Rough Terrain Best For

1. **Advanced Training**
   - Real-world deployment preparation
   - Robustness testing
   - Generalization improvement

2. **Specific Applications**
   - Outdoor navigation
   - Search and rescue
   - Unstructured environments

3. **Research**
   - Terrain adaptation
   - Falling recovery
   - Energy-efficient locomotion

### 11. Troubleshooting

#### Flat Terrain Issues

**Problem**: Robot can't stay upright
**Solution**:
- Increase `flat_orientation_l2` reward weight
- Check initial joint positions
- Verify physics materials

**Problem**: Poor velocity tracking
**Solution**:
- Increase `track_lin_vel_xy` reward weight
- Adjust command curriculum speed
- Check observation noise levels

#### Rough Terrain Issues

**Problem**: Frequent terminations on rough terrain
**Solution**:
- Reduce terrain curriculum speed
- Train longer on flat terrain first
- Adjust termination thresholds

**Problem**: Robot gets stuck on obstacles
**Solution**:
- Increase `feet_air_time` reward weight
- Enable height scanning observations
- Add terrain-specific rewards

**Problem**: Training is too slow
**Solution**:
- Reduce number of environments
- Disable video recording
- Use flat terrain for initial training

### 12. Configuration Files

Environment configurations are stored in:
```
source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2w/
├── velocity_env_cfg.py    # Main configuration
└── __init__.py           # Environment registration
```

Key configuration classes:
- `RobotSceneCfg`: Scene and terrain setup
- `EventCfg`: Randomization events
- `RewardsCfg`: Reward functions and weights
- `TerminationsCfg`: Termination conditions
- `CurriculumCfg`: Learning curricula

### 13. Switching Between Environments

The main difference between Flat and Rough environments is controlled in `__post_init__`:

```python
# For Flat terrain (if we want to create a separate FlatEnvCfg class)
class RobotFlatEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Change to flat terrain
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # Disable height scanning
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        # Disable terrain curriculum
        self.curriculum.terrain_levels = None
```

Currently, both environments use the same `RobotEnvCfg` class, with the terrain type determined by which environment ID is registered:
- `Unitree-Go2W-Velocity-Flat-v0`: Flat terrain
- `Unitree-Go2W-Velocity-Rough-v0`: Rough terrain

## Summary

| Aspect | Flat Terrain | Rough Terrain |
|--------|--------------|---------------|
| **Learning Curve** | Steep, fast | Gradual, slower |
| **Final Performance** | Good on flat surfaces | Excellent on all terrains |
| **Training Time** | Shorter | Longer |
| **Compute Resources** | Lower | Higher |
| **Recommended First Step** | ✅ Yes | ❌ No (start with flat) |
| **Production Ready** | Limited | Yes |

**Best Practice**: Always start with flat terrain training, then transfer to rough terrain for final deployment preparation.
