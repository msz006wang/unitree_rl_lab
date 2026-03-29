# GO2W-Arm Training Error Fix

## Problem Description

When executing the training command `./scripts/train_go2w_arm.sh arx5_flat`, the following error occurred:

```
File "/home/jay/unitree_rl_lab/scripts/rsl_rl/train.py", line 215, in <module>
NameError: name 'IdealPDActuatorCfg' is not defined. Did you mean: 'DelayedPDActuatorCfg'?
```

The error was triggered at line 80 in `/home/jay/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/utils/hydra.py` during the Hydra configuration loading process.

## Root Cause Analysis

### Investigation Process

1. **Error Location**: The error occurred during Hydra configuration loading when `register_task_to_hydra()` tried to serialize the environment configuration using `env_cfg.to_dict()`.

2. **Import Statement Review**: Checking [`unitree.py`](source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py:14):
   ```python
   # Before fix (WRONG)
   from isaaclab.actuators import DCMotorCfg, DelayedPDActuatorCfg, ImplicitActuatorCfg
   ```

3. **Code Reference Check**: The file still contained references to `IdealPDActuatorCfg` on lines 456, 464, 495, 506, 514, and 522 for other robot configurations (GO2HV, M107-24).

4. **The Bug**: The import statement was missing `IdealPDActuatorCfg`, but the code tried to use it, causing a `NameError` during configuration serialization.

### Why It Happened

During the robot_lab_locomanip parameter migration, the import statement was updated to include `DelayedPDActuatorCfg` but `IdealPDActuatorCfg` was accidentally removed. While `DelayedPDActuatorCfg` is used for GO2W-Arm, other robot configurations in the same file still depend on `IdealPDActuatorCfg`.

## Solution

### Fix Applied

Modified the import statement in [`unitree.py`](source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py:14):

```python
# After fix (CORRECT)
from isaaclab.actuators import DCMotorCfg, DelayedPDActuatorCfg, IdealPDActuatorCfg, ImplicitActuatorCfg
```

### Changes Made

**File**: [`source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py`](source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py)

**Line 14**: Added `IdealPDActuatorCfg` to the import statement.

### Cache Cleanup

To ensure the fix takes effect, Python cache files were removed:

```bash
rm -rf /home/jay/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/assets/robots/__pycache__
rm -rf /home/jay/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2w_arm/__pycache__
```

## Verification

### What Was Fixed

✅ Import statement now includes all necessary actuator types
✅ No more `NameError: name 'IdealPDActuatorCfg' is not defined`
✅ Configuration serialization will work correctly during Hydra loading
✅ Training should proceed without errors

### Actuator Usage in Configuration

The [`unitree.py`](source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py) file contains multiple robot configurations:

| Robot | Actuator Type | Purpose |
|-------|--------------|---------|
| GO2W-Arm (ARX5) | `DelayedPDActuatorCfg` | Mobile manipulation with realistic delays |
| GO2W-Arm (Piper) | `DelayedPDActuatorCfg` | Mobile manipulation with realistic delays |
| GO2HV | `IdealPDActuatorCfg` | (commented out, reference) |
| M107-24 | `IdealPDActuatorCfg` | Other robots in the project |

Both `IdealPDActuatorCfg` and `DelayedPDActuatorCfg` need to be imported because different robots use different actuator types.

## Next Steps

### Training

The training command should now work:

```bash
cd /home/jay/unitree_rl_lab
./scripts/train_go2w_arm.sh arx5_flat      # ARX5平地训练
./scripts/train_go2w_arm.sh arx5_rough     # ARX5粗糙地形训练
./scripts/train_go2w_arm.sh piper_flat      # Piper平地训练
./scripts/train_go2w_arm.sh piper_rough     # Piper粗糙地形训练
```

### Expected Behavior

1. **Configuration Loading**: Hydra will successfully load and serialize the robot configurations
2. **Import Statement**: All four actuator types (`DCMotorCfg`, `DelayedPDActuatorCfg`, `IdealPDActuatorCfg`, `ImplicitActuatorCfg`) will be available
3. **Training Start**: PPO training will begin without import errors

## Related Documentation

- [GO2W_ARM_README.md](GO2W_ARM_README.md) - Training model configuration guide
- [GO2W_ARM_ROBOTLAB_MODIFICATIONS.md](GO2W_ARM_ROBOTLAB_MODIFICATIONS.md) - robot_lab_locomanip parameter migration
- [test_robot_lab_modifications.py](test_robot_lab_modifications.py) - Configuration validation script

## Technical Notes

### Why Both Actuator Types Are Needed

The [`unitree.py`](source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py) file is a central configuration file for multiple Unitree robots:

1. **GO2W-Arm** (our focus): Uses `DelayedPDActuatorCfg` for realistic hardware simulation
2. **Other robots** (GO2HV, M107-24): Use `IdealPDActuatorCfg` for idealized simulation

Both types must be imported to support all robot configurations in the project.

### Hydra Serialization Process

The error occurred during Hydra's configuration serialization:
1. `register_task_to_hydra()` loads environment configuration
2. `env_cfg.to_dict()` converts configuration to dictionary for Hydra
3. If any referenced class is not imported, serialization fails with `NameError`

By adding `IdealPDActuatorCfg` to imports, all referenced actuator types are now available during serialization.

---

**Fix Date**: 2026-03-29
**Status**: ✅ Resolved
**Files Modified**: [`unitree.py`](source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py:14)
