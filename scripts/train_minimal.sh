#!/bin/bash
# 最小化训练脚本 - 避免内存和兼容性问题

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

TASK_NAME="Unitree-Go2WArm-TwoStage-Recovery-v0"
NUM_ENVS=4  # 使用最少的环境以避免内存问题

log_info "启动最小化训练测试..."
log_info "任务: ${TASK_NAME}, 环境: ${NUM_ENVS}"

cd "${PROJECT_ROOT}"

# 设置环境变量
export PYTHONPATH="${PROJECT_ROOT}/source:${PYTHONPATH}"
export ISAACSIM_PATH="/home/jay/IsaacLab/apps/isaacsim_4_5"
export ISAACLAB_PATH="/home/jay/IsaacLab"

# 禁用文件监视
export CARB_APP_DISABLE_FILE_WATCHING=1

# 限制内存使用
export MALLOC_TRIM_THRESHOLD_=131072

# 禁用一些可能导致崩溃的功能
export OMPI_MCA_btl_vader_single_copy_method=expect_copy

log_info "启动训练..."

# 直接使用Python，不通过bash包装
python3 - << 'PYEOF'
import os
import sys
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import gymnasium as gym
import torch

# Configure torch optimizations (minimal settings)
torch.backends.cuda.matmul.allow_tf32 = False  # 禁用可能不兼容的优化
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

# Import AppLauncher
from isaaclab.app import AppLauncher

# Create minimal parser
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Unitree-Go2WArm-TwoStage-Recovery-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--max_iterations", type=int, default=100)  # 最少迭代用于测试
parser.add_argument("--seed", type=int, default=42)

# Add AppLauncher arguments
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

print(f"Training {args.task} with {args.num_envs} environments")

# Launch Isaac Sim
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Import tasks after SimulationApp is instantiated
import unitree_rl_lab.tasks

try:
    import time
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab.utils.io import dump_yaml
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import get_checkpoint_path
    from isaaclab_tasks.utils.hydra import hydra_task_config

    # Import CLI args
    sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "rsl_rl"))
    import cli_args
    sys.path.pop(0)

    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from list_envs import import_packages
    sys.path.pop(0)

    sys.argv = [sys.argv[0]]

    @hydra_task_config(args.task, "rsl_rl_cfg_entry_point")
    def train_main(env_cfg, agent_cfg):
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args)
        env_cfg.scene.num_envs = args.num_envs
        agent_cfg.max_iterations = args.max_iterations
        env_cfg.seed = agent_cfg.seed
        env_cfg.sim.device = args.device

        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_dir = os.path.join(log_root_path, "test_run")

        env = gym.make(args.task, cfg=env_cfg)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

        print("Starting training...")
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
        env.close()

    train_main()

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

finally:
    simulation_app.close()

print("Training completed!")
PYEOF
