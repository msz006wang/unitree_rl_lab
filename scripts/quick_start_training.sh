#!/bin/bash
# ============================================================================
# Quick Start Training Script for GO2W
# ============================================================================
# This script provides the fastest way to start training GO2W robot
#
# Usage:
#   ./quick_start_training.sh [flat|rough]
# ============================================================================

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Check argument
if [ $# -eq 0 ]; then
    echo "Usage: $0 [flat|rough]"
    echo ""
    echo "Examples:"
    echo "  $0 flat   # Start training on flat terrain"
    echo "  $0 rough  # Start training on rough terrain"
    exit 1
fi

MODE=$1

# Validate mode
if [ "$MODE" != "flat" ] && [ "$MODE" != "rough" ]; then
    echo "Error: Invalid mode '$MODE'. Use 'flat' or 'rough'."
    exit 1
fi

# Set task name
if [ "$MODE" == "flat" ]; then
    TASK="Unitree-Go2W-Velocity-Flat-v0"
else
    TASK="Unitree-Go2W-Velocity-Rough-v0"
fi

echo "=========================================="
echo "  GO2W Quick Start Training"
echo "=========================================="
echo "Mode: $MODE"
echo "Task: $TASK"
echo "Project root: $PROJECT_ROOT"
echo ""
echo "Starting training..."
echo ""

# Change to project root directory
cd "$PROJECT_ROOT"

# Run training
python scripts/rsl_rl/train.py \
    --task "$TASK" \
    --num_envs 4096 \
    --device cuda:0 \
    --max_iterations 10000

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "  Training completed successfully!"
    echo "=========================================="
    echo ""
    echo "To view training results:"
    echo "  tensorboard --logdir logs/rsl_rl/"
    echo ""
    echo "To play with trained policy:"
    echo "  ./train_go2w.sh play-$MODE"
else
    echo ""
    echo "Training failed. Check logs for details."
    exit 1
fi
