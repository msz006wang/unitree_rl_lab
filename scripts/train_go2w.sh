#!/bin/bash
# ============================================================================
# Training Script for Unitree GO2W Robot
# ============================================================================
# This script provides convenient training options for the GO2W wheel-legged robot
# on both Flat and Rough terrain configurations.
#
# Usage:
#   ./train_go2w.sh [flat|rough|play] [options]
#
# Examples:
#   ./train_go2w.sh flat                    # Train on flat terrain
#   ./train_go2w.sh rough                   # Train on rough terrain
#   ./train_go2w.sh flat --num_envs 4096   # Train with 4096 environments
#   ./train_go2w.sh rough --headless        # Train without GUI
#   ./train_go2w.sh play                    # Play with trained policy
# ============================================================================

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root directory
cd "$PROJECT_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default parameters
NUM_ENVS=4096
HEADLESS=""  # Default to show GUI
DEVICE="cuda:0"
MAX_ITERATIONS=10000
SEED=42
VIDEO=false

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to display usage
usage() {
    cat << EOF
Usage: $0 [MODE] [OPTIONS]

Modes:
    flat            Train on flat terrain (plane)
    rough           Train on rough terrain (generated terrain)
    play-flat       Play with trained policy on flat terrain
    play-rough      Play with trained policy on rough terrain

Options:
    --num_envs N    Number of parallel environments (default: 4096)
    --headless      Run without GUI (default: enabled)
    --gui           Enable GUI visualization
    --device D      Device to use (default: cuda:0)
    --iterations N  Maximum training iterations (default: 10000)
    --seed N        Random seed (default: 42)
    --video         Record videos during training
    --resume        Resume from last checkpoint
    --help          Show this help message

Examples:
    $0 flat                              # Train on flat terrain with default settings
    $0 rough --num_envs 8192             # Train on rough terrain with 8192 envs
    $0 flat --gui --video                # Train with GUI and video recording
    $0 play-flat --load_run recent       # Play with recent checkpoint

EOF
    exit 1
}

# Parse command line arguments
MODE=""
EXTRA_ARGS=""
RESUME=""

if [ $# -eq 0 ]; then
    usage
fi

MODE=$1
shift

# Parse remaining arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num_envs)
            NUM_ENVS="$2"
            shift 2
            ;;
        --headless)
            HEADLESS="--headless"
            shift
            ;;
        --gui)
            HEADLESS=""
            shift
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --iterations)
            MAX_ITERATIONS="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --video)
            VIDEO=true
            shift
            ;;
        --resume)
            RESUME="--resume"
            shift
            ;;
        --help|-h)
            usage
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# Set task based on mode
case $MODE in
    flat)
        TASK="Unitree-Go2W-Velocity-Flat-v0"
        print_info "Training on FLAT terrain"
        ;;
    rough)
        TASK="Unitree-Go2W-Velocity-Rough-v0"
        print_info "Training on ROUGH terrain"
        ;;
    play-flat)
        TASK="Unitree-Go2W-Velocity-Flat-v0"
        RESUME="--resume"
        HEADLESS=""  # Always show GUI for play mode
        print_info "Playing on FLAT terrain"
        ;;
    play-rough)
        TASK="Unitree-Go2W-Velocity-Rough-v0"
        RESUME="--resume"
        HEADLESS=""  # Always show GUI for play mode
        print_info "Playing on ROUGH terrain"
        ;;
    *)
        print_error "Unknown mode: $MODE"
        usage
        ;;
esac

# Display configuration
echo ""
echo "=========================================="
echo "  GO2W Training Configuration"
echo "=========================================="
echo "Task:              $TASK"
echo "Environments:      $NUM_ENVS"
echo "Device:            $DEVICE"
echo "Max Iterations:    $MAX_ITERATIONS"
echo "Seed:              $SEED"
echo "Video:             $VIDEO"
echo "Headless:          $([ -n "$HEADLESS" ] && echo "Yes" || echo "No")"
echo "Resume:            $([ -n "$RESUME" ] && echo "Yes" || echo "No")"
echo "=========================================="
echo ""

# Build command (SCRIPT_DIR is now set to project root)
CMD="python scripts/rsl_rl/train.py \
    --task $TASK \
    --num_envs $NUM_ENVS \
    --device $DEVICE \
    --max_iterations $MAX_ITERATIONS \
    --seed $SEED \
    $HEADLESS \
    $RESUME \
    $EXTRA_ARGS"

# Add video recording if enabled
if [ "$VIDEO" = true ]; then
    CMD="$CMD --video --video_interval 2000"
fi

# Run training
print_info "Starting training..."
print_info "Command: $CMD"
echo ""

# Execute
eval $CMD

# Check exit status
if [ $? -eq 0 ]; then
    print_info "Training completed successfully!"
else
    print_error "Training failed with exit code $?"
    exit 1
fi
