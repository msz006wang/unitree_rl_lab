#!/bin/bash
# ============================================================================
# Training Script for Unitree G1 Robot
# ============================================================================
# This script provides convenient training options for G1 humanoid robot
# on both Original and Improved configurations with Flat terrain support.
#
# Usage:
#   ./train_g1.sh [original|improved|play] [options]
#
# Examples:
#   ./train_g1.sh original                   # Train with original config
#   ./train_g1.sh improved                    # Train with improved config
#   ./train_g1.sh original --num_envs 4096  # Train with 4096 environments
#   ./train_g1.sh improved --headless           # Train without GUI
#   ./train_g1.sh play-original                # Play with trained policy
# ============================================================================

set -e # Exit on error

# Get directory where this script is located
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
    original        Train with original configuration (16-level progressive terrain)
    improved        Train with improved configuration (16-level progressive terrain)
    flat-original   Train with original config on flat terrain
    flat-improved   Train with improved config on flat terrain
    play-original   Play with trained policy (original config)
    play-improved   Play with trained policy (improved config)

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
    $0 original                              # Train with original config (progressive terrain)
    $0 improved                              # Train with improved config (progressive terrain)
    $0 flat-original                         # Train on flat terrain (original config)
    $0 flat-improved                         # Train on flat terrain (improved config)
    $0 flat-improved --num_envs 512          # Quick test on flat terrain
    $0 original --gui --video                 # Train with GUI and video recording
    $0 play-improved --load_run recent         # Play with recent checkpoint

EOF
    exit 1
}

# Parse command line arguments
MODE=""
EXTRA_ARGS=""
RESUME=""

# Check if help is requested
if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
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
    original)
        TASK="Unitree-G1-29dof-Velocity"
        print_info "Training with ORIGINAL config (16-level progressive terrain)"
        ;;
    improved)
        TASK="Unitree-G1-29dof-Velocity-Improved"
        print_info "Training with IMPROVED config (16-level progressive terrain)"
        ;;
    flat-original)
        TASK="Unitree-G1-29dof-Velocity-Flat"
        print_info "Training on FLAT terrain (original config)"
        ;;
    flat-improved)
        TASK="Unitree-G1-29dof-Velocity-Flat-Improved"
        print_info "Training on FLAT terrain (improved config)"
        ;;
    play-original)
        TASK="Unitree-G1-29dof-Velocity"
        RESUME="--resume"
        HEADLESS=""  # Always show GUI for play mode
        print_info "Playing with ORIGINAL config"
        ;;
    play-improved)
        TASK="Unitree-G1-29dof-Velocity-Improved"
        RESUME="--resume"
        HEADLESS=""  # Always show GUI for play mode
        print_info "Playing with IMPROVED config"
        ;;
    *)
        print_error "Unknown mode: $MODE"
        usage
        ;;
esac

# Display configuration
echo ""
echo "=========================================="
echo "  G1 Training Configuration"
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
