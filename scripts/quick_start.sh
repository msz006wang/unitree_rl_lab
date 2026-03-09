#!/bin/bash
# 快速启动脚本 - G1 16级地形训练
# Quick Start Script - G1 16-Level Terrain Training

# 获取脚本所在目录的父目录（项目根目录）
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 切换到项目根目录
cd "$PROJECT_ROOT"

# 颜色定义 / Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息 / Print colored messages
print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# 显示帮助信息 / Show help information
show_help() {
    print_header "G1机器人16级地形训练快速启动 / G1 16-Level Terrain Training Quick Start"
    cat << EOF

使用方法 / Usage:
    $0 [选项] [参数]

选项 / Options:
    train               开始训练 (4096个环境) / Start training (4096 environments)
    train-small         开始训练 (512个环境，快速测试) / Start training (512 environments, quick test)
    train-improved      使用改进配置训练 (4096个环境) / Train with improved config (4096 environments)
    train-improved-small 使用改进配置训练 (512个环境) / Train with improved config (512 environments)
    visualize          可视化16个地形等级 / Visualize 16 terrain levels
    play               回放训练好的模型 / Playback trained model
    verify             验证配置文件 / Verify configuration files
    help               显示此帮助信息 / Show this help

示例 / Examples:
    $0 train           # 开始完整训练 / Start full training
    $0 train-small     # 快速测试 / Quick test
    $0 train-improved  # 使用改进配置训练 / Train with improved config
    $0 visualize       # 可视化地形 / Visualize terrains
    $0 play            # 回放模型 / Playback model

注意 / Note:
    - 首次运行前请确保已安装所有依赖 / Ensure all dependencies are installed before first run
    - 训练需要Isaac Sim运行环境 / Training requires Isaac Sim runtime environment
    - 更多详情请查看 docs/TERRAIN_CONFIG.md / See docs/TERRAIN_CONFIG.md for more details

EOF
}

# 验证配置 / Verify configuration
verify_config() {
    print_header "验证配置文件 / Verifying Configuration Files"
    python scripts/verify_config.py
    if [ $? -eq 0 ]; then
        print_success "配置验证通过 / Configuration verification passed"
        return 0
    else
        print_error "配置验证失败 / Configuration verification failed"
        return 1
    fi
}

# 开始训练 / Start training
start_training() {
    local num_envs=$1
    local task_name="Unitree-G1-29dof-Velocity"

    print_header "开始G1机器人训练 / Starting G1 Robot Training"
    echo "环境数量 / Number of environments: $num_envs"
    echo "地形等级 / Terrain levels: 16"
    echo "任务配置 / Task config: $task_name"
    echo ""

    # 检查配置 / Check configuration
    verify_config
    if [ $? -ne 0 ]; then
        print_error "配置验证失败，请检查配置文件 / Configuration verification failed"
        return 1
    fi

    print_success "开始训练... / Starting training..."
    echo ""

    # 启动训练 / Start training
    python scripts/rsl_rl/train.py \
        --task $task_name \
        --num_envs $num_envs
}

# 开始改进配置训练 / Start training with improved config
start_training_improved() {
    local num_envs=$1
    local task_name="Unitree-G1-29dof-Velocity-Improved"

    print_header "开始G1机器人训练 (改进配置) / Starting G1 Robot Training (Improved Config)"
    echo "环境数量 / Number of environments: $num_envs"
    echo "地形等级 / Terrain levels: 16"
    echo "任务配置 / Task config: $task_name"
    echo ""
    echo "改进特性 / Improvements:"
    echo "  ✅ 长时间行走支持 / Long-duration walking support"
    echo "  ✅ 摔倒恢复能力 / Fall recovery capability"
    echo "  ✅ 改进的Action空间 / Improved action space"
    echo ""

    # 检查配置 / Check configuration
    verify_config
    if [ $? -ne 0 ]; then
        print_error "配置验证失败，请检查配置文件 / Configuration verification failed"
        return 1
    fi

    print_success "开始训练... / Starting training..."
    echo ""

    # 启动训练 / Start training
    python scripts/rsl_rl/train.py \
        --task $task_name \
        --num_envs $num_envs
}

# 可视化地形 / Visualize terrains
visualize_terrains() {
    print_header "可视化16个地形等级 / Visualizing 16 Terrain Levels"
    echo "环境数量 / Number of environments: 16"
    echo "每个环境对应一个难度等级 / Each environment corresponds to one difficulty level"
    echo ""

    print_success "启动可视化... / Starting visualization..."
    echo ""

    # 启动可视化 / Start visualization
    python scripts/rsl_rl/visualize_terrains.py \
        --task Unitree-G1-29dof-Velocity \
        --num_envs 16 \
        --real-time
}

# 回放模型 / Playback model
play_model() {
    print_header "回放训练模型 / Playing Trained Model"
    echo "环境数量 / Number of environments: 32"
    echo ""

    print_success "启动回放... / Starting playback..."
    echo ""

    # 启动回放 / Start playback
    python scripts/rsl_rl/play.py \
        --task Unitree-G1-29dof-Velocity \
        --num_envs 32
}

# 主函数 / Main function
main() {
    case "$1" in
        train)
            start_training 4096
            ;;
        train-small)
            start_training 512
            ;;
        train-improved)
            start_training_improved 4096
            ;;
        train-improved-small)
            start_training_improved 512
            ;;
        visualize)
            visualize_terrains
            ;;
        play)
            play_model
            ;;
        verify)
            verify_config
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            print_error "未知选项 / Unknown option: $1"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# 运行主函数 / Run main function
if [ $# -eq 0 ]; then
    show_help
else
    main "$@"
fi
