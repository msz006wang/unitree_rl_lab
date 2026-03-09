#!/bin/bash
# Go2W机器人启动脚本 / Go2W Robot Launch Script
# 用于在IsaacLab中启动和测试Go2W轮足机器人 / For launching and testing Go2W wheel-legged robot in IsaacLab

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
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 打印带颜色的消息 / Print colored messages
print_header() {
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}========================================${NC}"
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

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# 显示帮助信息 / Show help information
show_help() {
    print_header "Go2W轮足机器人启动脚本 / Go2W Wheel-Legged Robot Launch Script"
    cat << EOF

使用方法 / Usage:
    $0 [选项] [参数]

选项 / Options:
    spawn               生成Go2W机器人到IsaacLab场景（使用URDF默认材质） / Spawn Go2W robot in IsaacLab (default URDF materials)
    spawn-materials     生成Go2W机器人并应用PBR材质 / Spawn Go2W robot with PBR materials
    spawn-mdl           生成Go2W机器人并应用高质量MDL材质 / Spawn Go2W robot with high-quality MDL materials
    test                测试Go2W配置和导入 / Test Go2W configuration and import
    train               开始训练 (4096个环境) / Start training (4096 environments)
    train-small         开始训练 (512个环境，快速测试) / Start training (512 environments, quick test)
    play                回放训练好的模型 / Playback trained model
    verify              验证配置文件 / Verify configuration files
    help                显示此帮助信息 / Show this help

示例 / Examples:
    $0 spawn           # 在IsaacLab中生成Go2W机器人（默认材质）/ Spawn Go2W (default materials)
    $0 spawn-materials # 生成Go2W机器人并应用PBR材质 / Spawn with PBR materials
    $0 spawn-mdl       # 生成Go2W机器人并应用MDL材质（高质量）/ Spawn with MDL materials (high quality)
    $0 test            # 测试配置是否正确 / Test if configuration is correct
    $0 train           # 开始完整训练 / Start full training
    $0 train-small     # 快速测试训练 / Quick test training
    $0 play            # 回放模型 / Playback model

配置 / Configuration:
    机器人配置 / Robot Config: source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py
    配置名称 / Config Name: UNITREE_GO2W_CFG
    关节数量 / Joint Count: 16 (12 leg joints + 4 wheel joints)

环境要求 / Environment Requirements:
    需要激活 IsaacLab 的 conda 环境 / Requires IsaacLab conda environment activated

    激活命令 / Activation command:
        conda activate env_isaaclab

注意 / Note:
    - 需要IsaacLab运行环境 / Requires IsaacLab runtime environment
    - 确保URDF文件存在于指定路径 / Ensure URDF file exists at specified path
    - 更多信息: scripts/README_GO2W_TEST.md / More info: scripts/README_GO2W_TEST.md

EOF
}

# 验证配置 / Verify configuration
verify_config() {
    print_header "验证Go2W配置文件 / Verifying Go2W Configuration Files"
    echo ""

    print_info "运行配置验证脚本... / Running configuration verification script..."
    echo ""

    python scripts/verify_go2w_config.py

    if [ $? -eq 0 ]; then
        echo ""
        print_success "配置验证通过 / Configuration verification passed"
        return 0
    else
        echo ""
        print_error "配置验证失败 / Configuration verification failed"
        return 1
    fi
}

# 在IsaacLab中生成Go2W机器人 / Spawn Go2W robot in IsaacLab
spawn_go2w() {
    print_header "在IsaacLab中生成Go2W机器人 / Spawning Go2W Robot in IsaacLab"
    echo ""

    print_info "配置信息 / Configuration Info:"
    echo "  机器人类型 / Robot Type: Unitree Go2W (Wheel-Legged)"
    echo "  配置文件 / Config File: unitree.py"
    echo "  配置对象 / Config Object: UNITREE_GO2W_CFG"
    echo "  资产格式 / Asset Format: URDF"
    echo "  执行器组 / Actuator Groups: legs (12), wheels (4)"
    echo ""

    # 检查 conda 环境 / Check conda environment
    if [ -z "$CONDA_DEFAULT_ENV" ]; then
        print_warning "未检测到激活的 conda 环境 / No active conda environment detected"
        echo ""
        print_info "请先激活 IsaacLab 环境 / Please activate IsaacLab environment first:"
        echo "  conda activate env_isaaclab"
        echo ""
        return 1
    fi

    print_success "Conda 环境 / Conda environment: $CONDA_DEFAULT_ENV"

    # 检查配置 / Check configuration
    verify_config
    if [ $? -ne 0 ]; then
        print_error "配置验证失败，请先修复配置问题 / Configuration verification failed"
        return 1
    fi

    echo ""
    print_success "启动IsaacLab并生成Go2W机器人... / Launching IsaacLab and spawning Go2W robot..."
    echo ""
    print_warning "提示 / Tips:"
    echo "  - 使用鼠标左键旋转视角 / Left mouse button to rotate view"
    echo "  - 使用鼠标滚轮缩放 / Mouse wheel to zoom"
    echo "  - 使用鼠标中键平移 / Middle mouse button to pan"
    echo "  - 按Ctrl+C或关闭窗口退出仿真 / Press Ctrl+C or close window to exit simulation"
    echo ""
    print_warning "首次启动可能较慢，请耐心等待... / First launch may be slow, please wait..."
    echo ""

    # 直接使用 python 运行脚本（依赖 conda 环境） / Use python directly (depends on conda env)
    # 不使用 --headless 以显示 GUI 窗口 / Don't use --headless to show GUI window
    python scripts/go2w_spawn.py
}

# 在IsaacLab中生成Go2W机器人并应用材质 / Spawn Go2W robot with materials in IsaacLab
spawn_go2w_with_materials() {
    local use_mdl=$1

    if [ "$use_mdl" = "true" ]; then
        print_header "在IsaacLab中生成Go2W机器人（MDL材质）/ Spawning Go2W Robot with MDL Materials"
    else
        print_header "在IsaacLab中生成Go2W机器人（PBR材质）/ Spawning Go2W Robot with PBR Materials"
    fi
    echo ""

    print_info "配置信息 / Configuration Info:"
    echo "  机器人类型 / Robot Type: Unitree Go2W (Wheel-Legged)"
    echo "  材质类型 / Material Type: $([ "$use_mdl" = "true" ] && echo "MDL (High Quality)" || echo "PreviewSurface (PBR)")"
    echo "  资产格式 / Asset Format: URDF"
    echo "  执行器组 / Actuator Groups: legs (12), wheels (4)"
    echo ""

    # 检查 conda 环境 / Check conda environment
    if [ -z "$CONDA_DEFAULT_ENV" ]; then
        print_warning "未检测到激活的 conda 环境 / No active conda environment detected"
        echo ""
        print_info "请先激活 IsaacLab 环境 / Please activate IsaacLab environment first:"
        echo "  conda activate env_isaaclab"
        echo ""
        return 1
    fi

    print_success "Conda 环境 / Conda environment: $CONDA_DEFAULT_ENV"

    # 检查配置 / Check configuration
    verify_config
    if [ $? -ne 0 ]; then
        print_error "配置验证失败，请先修复配置问题 / Configuration verification failed"
        return 1
    fi

    echo ""
    print_success "启动IsaacLab并生成Go2W机器人（带材质）... / Launching IsaacLab and spawning Go2W robot with materials..."
    echo ""

    if [ "$use_mdl" = "true" ]; then
        print_info "使用NVIDIA MDL材质库 / Using NVIDIA MDL material library"
        print_warning "需要访问NVIDIA Nucleus / Requires access to NVIDIA Nucleus"
        echo ""
    else
        print_info "使用PreviewSurface材质 / Using PreviewSurface materials"
        echo ""
    fi

    print_warning "提示 / Tips:"
    echo "  - 使用鼠标左键旋转视角 / Left mouse button to rotate view"
    echo "  - 使用鼠标滚轮缩放 / Mouse wheel to zoom"
    echo "  - 使用鼠标中键平移 / Middle mouse button to pan"
    echo "  - 按Ctrl+C或关闭窗口退出仿真 / Press Ctrl+C or close window to exit simulation"
    echo ""
    print_warning "首次启动可能较慢，请耐心等待... / First launch may be slow, please wait..."
    echo ""

    # 直接使用 python 运行脚本（依赖 conda 环境） / Use python directly (depends on conda env)
    if [ "$use_mdl" = "true" ]; then
        python scripts/go2w_spawn_with_materials.py --use-mdl
    else
        python scripts/go2w_spawn_with_materials.py
    fi
}

# 测试Go2W配置 / Test Go2W configuration
test_go2w() {
    print_header "测试Go2W配置 / Testing Go2W Configuration"
    echo ""

    print_info "步骤1: 验证配置文件 / Step 1: Verify configuration file"
    verify_config
    if [ $? -ne 0 ]; then
        return 1
    fi

    echo ""
    print_info "步骤2: 检查Conda环境 / Step 2: Check Conda environment"
    echo ""

    if [ -z "$CONDA_DEFAULT_ENV" ]; then
        print_warning "未检测到激活的 conda 环境 / No active conda environment detected"
        echo ""
        print_info "请先激活 IsaacLab 环境 / Please activate IsaacLab environment first:"
        echo "  conda activate env_isaaclab"
        echo ""
        return 1
    fi

    print_success "Conda 环境 / Conda environment: $CONDA_DEFAULT_ENV"

    echo ""
    print_success "所有测试通过！/ All tests passed!"
    echo ""
    print_info "下一步 / Next steps:"
    echo "  - 运行 '$0 spawn' 在IsaacLab中查看机器人 / Run '$0 spawn' to view robot in IsaacLab"
    echo "  - 运行 '$0 train' 开始训练 / Run '$0 train' to start training"

    return 0
}

# 开始训练 / Start training
start_training() {
    local num_envs=$1
    local task_name="Unitree-Go2W-Velocity"

    print_header "开始Go2W机器人训练 / Starting Go2W Robot Training"
    echo "环境数量 / Number of environments: $num_envs"
    echo "任务配置 / Task config: $task_name"
    echo ""

    # 检查 conda 环境 / Check conda environment
    if [ -z "$CONDA_DEFAULT_ENV" ]; then
        print_error "未检测到激活的 conda 环境 / No active conda environment detected"
        print_info "请先激活 IsaacLab 环境 / Please activate IsaacLab environment first:"
        echo "  conda activate env_isaaclab"
        return 1
    fi

    # 检查配置 / Check configuration
    verify_config
    if [ $? -ne 0 ]; then
        print_error "配置验证失败，请检查配置文件 / Configuration verification failed"
        return 1
    fi

    print_success "开始训练... / Starting training..."
    echo ""
    print_warning "注意 / Note: 训练任务配置文件需要预先创建 / Training task config must be created first"
    echo ""

    # 直接使用 python 启动训练（依赖 conda 环境） / Use python directly (depends on conda env)
    python scripts/rsl_rl/train.py \
        --task $task_name \
        --num_envs $num_envs
}

# 回放模型 / Playback model
play_model() {
    print_header "回放Go2W训练模型 / Playing Go2W Trained Model"
    echo "环境数量 / Number of environments: 32"
    echo ""

    # 检查 conda 环境 / Check conda environment
    if [ -z "$CONDA_DEFAULT_ENV" ]; then
        print_error "未检测到激活的 conda 环境 / No active conda environment detected"
        print_info "请先激活 IsaacLab 环境 / Please activate IsaacLab environment first:"
        echo "  conda activate env_isaaclab"
        return 1
    fi

    # 检查配置 / Check configuration
    verify_config
    if [ $? -ne 0 ]; then
        print_error "配置验证失败，请检查配置文件 / Configuration verification failed"
        return 1
    fi

    print_success "启动回放... / Starting playback..."
    echo ""

    # 直接使用 python 启动回放（依赖 conda 环境） / Use python directly (depends on conda env)
    python scripts/rsl_rl/play.py \
        --task Unitree-Go2W-Velocity \
        --num_envs 32
}

# 主函数 / Main function
main() {
    # 检查参数 / Check arguments
    if [ $# -eq 0 ]; then
        show_help
        exit 0
    fi

    case "$1" in
        spawn)
            spawn_go2w
            ;;
        spawn-materials)
            spawn_go2w_with_materials false
            ;;
        spawn-mdl)
            spawn_go2w_with_materials true
            ;;
        test)
            test_go2w
            ;;
        train)
            start_training 4096
            ;;
        train-small)
            start_training 512
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
main "$@"
