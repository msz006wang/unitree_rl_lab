#!/bin/bash
# 检查环境并启动训练

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}G1机器人训练 - 环境检查与启动${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 切换到项目根目录
cd "$(dirname "$0")/.."

# 1. 检查conda环境
echo "1. 检查Conda环境..."
if conda env list | grep -q "env_isaaclab"; then
    echo -e "${GREEN}  ✅ 找到环境: env_isaaclab${NC}"
else
    echo -e "${YELLOW}  ⚠️  未找到环境: env_isaaclab${NC}"
    echo ""
    echo "请先创建Isaac Lab环境:"
    echo "  conda create -n env_isaaclab python=3.10"
    echo "  conda activate env_isaaclab"
    echo "  pip install isaaclab"
    exit 1
fi

# 2. 激活环境
echo ""
echo "2. 激活环境..."
source /home/jay/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
echo -e "${GREEN}  ✅ 环境已激活${NC}"

# 3. 检查关键模块
echo ""
echo "3. 检查关键模块..."

# 检查isaaclab
if python -c "import isaaclab" 2>/dev/null; then
    echo -e "${GREEN}  ✅ isaaclab 可用${NC}"
else
    echo -e "${RED}  ❌ isaaclab 不可用${NC}"
    echo "  请运行: pip install isaaclab"
    exit 1
fi

# 检查isaac sim
if python -c "import isaacsim" 2>/dev/null; then
    echo -e "${GREEN}  ✅ isaacsim 可用${NC}"
else
    echo -e "${YELLOW}  ⚠️  isaacsim 可能不可用（可能仍能正常训练）${NC}"
fi

# 4. 验证配置
echo ""
echo "4. 验证配置..."
if python scripts/validate_improved_config.sh 2>/dev/null || bash scripts/validate_improved_config.sh; then
    echo -e "${GREEN}  ✅ 配置验证通过${NC}"
else
    echo -e "${RED}  ❌ 配置验证失败${NC}"
    exit 1
fi

# 5. 显示训练选项
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}训练选项${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "请选择训练模式:"
echo ""
echo "  1) 快速测试 (改进配置, 512 envs, ~30分钟)"
echo "  2) 完整训练 (改进配置, 4096 envs, ~2-4小时)"
echo "  3) 快速测试 (原始配置, 512 envs, ~30分钟)"
echo "  4) 完整训练 (原始配置, 4096 envs, ~2-4小时)"
echo "  5) 自定义"
echo "  0) 退出"
echo ""

read -p "请输入选项 (0-5): " choice

case $choice in
    1)
        echo ""
        echo -e "${GREEN}开始快速测试 (改进配置)...${NC}"
        echo ""
        python scripts/rsl_rl/train.py \
            --task Unitree-G1-29dof-Velocity-Improved \
            --num_envs 512
        ;;
    2)
        echo ""
        echo -e "${GREEN}开始完整训练 (改进配置)...${NC}"
        echo ""
        python scripts/rsl_rl/train.py \
            --task Unitree-G1-29dof-Velocity-Improved \
            --num_envs 4096
        ;;
    3)
        echo ""
        echo -e "${GREEN}开始快速测试 (原始配置)...${NC}"
        echo ""
        python scripts/rsl_rl/train.py \
            --task Unitree-G1-29dof-Velocity \
            --num_envs 512
        ;;
    4)
        echo ""
        echo -e "${GREEN}开始完整训练 (原始配置)...${NC}"
        echo ""
        python scripts/rsl_rl/train.py \
            --task Unitree-G1-29dof-Velocity \
            --num_envs 4096
        ;;
    5)
        echo ""
        echo "自定义训练"
        echo ""
        read -p "任务 (Unitree-G1-29dof-Velocity 或 Unitree-G1-29dof-Velocity-Improved): " task
        read -p "环境数量 (推荐: 512, 2048, 4096): " num_envs
        echo ""
        python scripts/rsl_rl/train.py \
            --task "$task" \
            --num_envs "$num_envs"
        ;;
    0)
        echo ""
        echo "退出"
        exit 0
        ;;
    *)
        echo ""
        echo -e "${RED}无效选项${NC}"
        exit 1
        ;;
esac
