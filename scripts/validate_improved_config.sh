#!/bin/bash
# 验证改进的G1配置
# Validates improved G1 configuration

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}验证改进的G1配置${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 切换到项目根目录
cd "$(dirname "$0")/.."

# 检查文件存在性
echo "检查文件存在性..."
files=(
    "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/extended_rewards.py"
    "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/velocity_env_cfg_improved.py"
    "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/actions_cfg.py"
)

all_files_exist=true
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo -e "  ${GREEN}✅${NC} $(basename $file)"
    else
        echo -e "  ${RED}❌${NC} $(basename $file) 不存在"
        all_files_exist=false
    fi
done

if [ "$all_files_exist" = false ]; then
    echo -e "${RED}配置验证失败：缺少必要文件${NC}"
    exit 1
fi

echo ""

# 检查扩展的reward函数
echo "检查扩展的reward函数..."
rewards=(
    "survival_reward"
    "distance_traveled_reward"
    "energy_efficiency_reward"
    "consistent_velocity_reward"
    "is_fallen"
    "fall_recovery_reward"
    "stand_up_progress_reward"
    "upright_orientation_reward"
)

rewards_file="source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/extended_rewards.py"
all_rewards_found=true
for reward in "${rewards[@]}"; do
    if grep -q "$reward" "$rewards_file" 2>/dev/null; then
        echo -e "  ${GREEN}✅${NC} $reward"
    else
        echo -e "  ${RED}❌${NC} $reward 未找到"
        all_rewards_found=false
    fi
done

if [ "$all_rewards_found" = false ]; then
    echo -e "${RED}配置验证失败：缺少reward函数${NC}"
    exit 1
fi

echo ""

# 检查改进配置的关键特性
echo "检查改进配置的关键特性..."
config_file="source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/velocity_env_cfg_improved.py"

features=(
    "extended_rewards"
    "survival = RewTerm"
    "fall_recovery = RewTerm"
    "distance_traveled = RewTerm"
    "stand_up_progress = RewTerm"
    "scale=0.35"
    "episode_length_s = 25.0"
)

all_features_found=true
for feature in "${features[@]}"; do
    if grep -q "$feature" "$config_file" 2>/dev/null; then
        echo -e "  ${GREEN}✅${NC} $feature"
    else
        echo -e "  ${RED}❌${NC} $feature 未找到"
        all_features_found=false
    fi
done

if [ "$all_features_found" = false ]; then
    echo -e "${RED}配置验证失败：缺少关键特性${NC}"
    exit 1
fi

echo ""

# 检查任务注册
echo "检查任务注册..."
init_file="source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/__init__.py"

if grep -q "Unitree-G1-29dof-Velocity-Improved" "$init_file" 2>/dev/null; then
    echo -e "  ${GREEN}✅${NC} Unitree-G1-29dof-Velocity-Improved"
else
    echo -e "  ${RED}❌${NC} Unitree-G1-29dof-Velocity-Improved 未注册"
    exit 1
fi

echo ""

# 检查Python语法
echo "检查Python语法..."
python_syntax_ok=true

if python3 -m py_compile "$rewards_file" 2>/dev/null; then
    echo -e "  ${GREEN}✅${NC} extended_rewards.py 语法正确"
else
    echo -e "  ${RED}❌${NC} extended_rewards.py 语法错误"
    python_syntax_ok=false
fi

if python3 -m py_compile "$config_file" 2>/dev/null; then
    echo -e "  ${GREEN}✅${NC} velocity_env_cfg_improved.py 语法正确"
else
    echo -e "  ${RED}❌${NC} velocity_env_cfg_improved.py 语法错误"
    python_syntax_ok=false
fi

if [ "$python_syntax_ok" = false ]; then
    echo -e "${RED}配置验证失败：Python语法错误${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}🎉 所有验证通过！${NC}"
exit 0
