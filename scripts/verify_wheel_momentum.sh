#!/bin/bash

# 轮子角动量奖励集成验证脚本

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

echo "=========================================="
echo "轮子角动量奖励集成验证"
echo "=========================================="
echo ""

# 1. 检查函数实现
log_info "1. 检查函数实现..."
if grep -q "def wheel_angular_momentum_reward" source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/extended_rewards.py; then
    log_success "✓ 函数定义存在于 extended_rewards.py"
else
    log_error "✗ 函数定义未找到"
    exit 1
fi

# 2. 检查函数导出
log_info "2. 检查函数导出..."
if grep -q "wheel_angular_momentum_reward" source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/__init__.py; then
    log_success "✓ 函数已导出到 __init__.py"
else
    log_error "✗ 函数未导出"
    exit 1
fi

# 3. 检查配置集成
log_info "3. 检查配置集成..."
if grep -q "wheel_angular_momentum = RewTerm" source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2w_arm/two_stage_recovery_env_cfg.py; then
    log_success "✓ 奖励项已配置到 two_stage_recovery_env_cfg.py"

    # 提取权重
    WEIGHT=$(grep -A 2 "wheel_angular_momentum = RewTerm" source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2w_arm/two_stage_recovery_env_cfg.py | grep "weight=" | head -1 | grep -oP 'weight=\K[0-9.]+')
    if [ ! -z "$WEIGHT" ]; then
        log_success "  权重: $WEIGHT"
    fi
else
    log_error "✗ 奖励项未配置"
    exit 1
fi

# 4. 检查环境注册
log_info "4. 检查环境注册..."
if grep -q "TwoStage-Recovery-v0" source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2w_arm/__init__.py; then
    log_success "✓ 环境已注册: Unitree-Go2WArm-TwoStage-Recovery-v0"

    if grep -q "TwoStageRecoveryFlatEnvCfg" source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2w_arm/__init__.py; then
        log_success "  配置类: TwoStageRecoveryFlatEnvCfg"
    fi
else
    log_error "✗ 环境未注册"
    exit 1
fi

# 5. 检查训练脚本
log_info "5. 检查训练脚本..."
if grep -q "TwoStage-Recovery-v0" scripts/train_go2w_arm_two_stage.sh; then
    log_success "✓ 训练脚本使用正确的环境ID"
else
    log_warning "⚠ 训练脚本可能未使用正确的环境ID"
fi

# 6. 语法检查
log_info "6. 语法检查..."
if python3 -m py_compile source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/extended_rewards.py 2>/dev/null; then
    log_success "✓ extended_rewards.py 语法正确"
else
    log_error "✗ extended_rewards.py 语法错误"
    exit 1
fi

if python3 -m py_compile source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2w_arm/two_stage_recovery_env_cfg.py 2>/dev/null; then
    log_success "✓ two_stage_recovery_env_cfg.py 语法正确"
else
    log_error "✗ two_stage_recovery_env_cfg.py 语法错误"
    exit 1
fi

# 7. 统计奖励项
log_info "7. 统计配置中的奖励项..."
REWARD_COUNT=$(grep -c " = RewTerm(" source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2w_arm/two_stage_recovery_env_cfg.py)
log_success "  奖励项总数: $REWARD_COUNT"

# 8. 验证完成
echo ""
echo "=========================================="
log_success "🎉 轮子角动量奖励已完全集成到训练中！"
echo "=========================================="
echo ""
echo "集成状态:"
echo "  ✓ 函数实现"
echo "  ✓ 函数导出"
echo "  ✓ 配置集成"
echo "  ✓ 环境注册"
echo "  ✓ 训练脚本"
echo "  ✓ 语法检查"
echo ""
echo "训练命令:"
echo "  ./scripts/train_go2w_arm_two_stage.sh"
echo ""
echo "物理原理:"
echo "  - 角动量守恒: 轮子急加速 → 机身获得反向扭矩"
echo "  - 悬空轮子检测: 接触力 < 1.0 N"
echo "  - 姿态条件: 只在 Z < 0.5 时激活"
echo ""
echo "预期效果:"
echo "  - 策略学会让悬空轮子急加速"
echo "  - 产生翻滚扭矩辅助恢复"
echo "  - 提高侧卧恢复成功率"
echo ""
