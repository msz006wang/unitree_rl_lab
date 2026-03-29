#!/bin/bash
# GO2W-Arm训练脚本
# 基于GO2W的训练策略和奖励函数

# 设置环境变量
export PYTHONPATH=/home/jay/unitree_rl_lab/source:$PYTHONPATH

echo "========================================"
echo "GO2W-Arm 训练脚本"
echo "========================================"
echo ""
echo "可用的训练环境:"
echo "1. ARX5机械臂 - 平地环境"
echo "2. ARX5机械臂 - 粗糙地形环境"
echo "3. Piper机械臂 - 平地环境"
echo "4. Piper机械臂 - 粗糙地形环境"
echo ""

# 检查用户选择的训练类型
TRAIN_TYPE=${1:-"arx5_flat"}

case $TRAIN_TYPE in
    "arx5_flat")
        echo "🚀 启动ARX5机械臂平地训练..."
        python /home/jay/unitree_rl_lab/scripts/rsl_rl/train.py --task Unitree-Go2WArm-Velocity-Flat-v0
        ;;
    "arx5_rough")
        echo "🚀 启动ARX5机械臂粗糙地形训练..."
        python /home/jay/unitree_rl_lab/scripts/rsl_rl/train.py --task Unitree-Go2WArm-Velocity-Rough-v0
        ;;
    "piper_flat")
        echo "🚀 启动Piper机械臂平地训练..."
        # 注意：需要修改velocity_env_cfg_piper.py为velocity_env_cfg.py
        python /home/jay/unitree_rl_lab/scripts/rsl_rl/train.py --task Unitree-Go2WArm-Velocity-Flat-v0
        ;;
    "piper_rough")
        echo "🚀 启动Piper机械臂粗糙地形训练..."
        # 注意：需要修改velocity_env_cfg_piper.py为velocity_env_cfg.py
        python /home/jay/unitree_rl_lab/scripts/rsl_rl/train.py --task Unitree-Go2WArm-Velocity-Rough-v0
        ;;
    *)
        echo "❌ 未知的训练类型: $TRAIN_TYPE"
        echo ""
        echo "使用方法:"
        echo "  ./train_go2w_arm.sh [arx5_flat|arx5_rough|piper_flat|piper_rough]"
        echo ""
        echo "默认: arx5_flat"
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo "训练完成！"
echo "========================================"
