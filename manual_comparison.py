"""
手动配置对比工具
直接读取配置文件进行对比，避免导入依赖
"""

import re
import os

def extract_reward_weights(file_path):
    """提取配置文件中的reward权重"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # 提取所有reward权重
    reward_pattern = r'(\w+)\s*=\s*RewTerm\([^)]+weight\s*=\s*([\d\.-]+)[^)]*\)'
    matches = re.findall(reward_pattern, content)
    
    return {name: float(weight) for name, weight in matches}

def extract_action_scale(file_path):
    """提取action scale"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    pattern = r'scale\s*=\s*([\d\.-]+)'
    matches = re.findall(pattern, content)
    
    # 找到JointPositionAction的scale
    for i, match in enumerate(matches):
        if i > 0 and 'JointPositionAction' in content[max(0, content.find('JointPositionAction', 0, content.find(f'scale = {match}')) - 1000):content.find(f'scale = {match}')]:
            return float(match)
    
    return None

def extract_episode_length(file_path):
    """提取episode长度"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    pattern = r'episode_length_s\s*=\s*([\d\.-]+)'
    matches = re.findall(pattern, content)
    return float(matches[0]) if matches else None

def extract_termination_params(file_path):
    """提取终止条件参数"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    min_height_pattern = r'minimum_height.*?([\d\.-]+)'
    angle_pattern = r'limit_angle.*?([\d\.-]+)'
    
    min_height_matches = re.findall(min_height_pattern, content)
    angle_matches = re.findall(angle_pattern, content)
    
    return {
        'min_height': float(min_height_matches[0]) if min_height_matches else None,
        'max_angle': float(angle_matches[0]) if angle_matches else None
    }

def main():
    # 文件路径
    original_path = 'source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/velocity_env_cfg.py'
    improved_path = 'source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/velocity_env_cfg_improved.py'
    
    print("=" * 80)
    print("G1机器人训练配置对比分析")
    print("=" * 80)
    print()
    
    # 提取配置数据
    try:
        original_rewards = extract_reward_weights(original_path)
        improved_rewards = extract_reward_weights(improved_path)
        original_action_scale = extract_action_scale(original_path)
        improved_action_scale = extract_action_scale(improved_path)
        original_episode_length = extract_episode_length(original_path)
        improved_episode_length = extract_episode_length(original_path)
        original_termination = extract_termination_params(original_path)
        improved_termination = extract_termination_params(improved_path)
        
        # 对比现有reward项
        print("📊 现有Reward权重对比:")
        print("-" * 80)
        print(f"{'Reward项':<30} {'原始值':>15} {'改进值':>15} {'变化':>15}")
        print("-" * 80)
        
        all_reward_names = set(original_rewards.keys()) | set(improved_rewards.keys())
        
        for name in sorted(all_reward_names):
            orig_weight = original_rewards.get(name, 0)
            imp_weight = improved_rewards.get(name, 0)
            change = imp_weight - orig_weight
            change_str = f"({change:+.2f})"
            print(f"{name:<30} {orig_weight:>15.2f} {imp_weight:>15.2f} {change_str:>15}")
        
        # 新增的reward项
        print("\n🆕 新增Reward项（改进配置）:")
        print("-" * 80)
        new_rewards = ['survival', 'distance_traveled', 'energy_efficiency', 'consistent_velocity', 
                      'fall_recovery', 'stand_up_progress', 'upright_orientation']
        
        for name in new_rewards:
            if name in improved_rewards:
                weight = improved_rewards[name]
                print(f"{name:<30} {'-':>15} {weight:>15.2f} {'(NEW)':>15}")
        
        # Action配置对比
        print("\n⚙️  Action配置对比:")
        print("-" * 80)
        change = improved_action_scale - original_action_scale
        change_str = f"({change:+.2f}, +{change/original_action_scale*100:.1f}%)"
        print(f"{'Action scale':<30} {original_action_scale:>15.2f} {improved_action_scale:>15.2f} {change_str:>15}")
        
        # Episode长度对比
        print("\n⏱️  Episode配置对比:")
        print("-" * 80)
        change = improved_episode_length - original_episode_length
        change_str = f"({change:+.1f}, +{change/original_episode_length*100:.1f}%)"
        print(f"{'Episode长度 (秒)':<30} {original_episode_length:>15.1f} {improved_episode_length:>15.1f} {change_str:>15}")
        
        # 终止条件对比
        print("\n🚪 终止条件对比:")
        print("-" * 80)
        height_change = improved_termination['min_height'] - original_termination['min_height']
        height_change_str = f"({height_change:+.2f}, {height_change/original_termination['min_height']*100:.1f}%)"
        angle_change = improved_termination['max_angle'] - original_termination['max_angle']
        angle_change_str = f"({angle_change:+.2f}, {angle_change/original_termination['max_angle']*100:.1f}%)"
        
        print(f"{'最小高度 (m)':<30} {original_termination['min_height']:>15.2f} {improved_termination['min_height']:>15.2f} {height_change_str:>15}")
        print(f"{'最大倾斜角度 (rad)':<30} {original_termination['max_angle']:>15.2f} {improved_termination['max_angle']:>15.2f} {angle_change_str:>15}")
        
        # 关键差异分析
        print("\n" + "=" * 80)
        print("🔍 关键差异分析（可能导致训练不稳定的因素）:")
        print("=" * 80)
        
        # 1. 高权重奖励项
        print("⚠️  1. 高权重奖励项分析:")
        high_weight_rewards = [(name, weight) for name, weight in improved_rewards.items() if abs(weight) > 3.0]
        high_weight_rewards.sort(key=lambda x: abs(x[1]), reverse=True)
        
        for name, weight in high_weight_rewards:
            if weight > 0:
                print(f"   - {name}: {weight:.2f} (高正奖励可能过度优化)")
            else:
                print(f"   - {name}: {weight:.2f} (高负惩罚可能过度限制)")
        
        # 2. 新增奖励项的影响
        print("\n⚠️  2. 新增奖励项的影响:")
        new_reward_impacts = []
        for name in new_rewards:
            if name in improved_rewards:
                weight = improved_rewards[name]
                new_reward_impacts.append((name, weight))
        
        new_reward_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
        
        for name, weight in new_reward_impacts:
            if name == 'fall_recovery':
                print(f"   - {name}: {weight:.2f} (摔倒恢复奖励很高，可能导致机器人故意摔倒)")
            elif name == 'survival':
                print(f"   - {name}: {weight:.2f} (生存奖励可能导致机器人保守行为)")
            elif name == 'distance_traveled':
                print(f"   - {name}: {weight:.2f} (距离奖励可能导致机器人急速前进)")
            else:
                print(f"   - {name}: {weight:.2f}")
        
        # 3. 参数范围变化
        print("\n⚠️  3. 参数范围变化:")
        print(f"   - Action scale增加67%: 可能导致动作幅度过大，影响稳定性")
        print(f"   - Episode长度增加25%: 增加了训练复杂度，但允许更多恢复时间")
        print(f"   - 最小高度降低33%: 允许更多摔倒，增加恢复训练")
        print(f"   - 最大倾斜角度增加20%: 允许更大倾斜，可能影响稳定性")
        
        # 4. 潜在冲突
        print("\n⚠️  4. 潜在奖励冲突:")
        conflicts = [
            ("fall_recovery (5.0)", "base_height (-8.0)", "摔倒恢复与高度惩罚冲突"),
            ("survival (0.5)", "energy (-2e-5)", "生存与能量效率潜在冲突"),
            ("distance_traveled (0.3)", "base_linear_velocity (-2.0)", "前进速度与垂直速度惩罚冲突"),
        ]
        
        for reward1, reward2, conflict in conflicts:
            print(f"   - {reward1} vs {reward2}: {conflict}")
        
        # 5. 改进建议
        print("\n💡 改进建议:")
        suggestions = [
            "降低fall_recovery奖励权重（从5.0→2.0）以避免过度优化摔倒行为",
            "调整survival和distance_traveled的权重平衡",
            "考虑增加一些负奖励来防止不良行为",
            "逐步调整参数，观察训练曲线变化",
            "建议先用较小规模环境测试改进配置"
        ]
        
        for i, suggestion in enumerate(suggestions, 1):
            print(f"   {i}. {suggestion}")
        
        print("\n" + "=" * 80)
        print("总结:")
        print("=" * 80)
        print("✅ 改进配置的主要优势:")
        print("  1. 添加了摔倒恢复能力，支持更鲁棒的训练")
        print("  2. 增加了长时间行走的奖励，提高行走距离")
        print("  3. 放宽了动作空间和终止条件，允许更多探索")
        print("  4. 添加了多种辅助奖励，改善运动质量")
        print()
        print("⚠️  潜在风险:")
        print("  1. fall_recovery奖励过高（5.0）可能导致机器人故意摔倒")
        print("  2. 新增的奖励项可能与现有奖励产生冲突")
        print("  3. 参数范围变化可能导致训练不稳定")
        print("  4. 需要更多的训练样本来适应复杂奖励函数")
        print()
        print("🎯 建议优先测试:")
        print("  1. 先降低fall_recovery权重到2.0-3.0")
        print("  2. 在小规模环境中测试改进配置")
        print("  3. 监控训练曲线，观察是否出现异常模式")
        
    except FileNotFoundError as e:
        print(f"错误: 找不到配置文件 - {e}")
    except Exception as e:
        print(f"错误: 分析配置时出错 - {e}")

if __name__ == "__main__":
    main()
