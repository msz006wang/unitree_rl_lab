import sys
from pathlib import Path

# Add project source to path
sys.path.append(str(Path(__file__).parent / "source" / "unitree_rl_lab"))

import glob
import numpy as np
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def read_tensorboard_logs(log_dir):
    """读取tensorboard日志文件"""
    print(f"正在读取日志目录: {log_dir}")

    # 查找所有日志文件（包括子目录）
    log_files = glob.glob(str(Path(log_dir) / "**" / "events.out.tfevents.*"), recursive=True)

    if not log_files:
        print(f"错误: 在 {log_dir} 中未找到tensorboard日志文件")
        return None

    print(f"找到 {len(log_files)} 个日志文件")

    # 使用最新的事件文件
    log_files.sort()
    latest_log = log_files[-1]
    print(f"使用最新的日志文件: {latest_log}")

    # 加载事件数据
    ea = event_accumulator.EventAccumulator(latest_log)
    ea.Reload()

    print("\n可用的标量:")
    for tag in ea.Tags()['scalars']:
        print(f"  - {tag}")

    # 读取所有标量数据
    data = {}
    for tag in ea.Tags()['scalars']:
        try:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            data[tag] = {'steps': steps, 'values': values}
            print(f"已读取 {tag}: {len(steps)} 个数据点")
        except Exception as e:
            print(f"读取 {tag} 时出错: {e}")

    return data


def analyze_key_mechanisms(data):
    """分析关键机制"""
    print("\n" + "=" * 80)
    print("关键机制分析")
    print("=" * 80)

    # 关键指标映射 - 根据实际的tensorboard标签调整
    key_metrics = {
        '三段式动态刹车': [
            'Episode_Reward/action_rate_l2',
            'Episode_Reward/joint_acc_l2',
            'Episode_Reward/wheel_angular_momentum',
        ],
        '角动量阻尼': [
            'Episode_Reward/angular_momentum_damping',
            'Episode_Reward/wheel_angular_momentum',
        ],
        '驻留成功提前终止': [
            'Episode_Termination/success_stable',
            'Episode_Reward/success_stable_reward',
        ],
        '多级姿态恢复课程': [
            'Curriculum/posture_curriculum',
            'Episode_Reward/orientation_improvement',
            'Episode_Reward/height_improvement',
        ],
        '判定站立成功奖励': [
            'Episode_Reward/success_stable_reward',
            'Episode_Reward/orientation_improvement',
        ]
    }

    for mechanism, metrics in key_metrics.items():
        print(f"\n【{mechanism}】")
        print("-" * 80)

        for metric in metrics:
            if metric in data:
                steps = data[metric]['steps']
                values = data[metric]['values']

                if len(values) > 0:
                    # 计算统计信息
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    min_val = np.min(values)
                    max_val = np.max(values)
                    latest_val = values[-1]

                    # 计算趋势
                    if len(values) > 10:
                        recent = values[-10:]
                        earlier = values[:10]
                        trend = "上升" if np.mean(recent) > np.mean(earlier) else "下降"
                        trend_mag = abs(np.mean(recent) - np.mean(earlier))
                    else:
                        trend = "数据不足"
                        trend_mag = 0

                    print(f"\n  {metric}:")
                    print(f"    最新值: {latest_val:.4f}")
                    print(f"    平均值: {mean_val:.4f} (±{std_val:.4f})")
                    print(f"    范围: [{min_val:.4f}, {max_val:.4f}]")
                    if trend != "数据不足":
                        print(f"    趋势: {trend} (变化量: {trend_mag:.4f})")
            else:
                print(f"\n  ⚠️  {metric}: 未找到数据")


def analyze_termination_causes(data):
    """分析终止原因"""
    print("\n" + "=" * 80)
    print("终止原因分析")
    print("=" * 80)

    termination_metrics = {
        'Episode_Termination/time_out': '时间限制',
        'Episode_Termination/success_stable': '站立成功',
    }

    for metric, name in termination_metrics.items():
        if metric in data:
            values = data[metric]['values']
            latest = values[-1]
            mean_val = np.mean(values)
            print(f"\n{name}:")
            print(f"  最新值: {latest:.4f}")
            print(f"  平均值: {mean_val:.4f}")
        else:
            print(f"\n{name}: 未找到数据")


def analyze_curriculum_progression(data):
    """分析课程进度"""
    print("\n" + "=" * 80)
    print("课程进度分析")
    print("=" * 80)

    if 'Curriculum/posture_curriculum' in data:
        levels = data['Curriculum/posture_curriculum']['values']
        steps = data['Curriculum/posture_curriculum']['steps']

        print(f"\n当前课程级别: {levels[-1]}")
        print(f"课程级别变化历史:")

        unique_levels = []
        for i, (step, level) in enumerate(zip(steps, levels)):
            if i == 0 or level != levels[i - 1]:
                unique_levels.append((step, level))
                print(f"  Step {step}: Level {level}")

        print(f"\n总共经历了 {len(unique_levels)} 个课程级别")

    # 检查相关指标
    if 'Episode_Reward/orientation_improvement' in data:
        orientation = data['Episode_Reward/orientation_improvement']['values']
        latest_orientation = orientation[-1]
        print(f"\n当前姿态改进奖励: {latest_orientation:.4f}")

    if 'Episode_Reward/height_improvement' in data:
        height = data['Episode_Reward/height_improvement']['values']
        latest_height = height[-1]
        print(f"当前高度改进奖励: {latest_height:.4f}")


def plot_key_trends(data):
    """绘制关键趋势图"""
    print("\n正在生成趋势图...")

    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('GO2W-ARM 训练关键指标趋势', fontsize=16, fontweight='bold')

    # 1. 总奖励和 episode 长度
    if 'Train/mean_reward' in data and 'Train/mean_episode_length' in data:
        ax1 = axes[0, 0]
        ax1.plot(data['Train/mean_reward']['steps'], data['Train/mean_reward']['values'],
                label='平均奖励', color='blue', alpha=0.7)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('奖励值')
        ax1.set_title('总奖励趋势')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax1_twin = ax1.twinx()
        ax1_twin.plot(data['Train/mean_episode_length']['steps'],
                     data['Train/mean_episode_length']['values'],
                     label='Episode长度', color='orange', alpha=0.7)
        ax1_twin.set_ylabel('Episode长度')
        ax1_twin.legend(loc='upper right')

    # 2. 成功率
    if 'Episode_Termination/success_stable' in data:
        ax2 = axes[0, 1]
        success_rate = np.array(data['Episode_Termination/success_stable']['values'])
        ax2.plot(data['Episode_Termination/success_stable']['steps'], success_rate,
                label='成功终止率', color='green', linewidth=2)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('成功率')
        ax2.set_title('成功终止趋势')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='目标线')
        ax2.legend()

    # 3. 课程级别
    if 'Curriculum/posture_curriculum' in data:
        ax3 = axes[1, 0]
        ax3.plot(data['Curriculum/posture_curriculum']['steps'],
                data['Curriculum/posture_curriculum']['values'],
                marker='o', markersize=3, color='purple')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('课程级别')
        ax3.set_title('课程进度')
        ax3.grid(True, alpha=0.3)

    # 4. 姿态和高度改进
    ax4 = axes[1, 1]
    if 'Episode_Reward/orientation_improvement' in data:
        ax4.plot(data['Episode_Reward/orientation_improvement']['steps'],
                data['Episode_Reward/orientation_improvement']['values'],
                label='姿态改进', color='red', alpha=0.7)

    if 'Episode_Reward/height_improvement' in data:
        ax4.plot(data['Episode_Reward/height_improvement']['steps'],
                data['Episode_Reward/height_improvement']['values'],
                label='高度改进', color='blue', alpha=0.7)

    ax4.set_xlabel('Step')
    ax4.set_ylabel('奖励值')
    ax4.set_title('姿态和高度改进')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. 动量阻尼
    ax5 = axes[2, 0]
    if 'Episode_Reward/angular_momentum_damping' in data:
        ax5.plot(data['Episode_Reward/angular_momentum_damping']['steps'],
                data['Episode_Reward/angular_momentum_damping']['values'],
                label='角动量阻尼', color='orange', alpha=0.7)

    if 'Episode_Reward/wheel_angular_momentum' in data:
        ax5.plot(data['Episode_Reward/wheel_angular_momentum']['steps'],
                data['Episode_Reward/wheel_angular_momentum']['values'],
                label='轮子角动量', color='cyan', alpha=0.7)

    ax5.set_xlabel('Step')
    ax5.set_ylabel('奖励值')
    ax5.set_title('动量阻尼趋势')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. 站立成功奖励
    if 'Episode_Reward/success_stable_reward' in data:
        ax6 = axes[2, 1]
        ax6.plot(data['Episode_Reward/success_stable_reward']['steps'],
                data['Episode_Reward/success_stable_reward']['values'],
                label='站立成功奖励', color='green', linewidth=2)
        ax6.set_xlabel('Step')
        ax6.set_ylabel('奖励值')
        ax6.set_title('站立成功奖励')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/tmp/go2warm_training_analysis.png', dpi=150, bbox_inches='tight')
    print(f"图表已保存到: /tmp/go2warm_training_analysis.png")


def identify_key_issues(data):
    """识别关键问题"""
    print("\n" + "=" * 80)
    print("关键问题诊断")
    print("=" * 80)

    issues = []

    # 1. 检查成功率
    if 'Episode_Termination/success_stable' in data:
        success_rate = data['Episode_Termination/success_stable']['values'][-1]
        if success_rate < 0.3:
            issues.append({
                '问题': '成功率过低',
                '当前值': success_rate,
                '建议': '可能是奖励函数配置或环境设置问题',
                '严重程度': '高'
            })

    # 2. 检查课程级别
    if 'Curriculum/posture_curriculum' in data:
        current_level = data['Curriculum/posture_curriculum']['values'][-1]
        if current_level < 5:
            issues.append({
                '问题': '课程级别偏低',
                '当前值': current_level,
                '建议': '机器人可能被困在早期课程级别，无法推进到站立阶段',
                '严重程度': '中'
            })

    # 3. 检查姿态改进奖励
    if 'Episode_Reward/orientation_improvement' in data:
        orientation_reward = data['Episode_Reward/orientation_improvement']['values'][-1]
        if orientation_reward > -1.0:  # 负值表示惩罚
            issues.append({
                '问题': '姿态改进奖励不足',
                '当前值': orientation_reward,
                '建议': '姿态恢复机制可能未能有效工作',
                '严重程度': '高'
            })

    # 4. 检查角动量阻尼
    if 'Episode_Reward/angular_momentum_damping' in data:
        ang_momentum = data['Episode_Reward/angular_momentum_damping']['values'][-1]
        if ang_momentum < -0.1:  # 负值表示惩罚
            issues.append({
                '问题': '角动量阻尼惩罚较高',
                '当前值': ang_momentum,
                '建议': '动态刹车机制可能过于激进或不足',
                '严重程度': '中'
            })

    # 5. 检查站立成功奖励
    if 'Episode_Reward/success_stable_reward' in data:
        standing_reward = data['Episode_Reward/success_stable_reward']['values'][-1]
        if standing_reward < 0.1:
            issues.append({
                '问题': '站立成功奖励过低',
                '当前值': standing_reward,
                '建议': '机器人可能很少达成站立成功条件',
                '严重程度': '高'
            })

    # 6. 检查episode长度
    if 'Train/mean_episode_length' in data:
        episode_length = data['Train/mean_episode_length']['values'][-1]
        if episode_length < 50:
            issues.append({
                '问题': 'Episode长度过短',
                '当前值': episode_length,
                '建议': '机器人可能在早期就被终止，没有足够时间学习',
                '严重程度': '中'
            })

    # 显示问题
    if issues:
        print(f"\n发现 {len(issues)} 个关键问题:\n")
        for i, issue in enumerate(issues, 1):
            print(f"{i}. 【{issue['问题']}】 (严重程度: {issue['严重程度']})")
            print(f"   当前值: {issue['当前值']}")
            print(f"   建议: {issue['建议']}\n")
    else:
        print("\n未检测到明显问题")

    return issues


def main():
    log_dir = "/home/jay/unitree_rl_lab/logs/rsl_rl/unitree_go2warm_twostage_recovery_v0"

    print("=" * 80)
    print("GO2W-ARM 训练 Tensorboard 分析")
    print("=" * 80)
    print(f"日志目录: {log_dir}\n")

    # 读取数据
    data = read_tensorboard_logs(log_dir)
    if data is None:
        return

    # 分析关键机制
    analyze_key_mechanisms(data)

    # 分析终止原因
    analyze_termination_causes(data)

    # 分析课程进度
    analyze_curriculum_progression(data)

    # 绘制趋势图
    plot_key_trends(data)

    # 识别关键问题
    issues = identify_key_issues(data)

    # 总结分析
    print("\n" + "=" * 80)
    print("总结分析")
    print("=" * 80)

    if issues:
        high_severity = [i for i in issues if i['严重程度'] == '高']
        if high_severity:
            print("\n⚠️  高优先级问题:")
            for issue in high_severity:
                print(f"  - {issue['问题']}: {issue['建议']}")

        print("\n💡 基于分析的可能原因:")
        print("  1. 机器人可能被困在侧卧姿态，课程学习未能推进到站立阶段")
        print("  2. 姿态恢复机制可能过于严格或参数不当，导致机器人无法逐步恢复")
        print("  3. 动量阻尼和动态刹车可能不够有效，无法为站立提供稳定基础")
        print("  4. 站立成功判定条件可能过于严格，机器人未能触发成功奖励")
        print("  5. Episode可能在早期被终止，缺乏足够的学习时间")

        print("\n📋 建议检查:")
        print("  1. 检查课程配置，确保各级别难度递进合理")
        print("  2. 调整姿态恢复阈值，给予机器人更多容错空间")
        print("  3. 优化动量阻尼参数，提高稳定性")
        print("  4. 检查站立成功判定条件，确保可达成")
        print("  5. 增加episode时间限制，给予更多学习机会")

    print(f"\n📊 详细分析图表已保存到: /tmp/go2warm_training_analysis.png")
    print("=" * 80)


if __name__ == "__main__":
    main()
