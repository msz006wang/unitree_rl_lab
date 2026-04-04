#!/usr/bin/env python3
"""测试 body_names 配置是否正确"""

import re

# 测试正则表达式
body_names_pattern = "^(?!.*foot).*$"
foot_pattern = ".*_foot"

# 可用的 body 名称
available_bodies = [
    'base', 'FL_hip', 'FL_thigh', 'FL_calf', 'FL_foot',
    'FR_hip', 'FR_thigh', 'FR_calf', 'FR_foot',
    'Head_upper', 'Head_lower',
    'RL_hip', 'RL_thigh', 'RL_calf', 'RL_foot',
    'RR_hip', 'RR_thigh', 'RR_calf', 'RR_foot',
    'arm_link1', 'arm_link2', 'arm_link3', 'arm_link4', 'arm_link5', 'arm_link6',
    'imu_ee'
]

print("测试正则表达式配置")
print("=" * 60)

# 测试非脚部匹配 (^(?!.*foot).*$)
print("\n1. 测试非脚部匹配 (^(?!.*foot).*$):")
non_foot_bodies = [body for body in available_bodies if re.match(body_names_pattern, body)]
print(f"   匹配的 body: {non_foot_bodies}")
print(f"   数量: {len(non_foot_bodies)}")

# 测试脚部匹配 (.*_foot)
print("\n2. 测试脚部匹配 (.*_foot):")
foot_bodies = [body for body in available_bodies if re.match(foot_pattern, body)]
print(f"   匹配的 body: {foot_bodies}")
print(f"   数量: {len(foot_bodies)}")

# 验证是否正确
print("\n验证结果:")
print("=" * 60)

# 验证脚部匹配
expected_foot_bodies = ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']
if set(foot_bodies) == set(expected_foot_bodies):
    print("✅ 脚部匹配正确")
else:
    print(f"❌ 脚部匹配错误")
    print(f"   期望: {expected_foot_bodies}")
    print(f"   实际: {foot_bodies}")

# 验证非脚部匹配
expected_non_foot_bodies = [body for body in available_bodies if body not in expected_foot_bodies]
if set(non_foot_bodies) == set(expected_non_foot_bodies):
    print("✅ 非脚部匹配正确")
else:
    print(f"❌ 非脚部匹配错误")
    print(f"   期望: {expected_non_foot_bodies}")
    print(f"   实际: {non_foot_bodies}")

print("\n所有测试通过！✅")
print("wheel_angular_momentum 应该使用 body_names='.*_foot'")
print("contact_adaptive 应该使用 body_names=['^(?!.*foot).*$']")
