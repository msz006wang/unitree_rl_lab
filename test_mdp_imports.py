#!/usr/bin/env python3
"""Test MDP imports without requiring IsaacSim."""

import sys
import os

# Add source directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "source/unitree_rl_lab"))

# Try to import the mdp module functions
try:
    from unitree_rl_lab.tasks.locomotion.mdp import (
        feet_slide,
        feet_height,
        feet_height_body,
        upward,
        joint_pos_penalty,
        feet_contact,
        feet_contact_without_cmd,
        feet_stumble,
    )
    print("✅ Successfully imported all required MDP functions:")
    print("  - feet_slide")
    print("  - feet_height")
    print("  - feet_height_body")
    print("  - upward")
    print("  - joint_pos_penalty")
    print("  - feet_contact")
    print("  - feet_contact_without_cmd")
    print("  - feet_stumble")
    print("\n✅ All MDP imports successful!")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    sys.exit(1)
