#!/usr/bin/env python3
"""
Demonstration of the Action Space Mismatch Problem

This script shows exactly how the current action space definition
breaks compatibility with RL libraries and causes silent failures.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


def demonstrate_problem():
    """Show the action space mismatch in action."""

    print("=" * 80)
    print("DEMONSTRATING ACTION SPACE MISMATCH PROBLEM")
    print("=" * 80)
    print()

    # =========================================================================
    # STEP 1: Define current action space (as in geoguessr_env.py)
    # =========================================================================
    print("STEP 1: Current Action Space Definition")
    print("-" * 80)

    action_space = spaces.Dict({
        'op': spaces.Discrete(2),
        'click': spaces.Box(
            low=np.array([0, 0], dtype=np.int32),
            high=np.array([1023, 511], dtype=np.int32),
            shape=(2,),
            dtype=np.int32,
        ),
        'answer': spaces.Box(
            low=np.array([-90.0, -180.0], dtype=np.float32),
            high=np.array([90.0, 180.0], dtype=np.float32),
            shape=(2,),
            dtype=np.float32,
        ),
    })

    print(f"Action Space: {action_space}")
    print()

    # =========================================================================
    # STEP 2: Show what RL algorithms generate
    # =========================================================================
    print("STEP 2: What RL Algorithms Generate")
    print("-" * 80)

    print("\nRL algorithms call action_space.sample() during exploration:")
    print()

    for i in range(3):
        sample = action_space.sample()
        print(f"Sample {i+1}: {sample}")

    print()
    print("Notice: ALL samples have op, click, AND answer keys!")
    print()

    # =========================================================================
    # STEP 3: Show what environment expects
    # =========================================================================
    print("STEP 3: What the Environment Expects")
    print("-" * 80)

    print("\nThe action_parser expects format:")
    print("  Click:  {'op': 'click', 'value': [x, y]}")
    print("  Answer: {'op': 'answer', 'value': [lat, lon]}")
    print()

    # =========================================================================
    # STEP 4: Show the parsing failure
    # =========================================================================
    print("STEP 4: Parsing Failure")
    print("-" * 80)

    def mock_parse_action(action):
        """Mock action parser matching actual implementation."""
        # This is what action_parser.py does
        if "value" not in action:
            raise KeyError("Action missing 'value' field")

        op = action["op"]
        if not isinstance(op, str):
            raise TypeError(f"'op' must be string, got {type(op).__name__}")

        return op, action["value"]

    print("\nAttempting to parse sampled actions:")
    print()

    for i in range(3):
        sample = action_space.sample()
        try:
            result = mock_parse_action(sample)
            print(f"✓ Sample {i+1}: Successfully parsed as {result}")
        except Exception as e:
            print(f"✗ Sample {i+1}: FAILED - {type(e).__name__}: {e}")

    print()
    print("Result: 100% failure rate on parsing!")
    print()

    # =========================================================================
    # STEP 5: Silent failure scenario
    # =========================================================================
    print("STEP 5: Silent Failure with Fallback Parser")
    print("-" * 80)

    def mock_parse_with_fallback(action, image_width=1024, image_height=512):
        """Mock fallback parser that returns center click."""
        try:
            return mock_parse_action(action)
        except Exception:
            # Fallback to center click (what actual code does)
            center_x = image_width // 2
            center_y = image_height // 2
            return "click", [center_x, center_y]

    print("\nWith fallback parser (current implementation):")
    print()

    for i in range(3):
        sample = action_space.sample()
        op, values = mock_parse_with_fallback(sample)
        print(f"Sample {i+1}:")
        print(f"  Input:    {sample}")
        print(f"  Fallback: op='{op}', values={values}")
        print()

    print("Result: All actions become center clicks!")
    print("        The agent's policy has NO EFFECT on actions.")
    print("        Training appears to work but learns nothing useful.")
    print()

    # =========================================================================
    # STEP 6: Show correct solution
    # =========================================================================
    print("STEP 6: Correct Solution - Unified Box Space")
    print("-" * 80)

    # SOLUTION: Use a unified continuous action space
    correct_action_space = spaces.Box(
        low=np.array([0, 0, 0], dtype=np.float32),
        high=np.array([1, 1, 1], dtype=np.float32),
        shape=(3,),
        dtype=np.float32,
    )

    print(f"\nCorrected Action Space: {correct_action_space}")
    print()
    print("Interpretation:")
    print("  action[0] < 0.5 → click,  else → answer")
    print("  action[1] → first param (x or lat normalized)")
    print("  action[2] → second param (y or lon normalized)")
    print()

    def parse_unified_action(action, image_width=1024, image_height=512):
        """Parse unified continuous action."""
        action_type_val = action[0]
        param1 = action[1]
        param2 = action[2]

        if action_type_val < 0.5:
            # Click action
            x = int(np.clip(param1 * image_width, 0, image_width - 1))
            y = int(np.clip(param2 * image_height, 0, image_height - 1))
            return {"op": "click", "value": [x, y]}
        else:
            # Answer action
            lat = np.clip(param1 * 180 - 90, -90, 90)
            lon = np.clip(param2 * 360 - 180, -180, 180)
            return {"op": "answer", "value": [float(lat), float(lon)]}

    print("\nSampling and parsing with corrected space:")
    print()

    for i in range(5):
        sample = correct_action_space.sample()
        parsed = parse_unified_action(sample)
        print(f"Sample {i+1}:")
        print(f"  Raw:    {sample}")
        print(f"  Parsed: {parsed}")
        print()

    print("Result: All actions parse successfully!")
    print("        Agent's policy directly controls actions.")
    print("        Training can proceed normally.")
    print()

    # =========================================================================
    # STEP 7: Compatibility comparison
    # =========================================================================
    print("STEP 7: Compatibility Comparison")
    print("-" * 80)

    comparison = """
╔════════════════════════════╦═══════════════╦═══════════════╗
║ Feature                    ║ Current (❌)  ║ Corrected (✓)║
╠════════════════════════════╬═══════════════╬═══════════════╣
║ Works with SB3             ║ NO            ║ YES           ║
║ Works with RLlib           ║ NO            ║ YES           ║
║ Works with CleanRL         ║ NO            ║ YES           ║
║ Works with TorchRL         ║ NO            ║ YES           ║
║ action_space.sample()      ║ INVALID       ║ VALID         ║
║ Policy learns effectively  ║ NO            ║ YES           ║
║ Silent failures            ║ YES (bad!)    ║ NO            ║
║ Simple policy architecture ║ NO (3 heads)  ║ YES (1 head)  ║
║ Continuous control (PPO)   ║ BROKEN        ║ OPTIMAL       ║
╚════════════════════════════╩═══════════════╩═══════════════╝
"""
    print(comparison)

    # =========================================================================
    # STEP 8: Migration path
    # =========================================================================
    print("\nSTEP 8: Migration Path")
    print("-" * 80)

    print("""
To fix this issue:

1. Update action space in geoguessr_env.py:

   self.action_space = spaces.Box(
       low=np.array([0, 0, 0], dtype=np.float32),
       high=np.array([1, 1, 1], dtype=np.float32),
       shape=(3,),
       dtype=np.float32,
   )

2. Update step() to parse unified actions:

   def step(self, action):
       action_type = action[0]
       if action_type < 0.5:
           # Click
           x = int(action[1] * self.image_width)
           y = int(action[2] * self.image_height)
           self._handle_click(x, y)
       else:
           # Answer
           lat = action[1] * 180 - 90
           lon = action[2] * 360 - 180
           reward = self._compute_reward(lat, lon)
           terminated = True

3. Optional: Create wrapper for backward compatibility:

   class LegacyActionWrapper(gym.ActionWrapper):
       '''Converts old dict format to new unified format.'''
       def action(self, action):
           if isinstance(action, dict):
               # Convert legacy format
               ...
           return action

4. Update tests to use new action format

5. Update documentation and examples
""")

    print("=" * 80)
    print("END OF DEMONSTRATION")
    print("=" * 80)


if __name__ == "__main__":
    demonstrate_problem()
