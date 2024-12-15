"""Test environment loading"""

import gymnasium as gym

def test_env_loading():
    # Testet, ob FrozenLake korrekt geladen wird.
    env = gym.make("FrozenLake-v1", is_slippery=False)
    obs, info = env.reset()
    assert obs is not None, "Environment failed to reset."
    # Optional: Weitere Checks, z. B. ob observation_space und action_space wie erwartet sind.
    assert env.observation_space is not None, "No observation space detected."
    assert env.action_space is not None, "No action space detected."
