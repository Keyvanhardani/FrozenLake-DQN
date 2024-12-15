"""Test environment loading"""
import gymnasium as gym

def test_env_loading():
    env = gym.make("FrozenLake-v1", is_slippery=False)
    assert env is not None
