"""Train DQN on FrozenLake-v1"""
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

if __name__ == "__main__":
    env_args = {
        "id": "FrozenLake-v1",
        "map_name": "4x4",
        "is_slippery": False,
        "render_mode": "human"
    }
    env = Monitor(gym.make(**env_args))

    dqn_args = {
        "policy": "MlpPolicy",
        "learning_rate": 0.0007,
        "buffer_size": 10_000,
        "learning_starts": 100,
        "target_update_interval": 1_000,
        "gamma": 0.99,
        "train_freq": 4,
        "tau": 1.0,
        "gradient_steps": 1,
        "exploration_fraction": 0.1,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
        "batch_size": 32,
        "verbose": 1,
        "env": env
    }

    model = DQN(**dqn_args)

    callback_args = {
        "eval_freq": 10_000,
        "deterministic": True,
        "n_eval_episodes": 25
    }
    eval_callback = EvalCallback(env, **callback_args)

    train_args = {
        "total_timesteps": 1_000,
        "callback": eval_callback,
        "log_interval": 100
    }

    model.learn(**train_args)
    model.save("../models/trained_model")
