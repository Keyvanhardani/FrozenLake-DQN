import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

log_dir = "./logs_halfcheetah/"
os.makedirs(log_dir, exist_ok=True)

env = gym.make("HalfCheetah-v4", render_mode="human")
env = Monitor(env, filename=os.path.join(log_dir, "monitor.csv"))

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=log_dir,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5
)

eval_env = gym.make("HalfCheetah-v4")
eval_env = Monitor(eval_env, filename=os.path.join(log_dir, "eval_monitor.csv"))

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path='./',
    log_path=log_dir,
    eval_freq=50_000,
    deterministic=True,
    render=False,
    n_eval_episodes=5
)

model.learn(total_timesteps=1_000_000, callback=eval_callback)
model.save("trained_halfcheetah_model")
env.close()
eval_env.close()
