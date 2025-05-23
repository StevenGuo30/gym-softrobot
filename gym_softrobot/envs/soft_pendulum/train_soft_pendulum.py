from stable_baselines3 import PPO
from soft_pendulum import SoftPendulumEnv
import gym_softrobot
from gym_softrobot.config import RendererType
from stable_baselines3.common.callbacks import CheckpointCallback
from tqdm import tqdm
import re
import os

env = SoftPendulumEnv(config_generate_video=True)
gym_softrobot.RENDERER_CONFIG = RendererType.MATPLOTLIB
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./tensorboard_log/",
    ent_coef=0.01,
    vf_coef=1e-3,
    gamma=0.90,
)

checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path="./PPO_results/",
    name_prefix="PPO_step_result",
    save_replay_buffer=True,
    save_vecnormalize=True,
)

model.learn(total_timesteps=10_000_000, callback=checkpoint_callback, progress_bar=True)

obs, _ = env.reset()

for step in tqdm(range(1000)):
    action, _states = model.predict(obs)
    obs, rewards, terminate, truncated, info = env.step(action)
    env.render()
    if truncated or terminate:
        print(f"Episode ended after {step+1} steps.")
        break
