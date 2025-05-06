from stable_baselines3 import PPO
from soft_pendulum import SoftPendulumEnv
import gym_softrobot
from gym_softrobot.config import RendererType
import os
from tqdm import tqdm

gym_softrobot.RENDERER_CONFIG = RendererType.MATPLOTLIB

env = SoftPendulumEnv(config_generate_video=True, final_time=50)
# Load the model
save_path = (
    "/Users/jiamiaoguo/Desktop/Code/gym-softrobot/PPO_result/ppo_soft_pendulum_1"
)
model = PPO.load(save_path)

obs, _ = env.reset()

for step in tqdm(range(1000)):
    action, _states = model.predict(obs)
    obs, rewards, terminate, truncated, info = env.step(action)
    env.render()
    print(f"{step=:2}| {rewards=}, {terminate=}")
    input("")
    if truncated or terminate:
        print(f"Episode ended after {step + 1} steps.")
        break
