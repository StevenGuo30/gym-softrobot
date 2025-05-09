from stable_baselines3 import PPO
from soft_pendulum import SoftPendulumEnv
import gym_softrobot
from gym_softrobot.config import RendererType
import os
from tqdm import tqdm

# gym_softrobot.RENDERER_CONFIG = RendererType.POVRAY

env = SoftPendulumEnv(config_generate_video=True, final_time=50)
# Load the model
# save_path = "/Users/jiamiaoguo/Desktop/Code/gym-softrobot/PPO_results_parallel/PPO_step_result_600000_steps"
save_path = "/Users/jiamiaoguo/Desktop/Code/gym-softrobot/PPO_results_parallel/PPO_step_result_1_10000000_steps"
# save_path = "/Users/jiamiaoguo/Desktop/Code/gym-softrobot/PPO_results/PPO_step_result_270000_steps"
model = PPO.load(save_path)

obs, _ = env.reset()

for step in tqdm(range(1000)):
    action, states = model.predict(obs)
    print(f"{step=:2}| {action=}")
    obs, rewards, terminate, truncated, info = env.step(action)
    env.render()
    # print(f"{step=:2}| {rewards=}, {terminate=}")
    # input("")
    if truncated or terminate:
        print(f"Episode ended after {step + 1} steps.")
        break
