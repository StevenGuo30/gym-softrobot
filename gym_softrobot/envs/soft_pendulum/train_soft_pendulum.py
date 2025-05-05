from stable_baselines3 import PPO
from soft_pendulum import SoftPendulumEnv
import gym_softrobot
from gym_softrobot.config import RendererType
from tqdm import tqdm

env = SoftPendulumEnv(config_generate_video=True)
gym_softrobot.RENDERER_CONFIG = RendererType.MATPLOTLIB
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard_log/",ent_coef=0.01, vf_coef=0.001)
model.learn(total_timesteps=10000000)

model.save("ppo_soft_pendulum")

obs,_ = env.reset()

for step in tqdm(range(1000)):
    action, _states = model.predict(obs)
    obs, rewards, terminate, truncated, info = env.step(action)
    env.render()
    if truncated or terminate:
        print(f"Episode ended after {step+1} steps.")
        break