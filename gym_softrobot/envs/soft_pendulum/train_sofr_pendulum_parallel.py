from stable_baselines3 import PPO
from soft_pendulum import SoftPendulumEnv
import gym_softrobot
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import os

# enviroment numbers
NUM_ENVS = 20

# Vectorized environment
env = SubprocVecEnv(
    [
        lambda: SoftPendulumEnv(config_generate_video=True, final_time=30)
        for _ in range(NUM_ENVS)
    ],
    start_method="fork",
)

# Create a new model with updated network architecture
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./tensorboard_log/",
    ent_coef=0.01,
    vf_coef=1e-3,
    gamma=0.90,
)
name_prefix = "PPO_step_result_3"
# Create checkpoint callback
checkpoint_callback = CheckpointCallback(
    save_freq=10_000,
    save_path="./PPO_results_parallel/",
    name_prefix=name_prefix,
    save_replay_buffer=True,
    save_vecnormalize=True,
)

# Train the model
model.learn(
    total_timesteps=100_000_000, callback=checkpoint_callback, progress_bar=True
)

# Save the model
model.save(f"PPO_results_parallel/{name_prefix}")
