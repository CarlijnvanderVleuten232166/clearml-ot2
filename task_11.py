from ot2_env_wrapper import OT2Env
from sim_class import Simulation
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
import wandb
import argparse
from clearml import Task
import os

os.environ["WANDB_API_KEY"] = "9b5bea1c2acd433e22ffba938dc5aaa296e4ade7"

# Initialize OT2 Environment
env = OT2Env()

# Set up wandb project
run = wandb.init(project="ot2-rl", sync_tensorboard=True)

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.0003)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--n_steps", type=int, default=2048)
parser.add_argument("--n_epochs", type=int, default=10)

args = parser.parse_args()

# Set up clearml
task = Task.init(project_name="Mentor Group S/Group 3", task_name="RL_Module_232166")
#copy these lines exactly as they are
#setting the base docker image
task.set_base_docker('deanis/2023y2b-rl:latest')
#task.set_repository("https://github.com/CarlijnvanderVleuten232166/clearml-ot2")
#setting the task to run remotely on the default queue
task.execute_remotely(queue_name="default")

# Set up model
model = PPO('MlpPolicy', env, verbose=1, 
            learning_rate=args.learning_rate, 
            batch_size=args.batch_size, 
            n_steps=args.n_steps, 
            n_epochs=args.n_epochs, 
            tensorboard_log=f"runs/{run.id}",)
# create wandb callback
wandb_callback = WandbCallback(model_save_freq=1000,
                                model_save_path=f"models/{run.id}",
                                verbose=2,
                                )

timesteps = 10000
for i in range(10):
    # add the reset_num_timesteps=False argument to the learn function to prevent the model from resetting the timestep counter
    # add the tb_log_name argument to the learn function to log the tensorboard data to the correct folder
    model.learn(total_timesteps=timesteps, callback=wandb_callback, progress_bar=True, reset_num_timesteps=False,tb_log_name=f"runs/{run.id}")
    # save the model to the models folder with the run id and the current timestep
    model.save(f"models/{run.id}/{timesteps*(i+1)}")

