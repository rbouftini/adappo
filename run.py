import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
import new
import ppo
import warnings

parser = argparse.ArgumentParser()
parser.add_argument("--alg", help="Testing Algorithm", 
                    choices=["new", "ppo"], default="new")
parser.add_argument("--env", help="Environment id (eg. LunarLander-v3)",
                    default="LunarLander-v3")
parser.add_argument("--num-eps", help="Number of episodes",
                    default="100", type=int)
args = parser.parse_args()
warnings.filterwarnings("ignore", message="CUDA initialization: Found no NVIDIA driver on your system")

def make_wrapped_env():
    def _init():
        env = gym.make(args.env, render_mode="rgb_array")
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10),
                                                observation_space= env.observation_space)

        return env
    return _init

rewards = []
print(f"Running {args.alg.upper()} on {args.env} task for {args.num_eps} episodes")
num_envs = 8
np.random.seed(42)
seeds = np.random.randint(1000, size=4)

for seed in seeds:
  env_fns = [make_wrapped_env() for i in range(num_envs)]
  envs = gym.vector.SyncVectorEnv(env_fns)
  _ = envs.reset(seed=[int(seed) + i for i in range(num_envs)])
  torch.manual_seed(seed)
  
  if args.alg == "new":
    agent, policy, value = new.create_agent(envs)
  else:
    agent, policy, value = ppo.create_agent(envs)

  rewards.append(agent.train(episodes=args.num_eps, num_envs= num_envs))