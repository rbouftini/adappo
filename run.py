import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
import matplotlib.pyplot as plt
import argparse
from algo import adappo, ppo, trpo
import warnings
import os
import wandb

parser = argparse.ArgumentParser()
parser.add_argument("--alg", help="Testing Algorithm", 
                    choices=["adappo", "ppo", "trpo"], default="adappo")
parser.add_argument("--env", help="Environment id (eg. LunarLander-v3)",
                    default="LunarLander-v3")
parser.add_argument("--num-eps", help="Number of episodes",
                    default="100", type=int)
parser.add_argument("--num-envs", help="Number of parallel environnements",
                    default="16", type=int)
parser.add_argument("--delta", help="Threshold delta value",
                    default="1.8", type=float)  
parser.add_argument("--wb-login", help="Enable W&B login",
                    action='store_true')                

args = parser.parse_args()
warnings.filterwarnings("ignore", message="CUDA initialization: Found no NVIDIA driver on your system")

if args.wb_login:
  os.environ["WANDB_MODE"] = "online"
  wandb.login(relogin=False)
else:
  os.environ["WANDB_MODE"] = "disabled"

def make_wrapped_env():
  def _init():
      env = gym.make(args.env, render_mode="rgb_array")
      if not isinstance(env.action_space, gym.spaces.discrete.Discrete):
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10),
                                                observation_space= env.observation_space)

      return env
  return _init

rewards = []
print(f"Running {args.alg.upper()} on {args.env} task for {args.num_eps} episodes")
num_envs = args.num_envs
np.random.seed(42)
seeds = np.random.randint(1000, size=5)

for seed in seeds:
  env_fns = [make_wrapped_env() for i in range(num_envs)]
  envs = gym.vector.SyncVectorEnv(env_fns)
  _ = envs.reset(seed=[int(seed) + i for i in range(num_envs)])
  torch.manual_seed(seed)

  wb = wandb.init(
    project="RL-Benchmarks",
    group=args.env,         
    name= args.alg,
    config={
       "alg": args.alg,
       "env": args.env,
       "num_eps": args.num_eps,
       "num_envs": args.num_envs,
       "delta": args.delta,
       "seed": int(seed),
    },
   )
  
  wandb.define_metric("episode")
  wandb.define_metric("reward", step_metric="episode")

  if args.alg == "adappo":
    agent, policy, value = adappo.create_agent(envs, args.delta)
  elif args.alg =="ppo":
    agent, policy, value = ppo.create_agent(envs)
  else:
    agent, policy, value = trpo.create_agent(envs)

  rewards = agent.train(episodes=args.num_eps, num_envs=num_envs)
  for ep_idx, ep_reward in enumerate(rewards, start=1):
      wandb.log({"episode": ep_idx, "reward": float(ep_reward)})

  wandb.log({
      "mean_reward": float(np.mean(rewards)),
      "std_reward": float(np.std(rewards)),
      "min_reward": float(np.min(rewards)),
      "max_reward": float(np.max(rewards)),
  })

  wb.finish()