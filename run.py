import os
import warnings

def _silence_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", module="glfw")
    warnings.filterwarnings("ignore", message=".*CUDA initialization.*")

_silence_warnings()
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide" 
import pygame

import numpy as np
import torch
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import argparse
import wandb
import shimmy
from algo import adappo, ppo, trpo, espo, spo, alphappo, rpo, spu, trpporb
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
from queue import Empty

def make_wrapped_env(env_id):
    def _init():
        env = gym.make(env_id, render_mode="rgb_array")
        env = FlattenObservation(env)
        if not isinstance(env.action_space, gym.spaces.discrete.Discrete):
            env = gym.wrappers.ClipAction(env)
            env = gym.wrappers.NormalizeObservation(env)
            # Clip normalized observations so outliers don't destabilize the running stats.
            env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10),
                                                    observation_space=env.observation_space)
        return env
    return _init


def run_seed(args, seed, num_envs, reporter=None):
    _silence_warnings()
    # Vectorized envs: the agent collects `num_envs` trajectories per update.
    env_fns = [make_wrapped_env(args.env) for _ in range(num_envs)]
    envs = gym.vector.SyncVectorEnv(env_fns)
    _ = envs.reset(seed=[int(seed) + i for i in range(num_envs)])
    torch.manual_seed(seed)

    # Optionally log rewards to W&B
    if args.wb_login:
        config_dict = vars(args).copy()
        config_dict["seed"] = int(seed)

        wb = wandb.init(
            project="AdaPPO",
            group=args.env,
            name=f"{args.alg}",
            config=config_dict,
        )
        wandb.define_metric("episode")
        wandb.define_metric("reward", step_metric="episode")

    if args.alg == "adappo":
        agent, policy, value = adappo.create_agent(envs, args.delta, args.clip_epsilon, args.lr, args.max_grad_norm)
    elif args.alg == "ppo":
        agent, policy, value = ppo.create_agent(envs, args.epochs, args.clip_epsilon, args.lr, args.max_grad_norm)
    elif args.alg == "rpo":
        agent, policy, value = rpo.create_agent(envs, args.epochs, args.beta, args.eps, args.eps1, args.lr, args.max_grad_norm)
    elif args.alg == "spo":
        agent, policy, value = spo.create_agent(envs, args.epochs, args.epsilon, args.lr, args.max_grad_norm)
    elif args.alg == "trpo":
        agent, policy, value = trpo.create_agent(envs, args.epochs, args.cg_iters, args.cg_damping, args.max_kl, args.backtrack_coeff, args.backtrack_iters, args.lr)
    elif args.alg == "trpporb":
        agent, policy, value = trpporb.create_agent(envs, args.epochs, args.delta, args.alpha, args.lr, args.max_grad_norm)
    elif args.alg == "espo":
        agent, policy, value = espo.create_agent(envs, args.epochs, args.delta, args.lr, args.max_grad_norm)
    elif args.alg == "alphappo":
        agent, policy, value = alphappo.create_agent(envs, args.epochs, args.alpha, args.beta, args.lr, args.max_grad_norm)
    elif args.alg == "spu":
        agent, policy, value = spu.create_agent(envs, args.epochs, args.delta, args.epsilon, args.lambd, args.lr, args.max_grad_norm)

    # Train this seed; stream each episode's reward to the parent for averaging.
    seed_rewards = agent.train(episodes=args.num_eps, num_envs=num_envs, reporter=reporter, seed=int(seed))

    if args.wb_login:
        for ep_idx, ep_reward in enumerate(seed_rewards, start=1):
            wandb.log({"episode": ep_idx, "reward": float(ep_reward)})

        wandb.log({
            "mean_reward": float(np.mean(seed_rewards)),
            "std_reward": float(np.std(seed_rewards)),
            "min_reward": float(np.min(seed_rewards)),
            "max_reward": float(np.max(seed_rewards)),
        })
        wb.finish()

    envs.close()
    return int(seed), seed_rewards

def run_experiment(args):
    if args.wb_login:
        os.environ["WANDB_MODE"] = "online"
        wandb.login(relogin=False)
    else:
        os.environ["WANDB_MODE"] = "disabled"

    num_envs = args.num_envs

    np.random.seed(42)
    seeds = np.random.randint(1000, size=args.num_seeds)
    n_seeds = len(seeds)

    print(f"Running {args.alg.upper()} on {args.env} for {args.num_eps} episodes "
          f"across {n_seeds} seed(s) in parallel")

    all_rewards = [None] * n_seeds
    seed_to_idx = {int(s): i for i, s in enumerate(seeds)}

    pending = {}  # episode -> rewards reported so far for that episode

    # A Manager queue lets the worker processes stream (seed, episode, reward) back.
    with Manager() as manager:
        reporter = manager.Queue()

        # One worker process per seed so all seeds train concurrently.
        with ProcessPoolExecutor(max_workers=n_seeds) as executor:
            futures = {
                executor.submit(run_seed, args, int(seed), num_envs, reporter): int(seed)
                for seed in seeds
            }

            while not all(f.done() for f in futures) or not reporter.empty():
                try:
                    _, episode, reward = reporter.get(timeout=0.5)
                except Empty:
                    continue
                pending.setdefault(episode, []).append(reward)
                if len(pending[episode]) == n_seeds:
                    rewards = pending.pop(episode)
                    print(f"Episode {episode}: mean reward {np.mean(rewards):.2f} "
                          f"± {np.std(rewards):.2f} (over {n_seeds} seeds)")

            # Collect each seed's full reward history into its slot.
            for future in as_completed(futures):
                seed, seed_rewards = future.result()
                all_rewards[seed_to_idx[seed]] = seed_rewards

    return all_rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # --- Experiment configuration ---
    parser.add_argument("--alg", choices=["adappo", "ppo", "trpo", "espo", "spo", "alphappo", "rpo", "spu", "trpporb"], default="ppo")
    parser.add_argument("--env", default="LunarLander-v3")
    parser.add_argument("--num-eps", default=100, type=int)
    parser.add_argument("--num-envs", default=16, type=int, help="Parallel environments per seed")
    parser.add_argument("--num-seeds", default=10, type=int, help="Number of random seeds to run in parallel")
    parser.add_argument("--wb-login", action='store_true', help="Log metrics to Weights & Biases")

    # --- Shared hyperparameters ---
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--max-grad-norm", default=1.0, type=float)

    # --- Algorithm-specific hyperparameters ---
    parser.add_argument("--clip-epsilon", default=0.2, type=float)
    parser.add_argument("--delta", default=0.03, type=float)
    parser.add_argument("--epsilon", default=0.2, type=float)
    parser.add_argument("--alpha", default=0.5, type=float)
    parser.add_argument("--beta", default=0.3, type=float)
    parser.add_argument("--eps", default=0.2, type=float)
    parser.add_argument("--eps1", default=0.1, type=float)
    parser.add_argument("--lambd", default=1.0, type=float)

    # --- TRPO-specific ---
    parser.add_argument("--cg-iters", default=5, type=int)
    parser.add_argument("--cg-damping", default=1e-3, type=float)
    parser.add_argument("--max-kl", default=0.01, type=float)
    parser.add_argument("--backtrack-coeff", default=0.1, type=float)
    parser.add_argument("--backtrack-iters", default=5, type=int)

    args = parser.parse_args()

    final_rewards = run_experiment(args)