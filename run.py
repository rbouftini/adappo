import numpy as np
import torch
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import argparse
from algo import adappo, ppo, trpo, espo, spo, alphappo, rpo, spu, trpporb
import warnings
import os
import wandb
import shimmy
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ["MUJOCO_GL"] = "egl"

def make_wrapped_env(env_id):
    def _init():
        env = gym.make(env_id, render_mode="rgb_array")
        env = FlattenObservation(env)
        if not isinstance(env.action_space, gym.spaces.discrete.Discrete):
            env = gym.wrappers.ClipAction(env)
            env = gym.wrappers.NormalizeObservation(env)
            env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10),
                                                    observation_space=env.observation_space)
        return env
    return _init

def run_seed(args, seed, num_envs):
    warnings.filterwarnings("ignore", message="CUDA initialization: Found no NVIDIA driver on your system")

    env_fns = [make_wrapped_env(args.env) for _ in range(num_envs)]
    envs = gym.vector.SyncVectorEnv(env_fns)
    _ = envs.reset(seed=[int(seed) + i for i in range(num_envs)])
    torch.manual_seed(seed)

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

    seed_rewards = agent.train(episodes=args.num_eps, num_envs=num_envs)

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
    warnings.filterwarnings("ignore", message="CUDA initialization: Found no NVIDIA driver on your system")

    if args.wb_login:
        os.environ["WANDB_MODE"] = "online"
        wandb.login(relogin=False)
    else:
        os.environ["WANDB_MODE"] = "disabled"

    print(f"Running {args.alg.upper()} on {args.env} task for {args.num_eps} episodes")
    num_envs = args.num_envs
    np.random.seed(42)
    seeds = np.random.randint(1000, size=10)

    all_rewards = [None] * len(seeds)
    seed_to_idx = {int(s): i for i, s in enumerate(seeds)}

    with ProcessPoolExecutor(max_workers=len(seeds)) as executor:
        futures = {
            executor.submit(run_seed, args, int(seed), num_envs): int(seed)
            for seed in seeds
        }

        for future in as_completed(futures):
            seed, seed_rewards = future.result()
            all_rewards[seed_to_idx[seed]] = seed_rewards

    return all_rewards

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Configs
    parser.add_argument("--alg", choices=["adappo", "ppo", "trpo", "espo", "spo", "alphappo", "rpo", "spu", "trpporb"], default="ppo")
    parser.add_argument("--env", default="LunarLander-v3")
    parser.add_argument("--num-eps", default=100, type=int)
    parser.add_argument("--num-envs", default=16, type=int)
    parser.add_argument("--wb-login", action='store_true')
    
    # Shared Hyperparameters
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--max-grad-norm", default=1.0, type=float)
    
    # Algorithm-Specific Hyperparameters
    parser.add_argument("--clip-epsilon", default=0.2, type=float)
    parser.add_argument("--delta", default=0.03, type=float)       
    parser.add_argument("--epsilon", default=0.2, type=float)      
    parser.add_argument("--alpha", default=0.5, type=float)        
    parser.add_argument("--beta", default=0.3, type=float)         
    parser.add_argument("--eps", default=0.2, type=float)          
    parser.add_argument("--eps1", default=0.1, type=float)         
    parser.add_argument("--lambd", default=1.0, type=float)        
    
    # TRPO Specific
    parser.add_argument("--cg-iters", default=5, type=int)
    parser.add_argument("--cg-damping", default=1e-3, type=float)
    parser.add_argument("--max-kl", default=0.01, type=float)
    parser.add_argument("--backtrack-coeff", default=0.1, type=float)
    parser.add_argument("--backtrack-iters", default=5, type=int)

    args = parser.parse_args()

    final_rewards = run_experiment(args)