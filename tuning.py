import numpy as np
import argparse
from argparse import Namespace
import wandb
import os
from run import run_experiment
from vizier.service import clients
from vizier.service import pyvizier as vz
from vizier._src.service.vizier_client import environment_variables
import sys
from contextlib import contextmanager
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

@contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


TASK_BOUNDS = {
    "Walker2d-v5": (0.0, 4000.0),
    "Hopper-v5": (0.0, 3000.0),
    "Swimmer-v5": (0.0, 500.0),
    "HalfCheetah-v5": (0.0, 5000.0),
    "InvertedPendulum-v5": (0.0, 1000.0),
}

def run_task(alg, task_name, n_eps, params):
    tuning_args = Namespace(
        alg=alg,
        env=task_name,
        num_eps=n_eps,
        num_envs=16,
        wb_login=False,
        
        # Shared
        epochs=params.get("epochs", 10),
        lr=params.get("lr", 3e-4),
        max_grad_norm=params.get("max_grad_norm", 1.0),
        
        # Algorithm Specific
        clip_epsilon=params.get("clip_epsilon", 0.15),
        delta=params.get("delta", 0.03), 
        epsilon=params.get("epsilon", 0.15),
        alpha=params.get("alpha", 0.5),
        beta=params.get("beta", 0.3),
        eps=params.get("eps", 0.2),
        eps1=params.get("eps1", 0.1),
        lambd=params.get("lambd", 1.0),
        
        # TRPO Specific
        cg_iters=params.get("cg_iters", 5),
        cg_damping=params.get("cg_damping", 1e-3),
        max_kl=params.get("max_kl", 0.01),
        backtrack_coeff=params.get("backtrack_coeff", 0.1),
        backtrack_iters=params.get("backtrack_iters", 5),
    )

    with suppress_output():
        rewards = np.array(run_experiment(tuning_args))

    num_episodes = rewards.shape[1]
    last_20 = max(1, int(num_episodes * 0.2))
    final_episodes = rewards[:, -last_20:]
    raw_mean = float(np.mean(final_episodes))

    min_val, max_val = TASK_BOUNDS[task_name]
    norm_score = float(np.clip((raw_mean - min_val) / (max_val - min_val), 0.0, 1.0))

    return task_name, raw_mean, norm_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alg", choices=["adappo", "ppo", "trpo", "espo", "spo", "alphappo", "rpo", "spu", "trpporb"], default="ppo")
    parser.add_argument("--wb-login", action="store_true")
    args = parser.parse_args()

    environment_variables.servicer_kwargs["database_url"] = f"sqlite:///vizier_tuning_new_{args.alg}.db"

    study_config = vz.StudyConfig(algorithm="DEFAULT")
    root = study_config.search_space.root
    
    # SEARCH SPACE
    if args.alg != "adappo":
        root.add_float_param("lr", 1e-5, 1e-2, scale_type=vz.ScaleType.LOG)
        root.add_int_param("epochs", 4, 20)
        
    if args.alg != "trpo" and args.alg != "adappo":
        root.add_float_param("max_grad_norm", 0.1, 2.0)
        
    if args.alg == "ppo":
        root.add_float_param("clip_epsilon", 0.01, 0.6)
        
    elif args.alg == "adappo":
        root.add_float_param("delta", 0.1, 20.0) 
        
    elif args.alg == "rpo":
        root.add_float_param("beta", 0.0, 1.0)
        root.add_float_param("eps", 0.01, 0.6)
        root.add_float_param("eps1", 0.01, 0.6)
        
    elif args.alg == "spo":
        root.add_float_param("epsilon", 0.01, 0.6)
        
    elif args.alg == "espo":
        root.add_float_param("delta", 0.01, 0.5)
        
    elif args.alg == "trpo":
        root.add_int_param("cg_iters", 1, 20)
        root.add_float_param("cg_damping", 1e-5, 1e-1, scale_type=vz.ScaleType.LOG)
        root.add_float_param("max_kl", 0.001, 0.1, scale_type=vz.ScaleType.LOG)
        root.add_float_param("backtrack_coeff", 0.1, 0.9)
        root.add_int_param("backtrack_iters", 1, 20)
        
    elif args.alg == "trpporb":
        root.add_float_param("delta", 0.001, 0.1, scale_type=vz.ScaleType.LOG)
        root.add_float_param("alpha", 0.0, 1.0)
        
    elif args.alg == "alphappo":
        root.add_float_param("alpha", -2.0, 2.5)
        root.add_float_param("beta", 0.0, 1.0)
        
    elif args.alg == "spu":
        root.add_float_param("delta", 0.001, 0.1)
        root.add_float_param("epsilon", 0.001, 0.1) 
        root.add_float_param("lambd", 0.1, 5.0)

    study_config.metric_information.append(
        vz.MetricInformation("mean_reward", goal=vz.ObjectiveMetricGoal.MAXIMIZE)
    )

    study = clients.Study.from_study_config(study_config, owner="tune", study_id=args.alg)

    tasks = ["Walker2d-v5", "Hopper-v5", "Swimmer-v5", "HalfCheetah-v5", "InvertedPendulum-v5"]

    num_eps = [400]*5
    NUM_TRIALS = 40

    start_time = time.time()

    for trial_idx in range(NUM_TRIALS):
        suggestions = study.suggest(count=1)

        for suggestion in suggestions:
            params = suggestion.parameters
            
            param_str = ", ".join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in params.items()])

            print(f"\n--- Starting Trial {trial_idx+1}/{NUM_TRIALS} ({args.alg.upper()}) ---")
            print(f"Suggested Params: {param_str}")

            normalized_means = {}
            with ProcessPoolExecutor(max_workers=len(tasks)) as executor:
                futures = {
                    executor.submit(run_task, args.alg, task_name, n_eps, dict(params)): task_name
                    for task_name, n_eps in zip(tasks, num_eps)
                }

                for future in as_completed(futures):
                    task_name, raw_mean, norm_score = future.result()
                    normalized_means[task_name] = norm_score
                    print(f"  > {task_name}: Raw Mean = {raw_mean:.2f} | Normalized = {norm_score:.4f}")

            final_reward = float(np.mean(list(normalized_means.values())))
            print(f"Trial {trial_idx+1} Final Normalized Score: {final_reward:.4f}")

            suggestion.complete(vz.Measurement({"mean_reward": final_reward}))

    end_time = time.time()
    print(f"\n--- Tuning Complete! --- Time Taken: {end_time - start_time:.2f}s")

    best_params, best_reward = None, None
    for optimal_trial in study.optimal_trials():
        optimal_trial = optimal_trial.materialize()
        best_params = optimal_trial.parameters
        best_reward = optimal_trial.final_measurement.metrics["mean_reward"].value
        print("Best Trial Parameters:", best_params)
        print("Best Trial Reward:", best_reward)

    if args.wb_login and best_params is not None:
        print("\n--- Logging Best Results to W&B ---")
        #os.environ["REQUESTS_CA_BUNDLE"] = "/etc/ssl/certs/ca-certificates.crt"
        #os.environ["SSL_CERT_FILE"] = "/etc/ssl/certs/ca-certificates.crt"
        wandb.login(relogin=False)

        best_config = {k: v for k, v in best_params.items()}
        best_config["alg"] = args.alg

        wandb.init(
            project="RL-Tuning-Summary",
            name=f"best_{args.alg}_params",
            config=best_config,
            mode="online",
        )
        wandb.log({"best_normalized_mean_reward": best_reward})
        wandb.finish()