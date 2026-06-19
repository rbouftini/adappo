# Decoupling Proximal Policy Optimization Hyperparameters

This repository implements a modification of [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347) that constrains the divergence between the trajectory distributions induced by the old and new policies. A single trajectory-divergence criterion governs the optimization and decouples PPO's main hyperparameters — the learning rate, number of epochs, and clipping range — improving training stability and final performance with only minimal changes to the algorithm.

Alongside the proposed method, the repository provides a suite of policy-gradient baselines under a common interface.

## Features

* **Single-criterion control**: a trajectory-divergence threshold replaces manual tuning of the learning rate, number of epochs, and clipping range.
* **Nine algorithms, one interface**:PPO plus eight policy-gradient baselines, selectable with a single flag.
* **Parallel multi-seed runs**: each algorithm trains across several seeds and environments in parallel and reports the mean reward per episode.
* **Reproducible and containerized**: pinned dependencies and a Docker image for one-command setup.

## Algorithms

Choose an algorithm with `--alg`:

| `--alg`    | Method                                                                 | Key hyperparameters |
|------------|------------------------------------------------------------------------|---------------------|
| `adappo`   | Trajectory-divergence criterion with KL-based early stopping (**this work**) | `--delta` |
| `ppo`      | PPO — clipped surrogate objective                                      | `--clip-epsilon`, `--epochs` |
| `trpo`     | TRPO — natural gradient via conjugate gradient with line search        | `--max-kl`, `--cg-iters`, `--cg-damping`, `--backtrack-coeff`, `--backtrack-iters` |
| `espo`     | Early stopping on the mean probability-ratio deviation                 | `--delta`, `--epochs` |
| `spo`      | Quadratic-penalty surrogate                                           | `--epsilon`, `--epochs` |
| `alphappo` | α-divergence-regularized surrogate                                    | `--alpha`, `--beta`, `--epochs` |
| `rpo`      | Adds a reflective term coupling consecutive steps                     | `--beta`, `--eps`, `--eps1`, `--epochs` |
| `spu`      | KL-masked surrogate update                                            | `--delta`, `--epsilon`, `--lambd`, `--epochs` |
| `trpporb`  | Trust-region surrogate with rollback on violating samples            | `--delta`, `--alpha`, `--epochs` |

The agent is chosen automatically from the environment's action space: a categorical policy for discrete actions, a Gaussian policy for continuous actions.

## Prerequisites

* [Docker](https://www.docker.com/)
* [Git](https://git-scm.com/)

## Getting Started

1. **Clone the repository**

   ```bash
   git clone https://github.com/rbouftini/adappo.git
   cd adappo
   ```

2. **Build the Docker image**

   ```bash
   docker build -t adappo .
   ```

3. **Start a container with a shell** — bind-mounts the repository so local edits are picked up:

   * **macOS/Linux**

     ```bash
     docker run -it --mount type=bind,src="$(pwd)",target=/adappo adappo bash
     ```

   * **PowerShell (Windows)**

     ```powershell
     docker run -it --mount "type=bind,src=$($pwd),target=/adappo" adappo bash
     ```

4. **Run an experiment** — inside the container:

   ```bash
   python run.py --alg adappo --env LunarLander-v3 --num-eps 100 --num-seeds 10
   ```

   Each run trains `--num-seeds` seeds in parallel and prints the mean reward per episode:

   ```
   Running ADAPPO on LunarLander-v3 for 100 episodes across 10 seed(s) in parallel
   Episode 1: mean reward -180.44 ± 25.48 (over 10 seeds)
   ...
   ```

## Usage

```
python run.py [--alg {adappo,ppo,trpo,espo,spo,alphappo,rpo,spu,trpporb}]
              [--env ENV] [--num-eps N] [--num-envs N] [--num-seeds N] [--wb-login]
              [hyperparameters ...]
```

**Run configuration**

| Argument       | Default          | Description |
|----------------|------------------|-------------|
| `--alg`        | `ppo`            | Algorithm to run |
| `--env`        | `LunarLander-v3` | Gymnasium / DM Control environment id |
| `--num-eps`    | `100`            | Episodes per seed |
| `--num-envs`   | `16`             | Parallel environments per seed |
| `--num-seeds`  | `10`             | Seeds to run in parallel |
| `--wb-login`   | off              | Log metrics to Weights & Biases |

**Shared hyperparameters:** `--lr` (`3e-4`), `--epochs` (`10`), `--max-grad-norm` (`1.0`).

**Algorithm-specific hyperparameters:** see the table above (`--delta`, `--clip-epsilon`, `--epsilon`, `--alpha`, `--beta`, `--eps`, `--eps1`, `--lambd`, and the TRPO group). Metrics are sent to Weights & Biases only when `--wb-login` is passed; otherwise W&B is disabled.

## Hyperparameter tuning

`tuning.py` runs a [Google Vizier](https://github.com/google/vizier) study to tune an algorithm across a set of MuJoCo control tasks (Walker2d, Hopper, Swimmer, HalfCheetah, InvertedPendulum). Each trial evaluates a parameter suggestion on every task, normalizes the returns to `[0, 1]`, and maximizes the mean normalized reward.

```bash
python tuning.py --alg adappo        # add --wb-login to log the best configuration
```

The per-algorithm search space is defined in `tuning.py`; the study is stored in a local SQLite database.

## Repository structure

```
adappo/
├── agent/
│   ├── ContinuousAgent.py   # Gaussian-policy agent + shared training loop
│   └── DiscreteAgent.py     # Categorical-policy agent + shared training loop
├── algo/
│   ├── adappo.py            # AdaPPO (this work)
│   ├── alphappo.py          # AlphaPPO
│   ├── espo.py              # ESPO
│   ├── ppo.py               # PPO
│   ├── rpo.py               # RPO
│   ├── spo.py               # SPO
│   ├── spu.py               # SPU
│   ├── trpo.py              # TRPO
│   └── trpporb.py           # TR-PPO-RB
├── run.py                   # Train an algorithm across seeds
├── tuning.py                # Vizier hyperparameter tuning
├── requirements.txt         # Pinned dependencies
├── Dockerfile               # Container build
├── LICENSE
├── README.md
├── .dockerignore
└── .gitignore
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
