import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from torch.distributions.normal import Normal

def layer_init(module, std=np.sqrt(2)):
    torch.nn.init.orthogonal_(module.weight, std)
    torch.nn.init.zeros_(module.bias)
    return module

class ContinuousPolicy(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.l1 = layer_init(nn.Linear(envs.single_observation_space.shape[0], 64, bias=True))  # Input is 8 dimentional (8 states)
        self.l2 = layer_init(nn.Linear(64, 64, bias=True))
        self.l3 = layer_init(nn.Linear(64, envs.single_action_space.shape[0], bias=True), std=0.01)  # Output layer (2 means)
        self.logstds = nn.Parameter(torch.zeros(1,envs.single_action_space.shape[0]))

    def forward(self, x):
        x = torch.tanh(self.l1(x))
        x = torch.tanh(self.l2(x))
        x = self.l3(x)
        return x

class Value(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.l1 = layer_init(nn.Linear(envs.single_observation_space.shape[0], 64, bias=True))
        self.l2 = layer_init(nn.Linear(64, 64, bias=True))
        self.l3 = layer_init(nn.Linear(64, 1, bias=True), std=1.)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x  # Output the estimated value of the state

class ContinuousAgent(ABC):
    def __init__(self, envs, policy, value):
        self.env = envs
        self.policy = policy
        self.value = value
        self.optimizer_policy = torch.optim.Adam(self.policy.parameters())
        self.optimizer_value = torch.optim.Adam(self.value.parameters(), lr=1e-3)

    def play(self):
        observation, info = self.env.reset()
        episode_over = False
        while not episode_over:
            # Convert observation to tensor and predict action probabilities
            observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
            probs = self.policy(observation)
            means = self.policy(observation)
            stds = torch.exp(self.policy.logstds.expand_as(means))
            probs = Normal(means, stds)

            # Sample an action from the probability distribution
            action = probs.sample().numpy()[0]

            # Perform the action and update the state
            observation, reward, terminated, truncated, info = self.env.step(action)
            episode_over = terminated or truncated

        self.env.close()

    def get_lr(self, it, max_lr, warmup_steps, warmdown_steps, max_steps):
      # 1) linear warmup for warmup_iters steps
      if it < warmup_steps:
          return max_lr * (it+1) / warmup_steps
      # 2) Stable learning rate
      if it < max_steps - warmdown_steps:
          return max_lr
      # 3) Decay learning rate
      else:
        decay_ratio = (max_steps - it) / warmdown_steps
        return max_lr * decay_ratio

    def get_action_value(self, observation, actions=None):
        means = self.policy(observation)
        stds = torch.exp(self.policy.logstds.expand_as(means))
        probs = Normal(means, stds)
        value = self.value(observation)
        if actions is None:
            actions = probs.sample()
        return actions, probs.log_prob(actions).sum(1), value

    # Advantages with Generalized Advantage Estimator
    def compute_gaes(self, b_rewards, b_values, discount,  gae_lambda):
        b_advantages = []
        b_returns = []

        for rewards, values in zip(b_rewards, b_values):
            values = torch.tensor(values)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            advantages = torch.zeros_like(rewards)
            lastgae = 0.0
            for t in reversed(range(len(rewards))):
              if t == len(rewards) - 1:
                nextvalue = 0.0
              else:
                nextvalue = values[t + 1]
              delta = rewards[t] + discount * nextvalue - values[t]
              lastgae = delta + discount * gae_lambda * lastgae
              advantages[t] = lastgae

            returns = advantages + values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)
            b_advantages.append(advantages.detach())
            b_returns.append(returns.detach())

        return b_advantages, b_returns


    def collect_trajectories(self, num_envs):
        observations, info = self.env.reset()
        b_actions = [[] for _ in range(num_envs)]
        b_states = [[] for _ in range(num_envs)]
        b_rewards = [[] for _ in range(num_envs)]
        b_logprobs = [[] for _ in range(num_envs)]
        b_values = [[] for _ in range(num_envs)]
        finished = [False for _ in range(num_envs)]

        while not np.all(finished):
          observations = torch.tensor(observations, dtype=torch.float32)
          actions, logprobs, values = self.get_action_value(observations)
          logprobs = logprobs.unsqueeze(1)

          for i in range(num_envs):
              if not finished[i]:
                b_actions[i].append(actions[i])
                b_states[i].append(observations[i].unsqueeze(0))
                b_logprobs[i].append(logprobs[i])
                b_values[i].append(values[i])

          observations, rewards, terminated, truncated, _ = self.env.step(actions.squeeze(1).numpy())

          for i in range(num_envs):
              if not finished[i]:
                b_rewards[i].append(rewards[i])
                if terminated[i] or truncated[i]:
                  finished[i] = True

        return b_actions, b_states, b_rewards, b_logprobs, b_values

    @abstractmethod
    def update_policy_value(self, b_actions, b_states, b_logprobs, b_advantages, b_rewards, epochs):
        pass

    def train(self, episodes,num_envs, discount=0.99, gae_lambda=0.95, clip_epsilon=0.2, max_lr=3e-4, warmup_steps= 20, warmdown_steps=0):
        saved_rewards = []
        for episode in range(episodes):
            b_actions, b_states, b_rewards, b_logprobs, b_values = self.collect_trajectories(num_envs)

            total_rewards = 0
            for rewards in b_rewards:
                total_rewards += sum(rewards) / num_envs
            saved_rewards.append(total_rewards)

            # Compute advantages and update networks
            b_advantages, b_rewards = self.compute_gaes(b_rewards, b_values, discount, gae_lambda)
            lr  = self.get_lr(episode, max_lr, warmup_steps, warmdown_steps, episodes)

            self.optimizer_policy.param_groups[0]['lr'] = lr

            print(f"Episode: {episode+1}, Total Rewards: {total_rewards}, lr:{lr:.4e}")
            self.update_policy_value(b_actions, b_states, b_logprobs, b_advantages, b_rewards)

        return saved_rewards