import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from torch.distributions.normal import Normal
import math

def layer_init(module, std=np.sqrt(2)):
    torch.nn.init.orthogonal_(module.weight, std)
    torch.nn.init.zeros_(module.bias)
    return module

class ContinuousPolicy(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.l1 = layer_init(nn.Linear(envs.single_observation_space.shape[0], 256, bias=True))  # Input is 8 dimentional (8 states)
        self.l2 = layer_init(nn.Linear(256, 256, bias=True))
        self.l3 = layer_init(nn.Linear(256, envs.single_action_space.shape[0], bias=True), std=0.01)  # Output layer (2 means)
        self.logstds = nn.Parameter(torch.zeros(1,envs.single_action_space.shape[0]))

    def forward(self, x):
        x = torch.tanh(self.l1(x))
        x = torch.tanh(self.l2(x))
        x = self.l3(x)
        return x

class Value(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.l1 = layer_init(nn.Linear(envs.single_observation_space.shape[0], 256, bias=True))
        self.l2 = layer_init(nn.Linear(256, 256, bias=True))
        self.l3 = layer_init(nn.Linear(256, 1, bias=True), std=1.)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x 

class ContinuousAgent(ABC):
    def __init__(self, envs, policy, value, max_lr):
        self.env = envs
        self.policy = policy
        self.value = value
        self.max_lr = max_lr
        self.optimizer_policy = torch.optim.Adam(self.policy.parameters())
        self.optimizer_value = torch.optim.Adam(self.value.parameters(), lr=8e-4)

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

    def get_lr(self, it, warmup_steps, warmdown_steps, max_lr,  min_lr):
      # 1) linear warmup for warmup_iters steps
      if it < warmup_steps:
          return max_lr * (it+1) / warmup_steps
      # 2) Stable learning rate
      if it > warmdown_steps:
          return min_lr
      # 3) Decay learning rate
      else:
        decay_ratio = (it- warmup_steps) / (warmdown_steps-warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)

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
            values = torch.cat(values).detach()
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
        observations, _= self.env.reset()
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

    def train(self, episodes, num_envs, discount=0.99, gae_lambda=0.97, warmup_steps=1, warmdown_steps=0):
        saved_rewards = []
        for episode in range(episodes):
            b_actions, b_states, b_rewards, b_logprobs, b_values = self.collect_trajectories(num_envs)

            total_rewards = 0
            for rewards in b_rewards:
                total_rewards += sum(rewards) / num_envs
            saved_rewards.append(total_rewards)

            # Compute advantages and update networks
            b_advantages, b_rewards = self.compute_gaes(b_rewards, b_values, discount, gae_lambda)
            lr  = self.get_lr(episode, warmup_steps, warmdown_steps, self.max_lr, self.max_lr)
            
            self.optimizer_policy.param_groups[0]['lr'] = lr

            print(f"Episode: {episode+1}, Total Rewards: {total_rewards:.4f}, lr:{lr:.4e}")
            self.update_policy_value(b_actions, b_states, b_logprobs, b_advantages, b_rewards)

        return saved_rewards