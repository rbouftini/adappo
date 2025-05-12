import torch
import torch.nn as nn
import torch.nn.functional as F
from DiscreteAgent import DiscreteAgent, DiscretePolicy, Value
from ContinuousAgent import ContinuousAgent, ContinuousPolicy
from gymnasium import spaces

def create_agent(envs):
  if isinstance(envs.single_action_space , spaces.discrete.Discrete):
    Agent, policy = DiscreteAgent, DiscretePolicy(envs)
  else:
    Agent, policy = ContinuousAgent, ContinuousPolicy(envs)

  value = Value(envs)

  class PPOAgent(Agent):
    def __init__(self, envs, policy, value):
      super().__init__(envs, policy, value)

    def update_policy_value(self, b_actions, b_states, b_logprobs, b_advantages, b_rewards, epochs):
      for epoch in range(epochs):
        policy_loss = 0
        value_loss = 0
        for actions, states, logprobs, advantages, rewards in zip(b_actions, b_states, b_logprobs, b_advantages, b_rewards):
          # Get new log-probabilities and values for the batch
          _, new_logprobs, new_values = self.get_action_value(torch.cat(states, dim=0), torch.stack(actions))

          # Compute the ratio of new to old probabilities
          ratio = torch.exp(new_logprobs - torch.cat(logprobs, dim=0).detach())
          clip_epsilon = 0.2

          # Clip the ratio to control policy updates
          clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)

          # Compute the policy loss (PPO objective)
          loss = -torch.min(clipped_ratio * advantages, ratio * advantages) 
          loss = - ratio * advantages
          policy_loss += loss.sum() / len(b_actions)

          # Compute the value loss (Mean Squared Error)
          perm = torch.randperm(len(rewards))
          rewards, new_values = rewards[perm], new_values[perm]
          value_loss += F.mse_loss(rewards.unsqueeze(1), new_values) / len(b_actions)

        # Backpropagate and update policy network
        self.optimizer_policy.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 1)
        self.optimizer_policy.step()

        # Backpropagate and update value network
        self.optimizer_value.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.value.parameters(), 1)
        self.optimizer_value.step()

  return PPOAgent(envs, policy, value), policy, value