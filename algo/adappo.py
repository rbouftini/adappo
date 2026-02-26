import torch
import torch.nn as nn
import torch.nn.functional as F
from agent.DiscreteAgent import DiscreteAgent, DiscretePolicy, Value
from agent.ContinuousAgent import ContinuousAgent, ContinuousPolicy
from gymnasium import spaces

def create_agent(envs, delta, clip_epsilon, max_lr, max_grad_norm):
    if isinstance(envs.single_action_space , spaces.discrete.Discrete):
        Agent, policy = DiscreteAgent, DiscretePolicy(envs)
    else:
        Agent, policy = ContinuousAgent, ContinuousPolicy(envs)

    value = Value(envs)

    class NewAgent(Agent):
        def __init__(self, envs, policy, value, delta, clip_epsilon, max_lr, max_grad_norm):
            super().__init__(envs, policy, value, max_lr)
            self.delta = float(delta)
            self.clip_epsilon = float(clip_epsilon)
            self.max_grad_norm = max_grad_norm

        def update_policy_value(self, b_actions, b_states, b_logprobs, b_advantages, b_rewards, epochs=100):
            for epoch in range(epochs):
                policy_loss = 0
                value_loss = 0
                avg_kl = 0.0
                
                for actions, states, logprobs, advantages, rewards in zip(b_actions, b_states, b_logprobs, b_advantages, b_rewards):
                    # Get new log-probabilities and values for the batch
                    _, new_logprobs, new_values = self.get_action_value(torch.cat(states, dim=0), torch.stack(actions))

                    with torch.no_grad():
                        log_ratio = (new_logprobs - torch.cat(logprobs, dim=0)).sum()
                        kl_term = ((torch.exp(log_ratio) - 1) - log_ratio)
                        avg_kl += kl_term / len(b_actions)

                    # Compute the ratio of new to old probabilities
                    ratio = torch.exp(new_logprobs - torch.cat(logprobs, dim=0).detach())
                    clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)

                    # Compute the policy loss
                    loss = -torch.min(clipped_ratio * advantages, ratio * advantages)
                    policy_loss += loss.sum() / len(b_actions)

                    # Compute the value loss (Mean Squared Error)
                    perm = torch.randperm(len(rewards))
                    rewards, new_values = rewards[perm], new_values[perm]
                    value_loss += F.mse_loss(rewards.unsqueeze(1), new_values) / len(b_actions)

                # KL-divergence early stopping check
                if avg_kl > self.delta:
                    break

                # Backpropagate and update policy network
                self.optimizer_policy.zero_grad()
                policy_loss.backward()
                self.optimizer_policy.step()

                # Backpropagate and update value network
                self.optimizer_value.zero_grad()
                value_loss.backward()
                self.optimizer_value.step()
  
    return NewAgent(envs, policy, value, delta, clip_epsilon, max_lr, max_grad_norm), policy, value