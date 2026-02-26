import torch
import torch.nn as nn
import torch.nn.functional as F
from agent.DiscreteAgent import DiscreteAgent, DiscretePolicy, Value
from agent.ContinuousAgent import ContinuousAgent, ContinuousPolicy
from gymnasium import spaces

def create_agent(envs, epochs, alpha, beta, max_lr, max_grad_norm):
    if isinstance(envs.single_action_space, spaces.discrete.Discrete):
        Agent, policy = DiscreteAgent, DiscretePolicy(envs)
    else:
        Agent, policy = ContinuousAgent, ContinuousPolicy(envs)

    value = Value(envs)

    class AlphaPPOAgent(Agent):
        def __init__(self, envs, policy, value, epochs, alpha, beta, max_lr, max_grad_norm):
            super().__init__(envs, policy, value, max_lr)
            self.epochs = epochs
            self.alpha = float(alpha)
            self.beta = float(beta)
            self.max_grad_norm = max_grad_norm

        def update_policy_value(self, b_actions, b_states, b_logprobs, b_advantages, b_rewards):
            for epoch in range(int(self.epochs)):
                policy_loss = 0.0
                value_loss = 0.0

                for actions, states, logprobs, advantages, rewards in zip(
                        b_actions, b_states, b_logprobs, b_advantages, b_rewards):

                    states_cat = torch.cat(states, dim=0)
                    actions_cat = torch.stack(actions)
                    old_logprobs = torch.cat(logprobs, dim=0).detach()

                    # Policy Evaluation
                    _, new_logprobs, new_values = self.get_action_value(states_cat, actions_cat)
                    log_ratio = new_logprobs - old_logprobs
                    ratio = torch.exp(log_ratio)

                    # Alpha Divergence Calculation (Numerically Stable)
                    if abs(self.alpha - 1.0) < 1e-4:
                        div = -log_ratio
                    elif abs(self.alpha - 0.0) < 1e-4:
                        div = ratio * log_ratio
                    else:
                        safe_ratio = torch.clamp(ratio, min=1e-8, max=1e8)
                        div = (safe_ratio**(1.0 - self.alpha) - 1.0) / (self.alpha * (self.alpha - 1.0))

                    # Linearly Combined Objective
                    surrogate = -(1.0 - self.beta) * ratio * advantages + self.beta * div
                    policy_loss += surrogate.sum() / len(b_actions)

                    # Value loss
                    perm = torch.randperm(len(rewards))
                    rewards_shuffled = rewards[perm]
                    new_values_shuffled = new_values[perm]

                    value_loss += F.mse_loss(rewards_shuffled.unsqueeze(1), new_values_shuffled) / len(b_actions)

                # Policy update
                self.optimizer_policy.zero_grad()
                policy_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer_policy.step()

                # Value update
                self.optimizer_value.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.value.parameters(), 1.0)
                self.optimizer_value.step()

    return AlphaPPOAgent(envs, policy, value, epochs, alpha, beta, max_lr, max_grad_norm), policy, value