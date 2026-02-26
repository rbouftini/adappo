import torch
import torch.nn as nn
import torch.nn.functional as F
from agent.DiscreteAgent import DiscreteAgent, DiscretePolicy, Value
from agent.ContinuousAgent import ContinuousAgent, ContinuousPolicy
from gymnasium import spaces

def create_agent(envs, epochs, delta, max_lr, max_grad_norm):
    if isinstance(envs.single_action_space, spaces.discrete.Discrete):
        Agent, policy = DiscreteAgent, DiscretePolicy(envs)
    else:
        Agent, policy = ContinuousAgent, ContinuousPolicy(envs)

    value = Value(envs)

    class ESPOAgent(Agent):
        def __init__(self, envs, policy, value, epochs, delta, max_lr, max_grad_norm):
            super().__init__(envs, policy, value, max_lr)
            self.epochs = epochs
            self.delta = float(delta)
            self.max_grad_norm = max_grad_norm

        def update_policy_value(self,
                                b_actions,
                                b_states,
                                b_logprobs,
                                b_advantages,
                                b_rewards):

            for epoch in range(int(self.epochs)):
                policy_loss = 0.0
                value_loss = 0.0
                total_abs_dev = 0.0
                total_count = 0

                for actions, states, logprobs, advantages, rewards in zip(b_actions, b_states, b_logprobs, b_advantages, b_rewards):
                    states_cat = torch.cat(states, dim=0)
                    actions_cat = torch.stack(actions)
                    old_logprobs = torch.cat(logprobs, dim=0).detach()

                    # Policy Evaluation
                    _, new_logprobs, new_values = self.get_action_value(
                        states_cat, actions_cat
                    )

                    ratios = torch.exp(new_logprobs - old_logprobs)

                    # Surrogate loss:
                    loss_pi = -(ratios * advantages)
                    policy_loss += loss_pi.sum() / len(b_actions)

                    # Accumulate |r - 1| for early stopping
                    abs_dev = torch.abs(ratios - 1.0)
                    total_abs_dev += abs_dev.sum().item()
                    total_count += abs_dev.numel()

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

                # Early stopping condition
                if total_count > 0:
                    ratio_dev = total_abs_dev / total_count
                    if ratio_dev > self.delta:
                        break

    return ESPOAgent(envs, policy, value, epochs, delta, max_lr, max_grad_norm), policy, value