import torch
import torch.nn as nn
import torch.nn.functional as F
from agent.DiscreteAgent import DiscreteAgent, DiscretePolicy, Value
from agent.ContinuousAgent import ContinuousAgent, ContinuousPolicy
from gymnasium import spaces

def create_agent(envs, epochs, beta, eps, eps1, max_lr, max_grad_norm):
    if isinstance(envs.single_action_space , spaces.discrete.Discrete):
        Agent, policy = DiscreteAgent, DiscretePolicy(envs)
    else:
        Agent, policy = ContinuousAgent, ContinuousPolicy(envs)

    value = Value(envs)

    class RPOAgent(Agent):
        def __init__(self, envs, policy, value, epochs, beta, eps, eps1, max_lr, max_grad_norm):
            super().__init__(envs, policy, value, max_lr)
            self.epochs = epochs
            self.beta = beta      # weight of reflective term
            self.eps = eps        # PPO clipping
            self.eps1 = eps1      # reflective clipping
            self.max_grad_norm = max_grad_norm

        def update_policy_value(self, b_actions, b_states, b_logprobs, b_advantages, b_rewards):
            
            # Use self.epochs instead of the hardcoded function argument
            for epoch in range(int(self.epochs)):
                policy_loss = 0
                value_loss = 0

                for actions, states, logprobs, advantages, rewards in zip(
                      b_actions, b_states, b_logprobs, b_advantages, b_rewards):

                    states_cat = torch.cat(states, dim=0)
                    actions_cat = torch.stack(actions)
                    old_logprobs = torch.cat(logprobs, dim=0).detach()

                    # Policy Evaluation
                    _, new_logprobs, new_values = self.get_action_value(states_cat, actions_cat)

                    ratios = torch.exp(new_logprobs - old_logprobs)

                    # 1. PPO term L0
                    clipped_ratio = torch.clamp(ratios, 1 - self.eps, 1 + self.eps)
                    L0 = torch.min(ratios * advantages, clipped_ratio * advantages)

                    # 2. Reflective term L1
                    if len(states) > 1:
                        # shift for (s', a')
                        adv_next = advantages[1:]
                        r_curr = ratios[:-1]
                        r_next = ratios[1:]

                        r_curr_clip = torch.clamp(r_curr, 1 - self.eps, 1 + self.eps)
                        r_next_clip = torch.clamp(r_next, 1 - self.eps1, 1 + self.eps1)

                        prod = r_curr * r_next
                        prod_clip = r_curr_clip * r_next_clip

                        L1 = torch.min(prod * adv_next, prod_clip * adv_next)
                    else:
                        L1 = 0.0

                    policy_loss += -(L0.mean() + self.beta * (L1.mean() if torch.is_tensor(L1) else 0))

                    # Value loss
                    perm = torch.randperm(len(rewards))
                    rewards_shuffled = rewards[perm]
                    new_values_shuffled = new_values[perm]

                    value_loss += F.mse_loss(
                        rewards_shuffled.unsqueeze(1),
                        new_values_shuffled
                    )

                # Policy update
                self.optimizer_policy.zero_grad()
                policy_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer_policy.step()

                # Value update
                self.optimizer_value.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.value.parameters(), 1)
                self.optimizer_value.step()

    return RPOAgent(envs, policy, value, epochs, beta, eps, eps1, max_lr, max_grad_norm), policy, value