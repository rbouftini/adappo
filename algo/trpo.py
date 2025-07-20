import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from agent.DiscreteAgent import DiscreteAgent, DiscretePolicy, Value
from agent.ContinuousAgent import ContinuousAgent, ContinuousPolicy
from gymnasium import spaces
from torch.nn.utils import parameters_to_vector, vector_to_parameters

def create_agent(envs):
  if isinstance(envs.single_action_space , spaces.discrete.Discrete):
    Agent, policy = DiscreteAgent, DiscretePolicy(envs)
  else:
    Agent, policy = ContinuousAgent, ContinuousPolicy(envs)

  value = Value(envs)

  class TRPOAgent(Agent):
      def __init__(self, envs, policy, value, cg_iters=10, cg_damping=1e-2, max_kl=0.01, backtrack_coeff=0.5,
                  backtrack_iters=10):
          super().__init__(envs, policy, value)
          self.cg_iters = cg_iters
          self.cg_damping = cg_damping
          self.max_kl = max_kl
          self.backtrack_coeff = backtrack_coeff
          self.backtrack_iters = backtrack_iters
          
      def fisher_prod(self, kl, p):
          grad_kl = torch.autograd.grad(kl, self.policy.parameters(), create_graph=True, retain_graph=True)
          flat_grad_kl = parameters_to_vector(grad_kl)
          grad_kl_p = flat_grad_kl @ p
          hvp = torch.autograd.grad(grad_kl_p, self.policy.parameters(), retain_graph=True)
          flat_hvp = parameters_to_vector(hvp)
          return flat_hvp + self.cg_damping * p

      def conjugate_gradient(self, grad_loss, kl):
          x = torch.zeros_like(grad_loss)
          r = grad_loss.clone()
          p = grad_loss.clone()
          rs_old = r @ r
          for _ in range(self.cg_iters):
              hp = self.fisher_prod(kl, p)
              alpha = rs_old / (p @ hp)
              x += alpha * p
              r -= alpha * hp
              rs_new = r @ r
              if rs_new < 1e-10:
                  break
              p = r + (rs_new/rs_old) * p
              rs_old = rs_new
          return x

      def compute_total_kl(self, b_actions, b_states, b_logprobs):
          # sum-then-average for KL
          kl_sum = 0.0
          n = 0
          for actions, states, logprobs in zip(b_actions, b_states, b_logprobs):
              _, new_logprobs, _ = self.get_action_value(torch.cat(states, dim=0), torch.stack(actions))
              old_lp = torch.cat(logprobs, dim=0).detach()
              kl_sum += (old_lp - new_logprobs).sum()
              n += old_lp.numel()
          return kl_sum / n

      def compute_policy_loss(self, b_actions, b_states, b_logprobs, b_advantages):
          all_adv = torch.cat(b_advantages, dim=0)
          adv_norm = (all_adv - all_adv.mean()) / (all_adv.std() + 1e-8)
          idx = 0
          loss = 0.0
          batch_count = len(b_actions)
          for actions, states, logprobs in zip(b_actions, b_states, b_logprobs):
              sz = len(logprobs)
              adv = adv_norm[idx:idx+sz]
              idx += sz
              _, new_logprobs, _ = self.get_action_value(torch.cat(states, dim=0), torch.stack(actions))
              old_lp = torch.cat(logprobs, dim=0).detach()
              ratio = torch.exp(new_logprobs - old_lp)
              loss += (ratio * adv).sum() / batch_count
          return loss

      def update_policy_value(self, b_actions, b_states, b_logprobs, b_advantages, b_rewards, epochs=15):
          max_kl = self.max_kl
          # compute policy loss and KL on same samples
          policy_loss = self.compute_policy_loss(b_actions, b_states, b_logprobs, b_advantages)
          # compute KL
          total_kl = self.compute_total_kl(b_actions, b_states, b_logprobs)

          # policy gradient
          grad_loss = torch.autograd.grad(policy_loss, self.policy.parameters(), retain_graph=True)
          grad_loss = parameters_to_vector(grad_loss)

          # natural gradient step
          natural_grad = self.conjugate_gradient(grad_loss, total_kl)
          step_size = torch.sqrt((2 * max_kl) / (grad_loss @ natural_grad))
          if step_size.isnan():
              step_size = 1e-10
          full_step = step_size * natural_grad

          # save old params for freeze
          old_params = parameters_to_vector(self.policy.parameters()).clone()

          # backtracking line search
          best_params = old_params
          for i in range(self.backtrack_iters):
              frac = self.backtrack_coeff ** i
              new_params = old_params + frac * full_step
              vector_to_parameters(new_params, self.policy.parameters())

              kl_new = self.compute_total_kl(b_actions, b_states, b_logprobs)
              loss_new = self.compute_policy_loss(b_actions, b_states, b_logprobs, b_advantages)
              # enforce trust-region and improvement
              if kl_new <= max_kl and loss_new >= policy_loss + 1e-4 * frac:
                  best_params = new_params.clone()
                  break

          vector_to_parameters(best_params, self.policy.parameters())

          # update value function
          for _ in range(epochs):
            value_loss = 0.0
            for rewards, states, actions in zip(b_rewards, b_states, b_actions):
                _, _, new_values = self.get_action_value(torch.cat(states, dim=0), torch.stack(actions))
                # shuffle for minibatch stability
                perm = torch.randperm(len(rewards))
                value_loss += F.mse_loss(rewards[perm].unsqueeze(1), new_values[perm]) / len(b_actions)

            self.optimizer_value.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(self.value.parameters(), 1.0)
            self.optimizer_value.step()
            
  return TRPOAgent(envs, policy, value), policy, value