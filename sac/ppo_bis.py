import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from model import DiagGaussian, MLPBase
from mpi_utils import sync_grads


class PPO(object):
    def __init__(self, num_inputs, action_space, clip_param=0.2, ppo_epoch=4, num_mini_batch=32, value_loss_coef=0.5, entropy_coef=0.01, lr=3e-4,
                 eps=1e-5, max_grad_norm=0.5, use_clipped_value_loss=True):

        self.policy = MLPBase(num_inputs)
        self.dist = DiagGaussian(self.policy.output_size, action_space)

        self.device = torch.device("cpu")

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.policy_optim = Adam(self.policy.parameters(), lr=lr, eps=eps)

    def select_action(self, state, evaluate=False):
        # state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        # if evaluate is False:
        #     action, _, _ = self.policy.sample(state)
        # else:
        #     _, _, action = self.policy.sample(state)
        # return action.detach().cpu().numpy()[0]
        value, actor_features = self.policy(state)
        dist = self.dist(actor_features)
        if evaluate:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value.detach(), action.detach(), action_log_probs.detach()

    def get_value(self, inputs):
        state = torch.FloatTensor(inputs).to(self.device).unsqueeze(0)
        value, _ = self.policy(state)
        return value

    def evaluate_actions(self, inputs, action):
        value, actor_features = self.policy(inputs)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy

    def update_parameters(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            data_generator = rollouts.feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, actions_batch, value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ = sample

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy = self.evaluate_actions(obs_batch, actions_batch)

                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                self.policy_optim.zero_grad()
                (value_loss * self.value_loss_coef + action_loss -
                 dist_entropy * self.entropy_coef).backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(),
                                         self.max_grad_norm)
                sync_grads(self.policy)
                self.policy_optim.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch