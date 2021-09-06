import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from mpi_modules.mpi_utils import sync_networks, sync_grads
from utils import soft_update, hard_update
from rl_modules.network import GaussianPolicy, QNetwork, DeterministicPolicy

class GSAC(object):
    def __init__(self, num_inputs, action_space, args):
        self.gamma_1 = args.gamma_one
        self.gamma_2 = args.gamma_two
        self.update_frequency = args.update_frequency
        self.tau = args.tau
        self.alpha = args.alpha

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = torch.device("cuda" if args.cuda else "cpu")

        # Defining first critic for gamma 1
        self.critic_1 = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)
        sync_networks(self.critic_1)
        self.critic_1_optim = Adam(self.critic_1.parameters(), lr= args.lr)

        self.critic_1_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)

        # Defining second critic for gamma 2
        self.critic_2 = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)
        sync_networks(self.critic_2)
        self.critic_2_optim = Adam(self.critic_2.parameters(), lr=args.lr)

        self.critic_2_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)

        hard_update(self.critic_1_target, self.critic_1)
        hard_update(self.critic_2_target, self.critic_2)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        sync_networks(self.policy)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return None, action.detach().cpu().numpy()[0], None

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        # Computations for first critic
        for _ in range(self.update_frequency):
            with torch.no_grad():
                next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
                qf1_next_target_1, qf2_next_target_1 = self.critic_1_target(next_state_batch, next_state_action)
                min_qf_next_target_1 = torch.min(qf1_next_target_1, qf2_next_target_1) - self.alpha * next_state_log_pi
                next_q_1_value = reward_batch + mask_batch * (self.gamma_1 * min_qf_next_target_1)
            qf1_1, qf2_1 = self.critic_1(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
            qf1_1_loss = F.mse_loss(qf1_1, next_q_1_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
            qf2_1_loss = F.mse_loss(qf2_1, next_q_1_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
            qf_1_loss = qf1_1_loss + qf2_1_loss
            self.critic_1_optim.zero_grad()
            qf_1_loss.backward()
            sync_grads(self.critic_1)
            self.critic_1_optim.step()

            soft_update(self.critic_1_target, self.critic_1, self.tau)

        # Computations for second critic
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target_2, qf2_next_target_2 = self.critic_2_target(next_state_batch, next_state_action)
            min_qf_next_target_2 = torch.min(qf1_next_target_2, qf2_next_target_2) - self.alpha * next_state_log_pi

            # computations from updated critic (1)
            qf1_next_target_1, qf2_next_target_1 = self.critic_1(next_state_batch, next_state_action)
            min_qf_next_target_1 = torch.min(qf1_next_target_1, qf2_next_target_1)# - self.alpha * next_state_log_pi

            next_q_2_value = reward_batch + self.gamma_1 * min_qf_next_target_1 + mask_batch * (self.gamma_2 * min_qf_next_target_2)
        qf1_2, qf2_2 = self.critic_2(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_2_loss = F.mse_loss(qf1_2, next_q_2_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_2_loss = F.mse_loss(qf2_2, next_q_2_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_2_loss = qf1_2_loss + qf2_2_loss

        self.critic_2_optim.zero_grad()
        qf_2_loss.backward()
        sync_grads(self.critic_2)
        self.critic_2_optim.step()

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_1_target, self.critic_1, self.tau)

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_2_pi, qf2_2_pi = self.critic_2(state_batch, pi)
        min_qf_2_pi = torch.min(qf1_2_pi, qf2_2_pi)

        policy_loss = ((self.alpha * log_pi) -  min_qf_2_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        sync_grads(self.policy)
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs

        return min_qf_2_pi.mean().item()

    # Save model parameters
    def save_model(self, path):
        actor_path = os.path.join(path, 'actor')
        critic_1_path = os.path.join(path, 'critic_1')
        critic_2_path = os.path.join(path, 'critic_2')
        print('Saving models to {}, {} and {}'.format(actor_path, critic_1_path, critic_2_path))
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic_1.state_dict(), critic_1_path)
        torch.save(self.critic_2.state_dict(), critic_2_path)

    # Load model parameters
    def load_model(self, actor_path, critic_1_path, critic_2_path):
        print('Loading models from {}, {} and {}'.format(actor_path, critic_1_path, critic_2_path))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_1_path is not None:
            self.critic_1.load_state_dict(torch.load(critic_1_path))
            hard_update(self.critic_1_target, self.critic_1)
        if critic_2_path is not None:
            self.critic_2.load_state_dict(torch.load(critic_2_path))
            hard_update(self.critic_2_target, self.critic_2)
