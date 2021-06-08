import glob
import os
import numpy as np
import torch
import torch.nn as nn

from a2c_ppo_acktr.envs import VecNormalize


# Get a render function
def get_render_func(venv):
    if hasattr(venv, 'envs'):
        return venv.envs[0].render
    elif hasattr(venv, 'venv'):
        return get_render_func(venv.venv)
    elif hasattr(venv, 'env'):
        return get_render_func(venv.env)

    return None


def get_vec_normalize(venv):
    if isinstance(venv, VecNormalize):
        return venv
    elif hasattr(venv, 'venv'):
        return get_vec_normalize(venv.venv)

    return None


# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def cleanup_log_dir(log_dir):
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)


class MeganBatchSampler:
    def __init__(self, sampler, num_steps, episode_len, strategy='poisson', m=10):
        assert strategy in ['uniform', 'geometric', 'poisson']
        self.sampler = sampler
        self.num_steps = num_steps
        self.episode_len = episode_len
        self.strategy = strategy
        self.m = m
        # self.episode_end_idx = [min(i * self.episode_len, self.num_steps) for i in range(1, self.num_steps // self.episode_len + 2)]
        # stop = 1

    def __iter__(self):
        if self.strategy == 'uniform':
            return ([min(e + np.random.randint(self.episode_len - e % self.episode_len), self.num_steps - 1) for e in indices] for indices in self.sampler)
        elif self.strategy == 'geometric':
            return ([min(e + np.random.geometric(1-self.m), self.num_steps - 1) for e in indices] for indices in self.sampler)
        elif self.strategy == 'poisson':
            return ([min(e + np.random.poisson(self.m), self.num_steps - 1) for e in indices] for indices in self.sampler)
        else:
            raise NotImplementedError


def sample_eta_gamma(strategy_eta, m_eta, strategy_gamma, m_gamma, boundary):
    assert strategy_eta in ['uniform', 'geometric', 'poisson']
    assert strategy_gamma in ['uniform', 'geometric', 'poisson']
    eta = None
    # First sample eta
    if strategy_eta == 'uniform':
        eta = np.random.randint(boundary)
    if strategy_eta == 'geometric':
        eta = min(np.random.geometric(m_eta), boundary)
    if strategy_eta == 'poisson':
        eta = min(np.random.poisson(m_eta), boundary)

    # Then sample gamma
    if strategy_gamma == 'uniform':
        return eta + np.random.randint(boundary - eta)
    if strategy_gamma == 'geometric':
        return eta + min(np.random.geometric(m_gamma), boundary - eta)
    if strategy_gamma == 'poisson':
        return eta + min(np.random.poisson(m_gamma), boundary - eta)

class MeganBisSampler:
    def __init__(self, sampler, num_steps, episode_len, strategy_eta='uniform', m_eta=25, strategy_gamma='uniform', m_gamma=25):
        self.sampler = sampler
        self.num_steps = num_steps
        self.episode_max_len = episode_len
        self.strategy_eta = strategy_eta
        self.m_eta = m_eta
        self.strategy_gamma = strategy_gamma
        self.m_gamma = m_gamma

        self.len_per_episode = [min((i + 1) * self.episode_max_len, self.num_steps) - i * self.episode_max_len
                                for i in range(self.num_steps // self.episode_max_len + 1)]
        # stop = 1

    def __iter__(self):
        # return ([min(e * self.episode_max_len + np.random.randint(self.len_per_episode[e]), self.num_steps - 1) for e in indices]
        #         for indices in self.sampler)
        return ([min(e * self.episode_max_len + sample_eta_gamma(self.strategy_eta, self.m_eta, self.strategy_gamma, self.m_gamma,
                                                       boundary=self.len_per_episode[e]), self.num_steps - 1) for e in indices]
                for indices in self.sampler)

