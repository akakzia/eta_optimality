# import math
# import torch
from datetime import datetime
import os
import json

# def create_log_gaussian(mean, log_std, t):
#     quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
#     l = mean.shape
#     log_z = log_std
#     z = l[-1] * math.log(2 * math.pi)
#     log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
#     return log_p
#
# def logsumexp(inputs, dim=None, keepdim=False):
#     if dim is None:
#         inputs = inputs.view(-1)
#         dim = 0
#     s, _ = torch.max(inputs, dim=dim, keepdim=True)
#     outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
#     if not keepdim:
#         outputs = outputs.squeeze(dim)
#     return outputs

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def init_storage(args):
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    # path to save the model
    logdir = os.path.join(args.save_dir, '{}_{}_{}'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), args.algo,
                                                           args.env_name))
    # if args.algo == 'SAC':
    #     logdir += '_{}'.format(args.gamma)
    # else:
    #     logdir += '_{}'.format(args.update_frequency)
    # path to save evaluations
    model_path = os.path.join(logdir, 'models')
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    with open(os.path.join(logdir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    return logdir, model_path