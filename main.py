import datetime
import gym
import numpy as np
import itertools
from collections import deque
from mpi4py import MPI
import torch
from sac import SAC
from ppo_bis import PPO
from replay_memory import ReplayMemory
from model import Policy
from storage import RolloutStorage
from envs import make_vec_envs, make_env
from arguments import get_args
from mpi_utils import sync_mpis_data


NUM_STEP_PPO = 1000


def main():
    args = get_args()

    # Environment
    env = gym.make(args.env_name)
    env.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    env.action_space.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    torch.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    np.random.seed(args.seed + MPI.COMM_WORLD.Get_rank())

    # Agent and memory
    if args.algo == "SAC":
        agent = SAC(env.observation_space.shape[0], env.action_space, args)
        # Replay buffer
        memory = ReplayMemory(args.replay_size, args.seed)
    elif args.algo == "PPO":
        agent = PPO(env.observation_space.shape[0], env.action_space.shape[0])
        memory = RolloutStorage(NUM_STEP_PPO, MPI.COMM_WORLD.Get_size(), env.observation_space.shape, env.action_space)
        state = env.reset()
        memory.obs[0].copy_(torch.FloatTensor(state).unsqueeze(0))
        memory.to(torch.device("cpu"))
    else:
        agent = None
        memory = None

    # Training Loop
    total_numsteps = 0
    updates = 0

    for i_episode in itertools.count(1):
        # TODO create a rollout module
        episode_reward = 0
        episode_steps = 0
        done = False
        state = env.reset()
        if args.algo == 'PPO':
            state = torch.FloatTensor(state).unsqueeze(0)

        while not done:
            if args.start_steps > total_numsteps and args.algo == 'SAC':
                action = env.action_space.sample()  # Sample random action
                value = None
                action_log_prob = None
            else:
                value, action, action_log_prob = agent.select_action(state)
            if args.algo == "SAC":
                if len(memory) > args.batch_size:
                    # Number of updates per step in environment
                    for i in range(args.updates_per_step):
                        # Update parameters of all the networks
                        critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)
                        updates += 1

            # Environment Step
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            if args.algo == 'SAC':
                # Ignore the "done" signal if it comes from hitting the time horizon.
                # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
                mask = 1 if episode_steps == env._max_episode_steps else float(not done)
                memory.push(state, action, reward, next_state, mask)  # Append transition to memory
            else:
                # If done then clean the history of observations.
                masks = torch.FloatTensor([[0.0] if done else [1.0]])
                bad_masks = torch.FloatTensor([[1.0] if done and episode_steps == env._max_episode_steps else [0.0]])
                next_state = torch.FloatTensor(next_state).unsqueeze(0)
                reward = reward.unsqueeze(0)
                # Stack data from all cpus
                sync_mpis_data([next_state, action, action_log_prob, value, reward, masks, bad_masks])
                memory.insert(next_state, action, action_log_prob, value, reward, masks, bad_masks)
            episode_steps += 1
            total_numsteps += 1

            state = next_state

            if episode_steps % NUM_STEP_PPO == 0 and args.algo == 'PPO':
                with torch.no_grad():
                    next_value = agent.get_value(state).detach()
                memory.compute_returns(next_value, False, args.gamma, 0.95, False)
                # Synchronize rollout storage for all cpus

                value_loss, action_loss, dist_entropy = agent.update_parameters(memory)
                memory.after_update()

        if total_numsteps > args.num_steps:
            break

        global_episode_reward = MPI.COMM_WORLD.allreduce(episode_reward, op=MPI.SUM) / MPI.COMM_WORLD.Get_size()
        if MPI.COMM_WORLD.Get_rank() == 0:
            print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps,
                                                                                          global_episode_reward))

        if i_episode % 10 == 0 and args.eval:
            avg_reward = 0.
            episodes = 10
            for _ in range(episodes):
                state = env.reset()
                episode_reward = 0
                done = False
                while not done:
                    try:
                        _, action, _ = agent.select_action(state, evaluate=True)

                        next_state, reward, done, _ = env.step(action)
                    except:
                        _, action, _ = agent.select_action(torch.FloatTensor(state).unsqueeze(0), evaluate=True)

                        next_state, reward, done, _ = env.step(action)
                    episode_reward += reward


                    state = next_state
                avg_reward += episode_reward
            avg_reward /= episodes

            global_avg_reward = MPI.COMM_WORLD.allreduce(avg_reward, op=MPI.SUM) / MPI.COMM_WORLD.Get_size()
            if MPI.COMM_WORLD.Get_rank() == 0:
                print("----------------------------------------")
                print("Test Episodes: {}, Avg. Reward: {}".format(episodes, global_avg_reward))
                print("----------------------------------------")

    env.close()


if __name__ == "__main__":
    main()
