import datetime
import gym
import numpy as np
import itertools
from collections import deque
from mpi4py import MPI
import torch
from sac import SAC
from gsac import GSAC
from replay_memory import ReplayMemory
from arguments import get_args


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
    elif args.algo == "GSAC":
        agent = GSAC(env.observation_space.shape[0], env.action_space, args)
    else:
        raise NotImplementedError

    # Replay buffer
    memory = ReplayMemory(args.replay_size, args.seed)

    # Training Loop
    total_numsteps = 0
    updates = 0

    for i_episode in itertools.count(1):
        episode_reward = 0
        episode_steps = 0
        done = False
        state = env.reset()

        while not done:
            if args.start_steps > total_numsteps:
                action = env.action_space.sample()  # Sample random action
            else:
                value, action, action_log_prob = agent.select_action(state)

            if len(memory) > args.batch_size and episode_steps % args.update_every == 0:
                # Number of updates per step in environment
                for i in range(args.updates_per_step):
                    # Update parameters of all the networks
                    agent.update_parameters(memory, args.batch_size, updates)
                    updates += 1
            # Environment Step
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            mask = 1 if episode_steps == env._max_episode_steps else float(not done)
            memory.push(state, action, reward, next_state, mask)  # Append transition to memory
            episode_steps += 1
            total_numsteps += 1

            state = next_state

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
