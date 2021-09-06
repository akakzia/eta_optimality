import datetime
import gym
import numpy as np
import itertools
from mpi4py import MPI
import torch
from rl_modules.replay_memory import ReplayMemory
from arguments import get_args
import mpi_modules.logger as logger
from utils import init_storage

def main():
    rank = MPI.COMM_WORLD.Get_rank()
    args = get_args()

    # Environment
    env = gym.make(args.env_name)
    env.seed(args.seed + rank)
    env.action_space.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    # Agent and memory
    if args.algo == "SAC":
        from rl_modules.sac import SAC
        agent = SAC(env.observation_space.shape[0], env.action_space, args)
    elif args.algo == "GSAC":
        from rl_modules.gsac import GSAC
        agent = GSAC(env.observation_space.shape[0], env.action_space, args)
    else:
        raise NotImplementedError

    # Replay buffer
    memory = ReplayMemory(args.replay_size, args.seed)

    # Set up logger
    if rank == 0:
        logdir, model_path = init_storage(args)
        logger.configure(dir=logdir)
        logger.info(vars(args))

    # stats
    stats = dict()
    stats['episodes'] = []
    stats['environment_steps'] = []
    stats['updates'] = []
    stats['avg_reward'] = []

    # Training Loop
    total_numsteps = 0
    updates = 0

    for i_episode in itertools.count(1):
        episode_reward = 0
        episode_steps = 0
        # done = False
        state = env.reset()

        # while not done:
        while episode_steps < args.max_episode_steps:
            # if args.start_steps > total_numsteps:
            #     action = env.action_space.sample()  # Sample random action
            # else:
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
            mask = 1 if episode_steps == env._max_episode_steps else float(not done)
            memory.push(state, action, reward, next_state, mask)  # Append transition to memory
            episode_steps += 1
            total_numsteps += 1

            state = next_state

            if done and episode_steps < env._max_episode_steps:
                state = env.reset()
                episode_reward = 0

        if total_numsteps > args.num_steps:
            break

        global_episode_reward = MPI.COMM_WORLD.allreduce(episode_reward, op=MPI.SUM) / MPI.COMM_WORLD.Get_size()
        if rank == 0:
            logger.info('Episode: {}, total numsteps: {}, episode steps: {}, reward: {}'.format(i_episode, total_numsteps, episode_steps,
                                                                                                global_episode_reward))
            # print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps,
            #                                                                               global_episode_reward))

        if i_episode % args.eval_every == 0 and args.eval:
            avg_reward = 0.
            episodes = args.eval_episodes
            for _ in range(episodes):
                state = env.reset()
                episode_reward = 0
                done = False
                while not done:
                    _, action, _ = agent.select_action(state, evaluate=True)

                    next_state, reward, done, _ = env.step(action)
                    episode_reward += reward


                    state = next_state
                avg_reward += episode_reward
            avg_reward /= episodes

            global_avg_reward = MPI.COMM_WORLD.allreduce(avg_reward, op=MPI.SUM) / MPI.COMM_WORLD.Get_size()
            if rank == 0:
                log_and_save(stats, i_episode, total_numsteps, updates, global_avg_reward)
                agent.save_model(path=model_path)
                # print("----------------------------------------")
                # print("Test Episodes: {}, Avg. Reward: {}".format(episodes, global_avg_reward))
                # print("----------------------------------------")

    env.close()

def log_and_save(stats, i_episode, total_numsteps, updates, avg_rew):
    stats['episodes'].append(i_episode)
    stats['environment_steps'].append(total_numsteps)
    stats['updates'].append(updates)
    stats['avg_reward'].append(avg_rew)
    for k, l in stats.items():
        logger.record_tabular(k, l[-1])
    logger.dump_tabular()

if __name__ == "__main__":
    main()
