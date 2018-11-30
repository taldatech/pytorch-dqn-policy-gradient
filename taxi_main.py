'''
This is the main script to run OpenAi's Taxi-v2 environment
'''

# imports
import argparse
import gym
import torch
import numpy as np
import random
import time
from datetime import datetime
import pickle
import os
from itertools import count
from utils.Schedule import LinearSchedule, ExponentialSchedule
from utils.Agent import TaxiAgent
from utils.OneHotGenerator import OneHotGenerator
from utils.Helpers import state_to_location, calc_moving_average, plot_rewards, state_to_one_hots, play_taxi
import matplotlib
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="train and play a Taxi-v2 agent")
    # modes
    parser.add_argument("-t", "--train", help="train or continue training an agent",
                        action="store_true")
    parser.add_argument("-p", "--play", help="play the environment using an a pretrained agent",
                        action="store_true")
    # arguments
    # for training and playing
    parser.add_argument("-n", "--name", type=str,
                        help="model name, for saving and loading, if not set, training will continue from a pretrained checkpoint")
    parser.add_argument("-m", "--mode", type=str,
                        help="model's mode or state representation ('one-hot', 'location-one-hot'), default: 'one-hot'")
    parser.add_argument("-e", "--episodes", type=int,
                        help="number of episodes to play or train, default: 2 (play), 5000 (train)")
    # for training
    parser.add_argument("-x", "--exploration", type=str,
                       help="epsilong-greedy scheduling ('exp', 'lin'), default: 'exp'")
    parser.add_argument("-d", "--decay_rate", type=int,
                        help="number of episodes for epsilon decaying, default: 800")
    parser.add_argument("-u", "--hidden_units", type=int,
                        help="number of neurons in the hidden layer of the DQN, default: 150")
    parser.add_argument("-o", "--optimizer", type=str,
                        help="optimizing algorithm ('RMSprop', 'Adam'), deafult: 'RMSProp'")
    parser.add_argument("-r", "--learn_rate", type=float,
                        help="learning rate for the optimizer, default: 0.0003")
    parser.add_argument("-g", "--gamma", type=float,
                        help="gamma parameter for the Q-Learning, default: 0.99")
    parser.add_argument("-s", "--buffer_size", type=int,
                        help="Replay Buffer size, default: 500000")
    parser.add_argument("-b", "--batch_size", type=int,
                        help="number of samples in each batch, default: 128")
    parser.add_argument("-i", "--steps_to_start_learn", type=int,
                        help="number of steps before the agents starts learning, default: 1000")
    parser.add_argument("-c", "--target_update_freq", type=int,
                        help="number of steps between copying the weights to the target DQN, default: 5000")
    parser.add_argument("-a", "--clip_grads", help="use Gradient Clipping regularization (default: False)",
                        action="store_true")
    parser.add_argument("-z", "--batch_norm", help="use Batch Normalization between DQN's layers (default: False)",
                        action="store_true")
    parser.add_argument("-y", "--dropout", help="use Dropout regularization on the layers of the DQN (default: False)",
                        action="store_true")
    parser.add_argument("-q", "--dropout_rate", type=float,
                        help="probability for a layer to be dropped when using Dropout, default: 0.4")
    args = parser.parse_args()

    # Training
    if (args.train):
        if (args.name):
            name = args.name
        else:
            name = None
        if (args.mode):
            if (args.mode == 'one-hot') or (args.mode == 'location-one-hot'):
                obs_represent = args.mode
            else:
                obs_represent = 'one-hot'
        else:
            obs_represent = 'one-hot'

        if (args.episodes):
            num_episodes = args.episodes
        else:
            num_episodes = 5000
        if (args.exploration):
            if (args.decay_rate):
                episode_deacy = args.decay_rate
            else:
                episode_deacy = 800
            if (args.exploration == 'lin'):
                exploration = LinearSchedule(total_timesteps=episode_deacy)
            else:
                exploration = ExponentialSchedule(decay_rate=episode_deacy)
        else:
            exploration = ExponentialSchedule(decay_rate=800)
        if (args.hidden_units):
            hidden_units = args.hidden_units
        else:
            hidden_units = 150
        if (args.learn_rate):
            lr = args.learn_rate
        else:
            lr = 0.0003
        if (args.optimizer == 'RMSProp' or args.optimizer == 'Adam'):
            optimizer = args.optimizer
        else:
            optimizer = 'RMSprop'
        if (args.gamma):
            gamma = args.gamma
        else:
            gamma = 0.99
        if (args.buffer_size):
            buffer_size = args.buffer_size
        else:
            buffer_size = 500000
        if (args.batch_size):
            batch_size = args.batch_size
        else:
            batch_size = 128
        if (args.steps_to_start_learn):
            steps_to_start_learn = args.steps_to_start_learn
        else:
            steps_to_start_learn = 1000
        if (args.target_update_freq):
            target_update_freq = args.target_update_freq
        else:
            target_update_freq = 5000
        if (args.clip_grads):
            clip_grads = True
            print("using Gradient Clipping")
        else:
            clip_grads = False
        if (args.batch_norm):
            batch_norm = True
            print("using Batch Normalization")
        else:
            batch_norm = False
        if (args.dropout):
            dropout = True
            print("using Dropout")
        else:
            dropout = False
        if (args.dropout_rate):
            dropout_rate = args.dropout_rate
        else:
            dropout_rate = 0.4
        
        env = gym.make('Taxi-v2')
        if name is not None:
            agent = TaxiAgent(env,
                             name=name,
                            obs_represent=obs_represent,
                            exploration=exploration,
                            optimizer=optimizer,
                            learning_rate=lr,
                            n_hidden=hidden_units,
                            gamma=gamma,
                            replay_buffer_size=buffer_size,
                            steps_to_start_learn=steps_to_start_learn,
                            target_update_freq=target_update_freq,
                            clip_grads=clip_grads,
                            use_batch_norm=batch_norm,
                            use_dropout=dropout,
                            dropout_rate=dropout_rate)
        else:
            name = 'user_' + datetime.utcnow().strftime("%Y%m%d%H%M%S")
            agent = TaxiAgent(env,
                             name=name,
                            obs_represent=obs_represent,
                            exploration=exploration,
                            optimizer=optimizer,
                            learning_rate=lr,
                            n_hidden=hidden_units,
                            gamma=gamma,
                            replay_buffer_size=buffer_size,
                            steps_to_start_learn=steps_to_start_learn,
                            target_update_freq=target_update_freq,
                            clip_grads=clip_grads,
                            use_batch_norm=batch_norm,
                            use_dropout=dropout,
                            dropout_rate=dropout_rate)
            if obs_represent == 'location-one-hot':
                agent.load_agent_state(path='./taxi_agent_ckpt/taxi_agent_location_one_hots.pth', copy_to_target_network=True, load_optimizer=True)
            else:
                agent.load_agent_state(path='./taxi_agent_ckpt/taxi_agent_one_hot_pretrained.pth', copy_to_target_network=True, load_optimizer=True)
        LOG_EVERY_N_STEPS = 5000
        training_status_path = './taxi_agent_ckpt/' + agent.name + '_training.status'
        if (os.path.isfile(training_status_path)):
            with open(training_status_path, 'rb') as fp:
                training_status = pickle.load(fp)
                mean_episode_reward = training_status['mean_episode_reward']
                best_mean_episode_reward = training_status['best_mean_episode_reward']
                episode_durations = training_status['episode_durations'] 
                episodes_rewards = training_status['episodes_rewards']
                total_steps = training_status['total_steps']
        else:
            mean_episode_reward = -float('nan')
            best_mean_episode_reward = -float('inf')
            episode_durations = []
            episodes_rewards = []
            total_steps = 0

        # Start training
        start_time = time.time()
        for episode in range(num_episodes):
            episode_start_time = time.time()
            last_obs = env.reset()
            episode_reward = 0
            agent.episodes_seen += 1
            for t in count():
                agent.steps_count += 1
                total_steps += 1
                ### Step the env and store the transition
                # Store lastest observation in replay memory and last_idx can be
                # used to store action, reward, done
                if (obs_represent == 'location-one-hot'):
                    last_idx = agent.replay_buffer.store_frame(state_to_one_hots(list(env.env.decode(last_obs))))
                else:
                    last_idx = agent.replay_buffer.store_frame(agent.one_hot_generator.to_one_hot(last_obs))
                # encode_recent_observation will take the latest observation
                # that you pushed into the buffer and compute the corresponding
                # input that should be given to a Q network by appending some
                # previous frames.
                recent_observation = agent.replay_buffer.encode_recent_observation()
                action = agent.select_greedy_action(recent_observation)
                obs, reward, done, _ = env.step(action)
                episode_reward += reward
                # Store other info in replay memory
                agent.replay_buffer.store_effect(last_idx, action, reward, done)
                ### Perform experience replay and train the network.
                agent.learn(batch_size)
                ### Log progress and keep track of statistics
                if len(episodes_rewards) > 0:
                    mean_episode_reward = np.mean(episodes_rewards[-100:])
                if len(episodes_rewards) > 100:
                    best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)

                if total_steps % LOG_EVERY_N_STEPS == 0 and total_steps > agent.steps_to_start_learn:
                    print("Timestep %d" % (agent.steps_count,))
                    print("mean reward (100 episodes) %f" % mean_episode_reward)
                    print("best mean reward %f" % best_mean_episode_reward)
                    print("episodes %d" % len(episodes_rewards))
                    print("exploration value %f" % agent.epsilon)
                    total_time = time.time() - start_time
                    print("time since start: %.2f secs" % total_time)
                    training_status = {}
                    training_status['mean_episode_reward'] = mean_episode_reward
                    training_status['best_mean_episode_reward'] = best_mean_episode_reward
                    training_status['episode_durations'] = episode_durations
                    training_status['episodes_rewards'] = episodes_rewards
                    training_status['total_steps'] = total_steps
                    with open(training_status_path, 'wb') as fp:
                        pickle.dump(training_status, fp)
                    print("Saved training status @ ", training_status_path)
                # Resets the environment when reaching an episode boundary.
                if done:
                    episode_durations.append(t + 1)
                    episodes_rewards.append(episode_reward)
                    print("Episode: ", agent.episodes_seen,
                          " Done, Reward: ", episode_reward,
                          " Step: ", agent.steps_count,
                         " Episode Time: %.2f secs" % (time.time() - episode_start_time))
                    break
                last_obs = obs
        training_status = {}
        training_status['mean_episode_reward'] = mean_episode_reward
        training_status['best_mean_episode_reward'] = best_mean_episode_reward
        training_status['episode_durations'] = episode_durations
        training_status['episodes_rewards'] = episodes_rewards
        training_status['total_steps'] = total_steps
        with open(training_status_path, 'wb') as fp:
            pickle.dump(training_status, fp)
        print("Saved training status @ ", training_status_path)
        agent.save_agent_state()
        print("Training Complete!")
        # Plot
        if (len(episodes_rewards) < 100):
            plot_rewards(episodes_rewards, 10)
        else:
            plot_rewards(episodes_rewards, 100)

    # Playing
    elif(args.play):
        if (args.name):
            name = args.name
        else:
            name = None
        if (args.mode):
            if (args.mode == 'one-hot') or (args.mode == 'location-one-hot'):
                obs_represent = args.mode
            else:
                obs_represent = 'one-hot'
        else:
            obs_represent = 'one-hot'

        if (args.episodes):
            num_episodes = args.episodes
        else:
            num_episodes = 2

        env = gym.make('Taxi-v2')
        if name is not None:
            agent = TaxiAgent(env, name=name, obs_represent=obs_represent)
        else:
            name = 'user_' + datetime.utcnow().strftime("%Y%m%d%H%M%S")
            agent = TaxiAgent(env, name=name, obs_represent=obs_represent)
            if obs_represent == 'location-one-hot':
                agent.load_agent_state(path='./taxi_agent_ckpt/taxi_agent_location_one_hots.pth', copy_to_target_network=True, load_optimizer=True)
            else:
                agent.load_agent_state(path='./taxi_agent_ckpt/taxi_agent_one_hot_pretrained.pth', copy_to_target_network=True, load_optimizer=True)
        # PLAY
        play_taxi(env, agent, num_episodes)
    else:
        print("No mode selected")
        raise SystemExit("no --train or --play flags")

if __name__ == "__main__":
    main()