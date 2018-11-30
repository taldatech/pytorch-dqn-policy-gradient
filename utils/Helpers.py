'''
Helper functions for the Taxi and Acrobot environments
'''
# imports
import gym
import torch
import numpy as np
import random
import time
import pickle
import os
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import clear_output
from utils.OneHotGenerator import OneHotGenerator
from utils.Agent import preprocess_frame
from itertools import count

def state_to_location(state):
    state = np.array(list(env.env.decode(state)), dtype=np.float)
    # normalize
#     state[0] = state[0] / 4.0 # 5 x 5
#     state[1] = state[1] / 4.0 # 5 x 5
#     state[2] = state[2] / 4.0 # 5 locations for passenger
#     state[3] = state[3] / 3.0 # 4 destinations
    return state

def calc_moving_average(lst, window_size=10):
    '''
    This function calculates the moving average of `lst` over
    `window_size` samples.
    Parameters:
        arr: list (list)
        window_size: size over which to average (int)
    Returns:
        mean_arr: array with the averages (np.array)
    '''
    assert len(lst) >= window_size
    mean_arr = []
    for j in range(1, window_size):
        mean_arr.append(np.mean(lst[:j]))
    i = 0
    while i != (len(lst) - window_size + 1):
        mean_arr.append(np.mean(lst[i : i + window_size]))
        i += 1
    return np.array(mean_arr)

def plot_rewards(episode_rewards, window_size=10, title=''):
    '''
    This function plots the rewards vs. episodes and the mean rewards vs. episodes.
    The mean is taken over `windows_size` episodes.
    Parameters:
        episode_rewards: list of all the rewards (list)
    '''
    num_episodes = len(episode_rewards)
    mean_rewards = calc_moving_average(episode_rewards, window_size)
    plt.plot(list(range(num_episodes)), episode_rewards, label='rewards')
    plt.plot(list(range(num_episodes)), mean_rewards, label='mean_rewards')
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.show()
    
def state_to_one_hots(state):
    '''
    This function takes a state in location-tuple form and encodes it as
    a concatenation of one-hots.
    Parameters:
        state: state in location list/tuple form (Taxi Row, Taxi Col, PassLoc, DestIdx)
    Returns:
        state_one_hots: np.array of zeros and ones representing the state
    '''
    assert len(state) == 4
    four_one_hot_gen = OneHotGenerator(5)
    three_one_hot_gen = OneHotGenerator(4)
    row_one_hot = four_one_hot_gen.to_one_hot(state[0])
    col_one_hot = four_one_hot_gen.to_one_hot(state[1])
    passenger_one_hot = four_one_hot_gen.to_one_hot(state[2])
    dest_one_hot = three_one_hot_gen.to_one_hot(state[3])
    return np.concatenate((row_one_hot, col_one_hot, passenger_one_hot, dest_one_hot), axis=0)

def play_taxi(env, agent, num_episodes=5):
    '''
    This function plays the Taxi-v2 environment given an agent.
    Parameters:
        agent: the agent that holds the policy (TaxiAgent)
        num_episodes: number of episodes to play
    '''
    if agent.obs_represent == 'location-one-hot':
        print("Playing Taxi-v2 with " , agent.name ,"agent using Location-One-Hot representation")
        max_steps_per_episode = 300
        for episode in range(num_episodes):
            episode_start_time = time.time()
            obs = env.reset()
            obs = state_to_one_hots(list(env.env.decode(obs)))
            done = False
            print("Episode: ", episode + 1)
            time.sleep(1)
            r = 0
            last_action = 0
            for step in range(max_steps_per_episode):
                clear_output(wait=True)
                env.render()
                time.sleep(1)
                action = agent.predict_action(obs)
                new_obs, reward, done, info = env.step(action)
                r += reward
                if done:
                    clear_output(wait=True)
                    env.render()
                    time.sleep(3)
                    clear_output(wait=True)
                    print("Reward: ", r, "Total time: %.2f" % (time.time() - episode_start_time))
                    break
                last_action = action
                obs = state_to_one_hots(list(env.env.decode(new_obs)))
        env.close()
    else:
        # mode == 'one-hot'
        print("Playing Taxi-v2 with " , agent.name ,"agent using One-Hot representation")
        max_steps_per_episode = 300
        for episode in range(num_episodes):
            episode_start_time = time.time()
            obs = env.reset()
            obs = agent.one_hot_generator.to_one_hot(obs)
            done = False
            print("Episode: ", episode + 1)
            time.sleep(1)
            r = 0
            last_action = 0
            for step in range(max_steps_per_episode):
                clear_output(wait=True)
                env.render()
                time.sleep(1)
                with torch.no_grad():
                    obs = torch.from_numpy(obs).unsqueeze(0).type(torch.FloatTensor).to(agent.device)
                    q_actions = agent.Q_target(obs)
                    action = torch.argmax(q_actions).item()

                new_obs, reward, done, info = env.step(action)
                r += reward
                if done:
                    clear_output(wait=True)
                    env.render()
                    time.sleep(3)
                    clear_output(wait=True)
                    print("Reward: ", r, " Total time: %.2f" % (time.time() - episode_start_time) )
                    break
                last_action = action
                obs = agent.one_hot_generator.to_one_hot(new_obs)
        env.close()

def play_acrobot(env, agent, num_episodes=5):
    '''
    This function plays the Acrobot-v1 environment given an agent.
    Parameters:
        agent: the agent that holds the policy (AcrobotAgent)
        num_episodes: number of episodes to play
    '''
    if agent.obs_represent == 'frame_seq':
        print("Playing Acrobot-v1 with " , agent.name ,"agent using Frame Sequence")
        start_time = time.time()
        for episode in range(num_episodes):
            print("### Episode ", episode + 1, " ###")
            episode_start_time = time.time()
            env.reset()
            last_obs = preprocess_frame(env, mode='atari', render=True)
            episode_reward = 0
            for t in count():
                ### Step the env and store the transition
                # Store lastest observation in replay memory and last_idx can be used to store action, reward, done
                last_idx = agent.replay_buffer.store_frame(last_obs)
                # encode_recent_observation will take the latest observation
                # that you pushed into the buffer and compute the corresponding
                # input that should be given to a Q network by appending some
                # previous frames.
                recent_observation = agent.replay_buffer.encode_recent_observation()
                action = agent.predict_action(recent_observation)
                _ , reward, done, _ = env.step(action)
                episode_reward += reward
                # Store other info in replay memory
                agent.replay_buffer.store_effect(last_idx, action, reward, done)
                if done:
                    print("Episode: ", episode + 1, " Done, Reward: ", episode_reward, 
                          " Episode Time: %.2f secs" % (time.time() - episode_start_time))
                    break
                last_obs = preprocess_frame(env, mode='atari', render=True)
        env.close()
    else:
        # mode == 'frame diff'
        print("Playing Acrobot-v1 with " , agent.name ,"agent using Frame Difference")
        start_time = time.time()
        for episode in range(num_episodes):
            print("### Episode ", episode + 1, " ###")
            episode_start_time = time.time()
            env.reset()
            last_obs = preprocess_frame(env, mode='control', render=True)
            current_obs = preprocess_frame(env, mode='control', render=True)
            state = current_obs - last_obs
            episode_reward = 0
            for t in count():
                action = agent.predict_action(state)
                _ , reward, done, _ = env.step(action)
                episode_reward += reward
                if done:
                    print("Episode: ", episode + 1, " Done, Reward: ",
                          episode_reward,
                         " Episode Time: %.2f secs" % (time.time() - episode_start_time))
                    break
                last_obs = current_obs
                current_obs = preprocess_frame(env, mode='control', render=True)
                state = current_obs - last_obs
        env.close()
