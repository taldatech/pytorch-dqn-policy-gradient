import sys
import os
import time

import gym
import numpy as np
from itertools import count
import utils.taxi_policy as taxi_policy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import pickle
import argparse


class PGAgent(nn.Module):
    def __init__(self, number_of_actions, num_of_states, HL_size, gamma, exploration, device, clip_grads):
        """
        this class represnts an agient which learns with policy gradient methods
        in the simple case the policy is represented by a shallow NN with the following proprities
        :param number_of_actions: the number of actions available to the agent, which is also the output dimension
                                    of the network that represents the policy
        :param num_of_states: if state is represented by a vector then this would be the size of the vector
                            which is also the size of the NN input layer
        :param HL_size: hidden layer size
        :param gamma: the discount factor
        """
        super(PGAgent, self).__init__()

        self.num_of_states, self.number_of_actions = num_of_states, number_of_actions
        self.device = device

        # arch with no variance reduction AC
        # self.affine1 = nn.Linear(num_of_states, HL_size)
        # self.affine2 = nn.Linear(HL_size, number_of_actions)

        # another arch i played with
        # self.affine1 = nn.Linear(num_of_states, 2048)
        # self.affine2 = nn.Linear(2048, 1024)
        # self.affine3 = nn.Linear(1024, number_of_actions)

        # architecture that also predicts the value function
        self.affine1 = nn.Linear(num_of_states, HL_size)
        self.affine2 = nn.Linear(HL_size, number_of_actions)
        self.affine3 = nn.Linear(HL_size, 1)  # value function is just one number

        self.clip_grads = clip_grads
        self.optimizer = None
        self.gamma = gamma
        self.explore_factor = exploration
        assert 0 < gamma <= 1, 'gamma i.e. the discount factor should be between 0 and 1'
        assert 0 < exploration <= 1, 'epsilon i.e. the exploration factor should be between 0 and 1'

        self.saved_log_probs = []   # saves chosen actions log probability during an episode
        self.rewards = []           # saves the chosen actions rewards during an episode
        self.save_value_function = []    # saves the states during an episode

    def forward(self, state):
        """
        defines the forward pass in the shallow NN we defined
        note: this is a private method and not a part of the api
        :param state:
        :return: softmax on actions
        """
        x = F.relu(self.affine1(state))
        V_s = self.affine3(x)
        x = F.relu(self.affine2(x))
        action_scores = x
        return F.softmax(action_scores, dim=1), V_s

    def set_optimizer(self, optimizer):
        """

        :param optimizer: optimerzer object from the torch.optim library
        :return: None
        """
        if not isinstance(optimizer,torch.optim.Optimizer):
            raise ValueError(' the given optimizer is not supported'
                             'please provide an optimizer that is an instance of'
                             'torch.optim.Optimizer')
        self.optimizer = optimizer

    def select_action(self, state):
        """
        when the agent is given a state of the world use this method to make the agent chose an action acording
        to the policy
        :param state: state of the world, with the same diminsions that the agient knows
        :return: sampled action according to the policy the agent learned
        """
        one_hot = np.zeros([1, self.num_of_states])  # encode state in oneHot vector
        one_hot[0, state] = 1
        state = torch.tensor(one_hot, device=self.device).float()
        probs, V_s = self.forward(state)
        pai_s = Categorical(probs)

        if self.should_explore():
            action = torch.tensor(taxi_policy.optimal_human_policy(int(state.argmax())), device=self.device)
        else:
            action = pai_s.sample()

        self.saved_log_probs.append(pai_s.log_prob(action))
        self.save_value_function.append(V_s)
        return action.item()

    def should_explore(self):
        """
        applies greedy exploration policy epsilon greedy
        with diminishing epsilon as the agent gets better
        :return: returns true if the agent should explore
        """
        explore = Categorical(torch.tensor([1 - self.explore_factor, self.explore_factor])).sample()
        return explore == 1

    def random_action(self):
        """
        retunrs a random action polled from discrete uniform distribution over the number of actions
        :return: a random action
        """
        uniform_sampler = Categorical(torch.tensor([1/self.number_of_actions]*self.number_of_actions))
        return uniform_sampler.sample()

    def update(self, episode):
        """
        after an episode is finished use this method to update the policy the agent has learned so far acording
        to the monte carlo samples the agent have seen during the episode
        episode parameter is for "episode normalization" code which is not used
        :return: policy loss
        """
        if self.optimizer is None:
            raise ValueError('optimizer not set!'
                             'please use agent.set_optimizer method to set an optimizer')

        R = 0
        policy_loss = []
        value_losses = []
        rewards = []
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            rewards.insert(0, R)

        rewards = torch.tensor(rewards, device=self.device)
        # code for trying baseline reduction
        # V_s = torch.tensor(np.zeros(self.number_of_actions), device=self.device)
        # for state in range(self.num_of_states):
        #     V_s[state] = rewards[self.visited_states == state].mean()

        # code for reward normalization trick i found online .. helps with convergence

        eps = np.finfo(np.float32).eps.item()
        rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
        for log_prob, reward, v_s in zip(self.saved_log_probs, rewards, self.save_value_function):
            advantage = reward - v_s.item()
            policy_loss.append(-log_prob * advantage)
            value_losses.append(F.smooth_l1_loss(v_s.squeeze(), torch.tensor([reward], device=self.device)))

        total_loss = torch.stack(policy_loss).sum().to(self.device) + torch.stack(value_losses).sum().to(self.device)
        self.optimizer.zero_grad()
        total_loss.backward()

        if self.clip_grads:
            for param in self.parameters():
                param.grad.data.clamp_(-1, 1)

        self.optimizer.step()
        del self.rewards[:]
        del self.saved_log_probs[:]
        del self.save_value_function[:]
        return float(total_loss)


def eval_agent(agent, env, num_of_eps=1, render=False):
    """
    evaluates the given PG agent with the env for  ' num_of_eps '
    note: sets the PG agent explore_factor to 0 before evaluating and restores it afterwards
    :param agent: agent
    :param env: env
    :param num_of_eps: number of episodes
    :param render: ef we should render each episode
    :return: the rewards collected in each episode
    """
    orig_exp_factor = agent.explore_factor
    agent.explore_factor = 0.0001
    rewards = []
    for episode in range(num_of_eps):
        state = env.reset()
        episode_reward = []
        for t in range(201):
            action = agent.select_action(state)
            state, reward, done, _ = env.step(action)
            agent.rewards.append(reward)
            if render:
                os.system('cls')
                env.render()
                time.sleep(1)
            episode_reward.append(reward)
            if done:
                break

        # time.sleep(2)
        rewards.append(np.sum(episode_reward))
        print('Episode {0}\t length: {1}\tepisode reward: {2} '.format(episode, t, rewards[-1]))

    agent.explore_factor = orig_exp_factor
    return rewards


def save_pickle(serializable_object, save_base_dir, pickle_name):
    """
    saves the serializable object into a pickle with the given name into the given directory
    :param serializable_object: object to pickle
    :param save_base_dir: where to save
    :param pickle_name: filename
    :return:
    """

    file_name = os.path.join(save_base_dir, pickle_name)
    pickle.dump(serializable_object, open(file_name, "wb"))


def main(load, agent_path, HL_size, save_base_dir='.', num_train_episodes=50000, num_eval_eps=3, render=True):
    """
    loads and evaluates agent or trains and evaluates the agent according to args
    if chosen to train the trained agent will be saved in the following format
    PG_taxi_agent_HL{HL_size}_trained{num_train_episodes}.pt with stats collected during training in
    save_base_dir\stats\[StatName]_HL{HL_size}_trained{num_train_episodes}.p
    saved stats are:  rewards (in training episodes), losses (in training episodes),
    states (starting states in training episodes), episode_len( in training ...), eval_rewards
    :param load: bool if to load the agent given in agent path and has Hidden layer size of HL_size
    :param agent_path: agent to load. pytorch state_dict (relevant if load is true)
    :param HL_size: size of the hidden layer for the agent to create and train or to load
    :param save_base_dir: where to save the newly created and trained agent (relevant if load is false)
    :param num_train_episodes: number of training episodes to train (relevant if load is false)
    :param num_eval_eps: number of evaluation episodes we want either after training or loading
    :param render: bool if we want a visual rendering of the evaluation episodes in the console (will render every ep.)
    :return: list of rewards collected in the evaluation episodes
    """

    # env and agent initialization
    seed = 0
    env = gym.make('Taxi-v2')
    env.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    agent = PGAgent(env.action_space.n, env.observation_space.n,
                    HL_size=HL_size, gamma=0.99, exploration=0.1, device=device, clip_grads=False)
    optimizer = optim.Adam(agent.parameters(), lr=1e-2)
    agent.set_optimizer(optimizer)
    agent.to(device)

    # load and evaluate
    if load:
        agent.load_state_dict(torch.load(agent_path))
        rewards = eval_agent(agent, env, render=render, num_of_eps=num_eval_eps)
        return rewards

    # training
    # initializing statistics
    last_100_ep_reward = []
    loss_in_episodes = []
    rewards_in_episodes = []
    starting_states = []
    episode_len = []

    for i_episode in count(1):
        state = env.reset()
        ep_len = 1000
        for t in range(1000):  # Don't infinite loop while learning

            action = agent.select_action(state)
            state, reward, done, _ = env.step(action)
            agent.rewards.append(reward)

            if i_episode % 200 == 0:  # option to render states in "video" form
                env.render()
            if done:
                ep_len = t
                break

        episode_reward = np.sum(agent.rewards)
        policy_loss = agent.update(i_episode)
        last_100_ep_reward.insert(0, episode_reward)
        # printing stats
        log_interval = 2
        if i_episode % log_interval == 0:
            print('Episode {} \tstarting state: {:3d} \tLast length: {:5d}\tepisode reward: {:.3f} '
                  '\tpolicy loss: {:5f} \taverage episode reward: {: .3f}\t epsilon: {:.3f}'.format(
                   i_episode, state, ep_len, episode_reward, policy_loss, np.mean(last_100_ep_reward), agent.explore_factor))
        # saving stats
        loss_in_episodes.append(policy_loss)
        rewards_in_episodes.append(episode_reward)
        starting_states.append(state)
        episode_len.append(ep_len)

        if i_episode % 2000 == 0:
            agent.explore_factor *= 0.99

        # check if problem is solved according to spec
        if i_episode > 100:
            last_100_ep_reward.pop()
            if np.mean(last_100_ep_reward) > env.spec.reward_threshold:
                print("Solved! Running reward is now {} and "
                      "the last episode runs to {} time steps!".format(episode_reward, t))
                break

        if i_episode == num_train_episodes:  # enough training !!
            break

    base_dir = save_base_dir
    os.makedirs(base_dir, exist_ok=True)
    torch.save(agent.state_dict(), os.path.join(base_dir,
                                                r'PG_taxi_agent_HL{0}_trained{1}.pt'.format(HL_size, i_episode)))

    # saving stats
    base_dir = os.path.join(base_dir, 'stats')
    os.makedirs(base_dir, exist_ok=True)
    save_pickle(rewards_in_episodes, base_dir, 'rewards_HL{0}_trained{1}.p'.format(HL_size, i_episode))
    save_pickle(loss_in_episodes, base_dir, 'losses_HL{0}_trained{1}.p'.format(HL_size, i_episode))
    save_pickle(starting_states, base_dir, 'states_HL{0}_trained{1}.p'.format(HL_size, i_episode))
    save_pickle(episode_len, base_dir, 'episode_lens_HL{0}_trained{1}.p'.format(HL_size, i_episode))

    raw_rewards = eval_agent(agent, env, num_eval_eps, render=render)
    save_pickle(raw_rewards, base_dir, 'eval_rewards_HL{0}_trained{1}.p'.format(HL_size, i_episode))

    return raw_rewards


if __name__ == '__main__':

    default_dir = os.path.join(os.getcwd(), 'taxi_agent_PG')
    default_agent = os.path.join(os.getcwd(), 'taxi_agent_PG', 'PG_taxi_agent_HL128_trained100000.pt')

    parser = argparse.ArgumentParser(description="train and play a Policy Gradient Taxi-v2 agent")
    # modes
    parser.add_argument("-t", "--train", help="train a new agent (if not given then we'll load the default agent"
                                              " and evaluate it "
                                              "note: give -r to see screen rendering )",
                        action="store_true")
    # arguments
    # for loading
    parser.add_argument("-path", "--agent_path", type=str, nargs='?', default=default_agent,
                        help="path to a saved pre-trained PGAgent "
                             "(default agent is taxi_agent_PG\PG_taxi_agent_HL128_trained100000.pt)")

    # for training
    parser.add_argument("-save_dir", "--save_base_dir", type=str, nargs='?', default=default_dir,
                        help="where the agent and training stats will be saved (default is in taxi_agent_PG)")

    parser.add_argument("-eps_train", "--num_train_episodes", type=int, nargs='?', default=100000,
                        help="number of training episodes (default: 100000)")

    # for both
    parser.add_argument("-eps_eval", "--num_eval_eps", type=int, nargs='?', default=3,
                        help="number of evaluation episodes (default: 3)")

    parser.add_argument("-HL", "--HL_size", type=int, nargs='?', default=128,
                        help="size of the hidden layer (default: 128)"
                             "note: if loading an agent make sure to give the same HL_size as the saved agent")

    parser.add_argument("-r", "--render", action="store_true",
                        help="if to render evaluation episodes on console screen")

    args = parser.parse_args()

    # (load, agent_path, HL_size, save_base_dir='.', num_train_episodes=50000, num_eval_eps=3, render=True)
    main(not args.train, args.agent_path, args.HL_size, args.save_base_dir, args.num_train_episodes, args.num_eval_eps,
         args.render)
