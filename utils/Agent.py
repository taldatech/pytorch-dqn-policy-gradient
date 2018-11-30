'''
This file implements agents for different tasks.
Author: Tal Daniel
'''
# imports
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random
import gym.spaces
import os

from utils.OneHotGenerator import OneHotGenerator
from utils.DQN_model import DQN_DNN, DQN_CNN
from utils.ReplayBuffer import ReplayBuffer
from utils.Schedule import ExponentialSchedule, LinearSchedule
from PIL import Image

class TaxiAgent():
    '''
    This class implements a DQN agent for OpenAi's Taxi-v2 environment (https://gym.openai.com/envs/Taxi-v2/)
    Environment details:
         Observations:
                There are 500 discrete states since there are 25 taxi positions, 5 possible locations of the passenger
                (including the case when the passenger is the taxi), and 4 destination locations.
         Actions:
            There are 6 discrete deterministic actions:
            - 0: move south
            - 1: move north
            - 2: move east
            - 3: move west
            - 4: pickup passenger
            - 5: dropoff passenger
        Rewards:
            There is a reward of -1 for each action and an additional reward of +20 for delievering the passenger.
            There is a reward of -10 for executing actions "pickup" and "dropoff" illegally.
        Rendering:
            - blue: passenger
            - magenta: destination
            - yellow: empty taxi
            - green: full taxi
            - other letters: locations
    '''
    def __init__(
        self,
        env,
        name='',
        n_hidden=150,
        optimizer='RMSprop',
        momentum=0.9,
        loss='MSE',
        exploration=None,
        use_l1_regularizer=False,
        l1_lambda=1.0,
        replay_buffer_size=100000,
        gamma=0.99,
        learning_rate=0.0003,
        steps_to_start_learn=50000,
        target_update_freq=10000,
        obs_represent='one-hot',
        clip_grads=False,
        use_batch_norm=False,
        use_dropout=False,
        dropout_rate=0.5
        ):
        '''
        Initialize a Deep Q-learning algorithm agent for Taxi-v2 environment.

        Parameters:
            env: gym environment to train on (gym.Env)
            name: name of the model, used for saving and loading the model (str)
            n_hidden: number of hidden neurons in the hidden layer (int)
            optimizer: Optimizer for the DQN (str)
                options: 'RMSprop', 'Adam', 'Nesterov'
            momentum: momentum parameter for the optimizer,
                        used for 'Nesterov' (float)
            loss: Loss function for the DQN (str)
                options: 'MSE', 'SmoothL1'
            exploration: Exploration scheduling strategy (Schedule)
            regularizer: Add regularization to the loss (str)
                options: 'l1'
            l1_lambda: Lambda parameter for the regularization (float)
            replay_buffer_size: Number of memories to store in the replay buffer (int)
            gamma: Discount Factor (float)
            learning_rate: Learning rate for the optimizer (float)
            steps_to_start_learn: Number of environment steps from which to start replaying experiences (int)
            target_update_freq: Number of experience replay rounds (not steps!) to perform between
                each update to the target Q network (int)
            clip_grads: Whether or not the gradients should be clipped (bool)
            obs_represent: Representation of the states for the Taxi agent (str)
                options: 'one-hot' (one-hot-vector), 'locations' (a 4-tuple (taxiRow, taxiCol, passLoc, destIdx)),
                            'state-int' (one integer), 'location-one-hot' (concatenation of one-hots)
            use_batch_norm: Use Batch Normalizaton between DQN layers (bool)
            use_dropout: Use Dropout Regularization on the first layer (bool)
            dropout_rate: Probability to drop each neurin in the first layer if `use_dropout` is True
        '''

        assert type(env.observation_space) == gym.spaces.Discrete
        assert type(env.action_space) == gym.spaces.Discrete
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Hyper-params
        self.name = name
        self.gamma = gamma
        self.n_hidden = n_hidden
        self.n_actions = env.action_space.n
        self.n_obs = env.observation_space.n
        self.replay_buffer_size = replay_buffer_size
        self.replay_buffer = ReplayBuffer(replay_buffer_size, 1)
        self.steps_to_start_learn = steps_to_start_learn
        self.target_update_freq = target_update_freq
        self.clip_grads=clip_grads
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self.use_l1_regularizer = use_l1_regularizer
        self.l1_lambda = l1_lambda
        self.obs_represent = obs_represent
        if (self.obs_represent == 'one-hot'):
            print("Using One-Hot-Vector representation for states")
            self.one_hot_generator = OneHotGenerator(self.n_obs)
        elif self.obs_represent == 'state-int':
            self.n_obs = 1
            print("Using State-Integer representation for states")
        elif self.obs_represent == 'location-one-hot':
            print("Using Location-One-Hots representation for states")
            self.n_obs = 19 # 5 bits for row, col, passenger location and 4 for destIdx
        else:
            self.n_obs = len(list(env.env.decode(0))) # get state of the game as a tuple
            print("Using Locations-Tuple representation for states")
        if loss == 'SmoothL1':
            self.loss_criterion = nn.SmoothL1Loss()
        else:
            self.loss_criterion = nn.MSELoss()
        self.learning_rate = learning_rate
        if exploration is None:
            self.explore_schedule = ExponentialSchedule()
        else:
            self.explore_schedule = exploration

        self.epsilon = self.explore_schedule.value(0)

        # Initialize DQN's and optimizer
        self.Q_train = DQN_DNN(
            self.n_obs,
            self.n_hidden,
            self.n_actions,
            self.use_batch_norm,
            self.use_dropout,
            self.dropout_rate).to(self.device)

        self.Q_target = DQN_DNN(
            self.n_obs,
            self.n_hidden,
            self.n_actions,
            self.use_batch_norm,
            self.use_dropout,
            self.dropout_rate).to(self.device)

        # set modes
        self.Q_train.train()
        self.Q_target.eval()

        # optimizer
        self.momentum = momentum
        if optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.Q_train.parameters(), lr=self.learning_rate)
        elif optimizer == 'Nesterov':
            self.optimizer = torch.optim.SGD(self.Q_train.parameters(), lr=self.learning_rate, momentum=self.momentum, nesterov=True)
        else:
            self.optimizer = torch.optim.RMSprop(self.Q_train.parameters(), lr=self.learning_rate)

        # statistics
        self.steps_count = 0
        self.episodes_seen = 0
        self.num_param_updates = 0

        # load checkpoint if it exists
        self.load_agent_state()

        # init target network with the same weights
        self.Q_target.load_state_dict(self.Q_train.state_dict())

        print("Created Agent for Taxi-v2")

    def select_greedy_action(self, obs):
        '''
        This method picks an action to perform according to an epsilon-greedy policy.
        Parameters:
            obs: current state or observation from the environment (int or tuple)
                for One-Hot the state is a number (out of 500)
                else the state is a tuple (taxiRow, taxiCol, passLoc, destIdx)
        Returns:
            action (int)
        '''
        self.epsilon = self.explore_schedule.value(self.episodes_seen)
        threshold = self.epsilon
        rand_num = random.random()
        if (rand_num > threshold):
            # Pick according to current Q-values
            if self.obs_represent == 'one-hot' or self.obs_represent == 'location-one-hot':
                if type(obs) is int:
                    obs = self.one_hot_generator.to_one_hot(obs)
            elif self.obs_represent == 'state-int':
                obs = obs / 499.0
            else:
                assert len(obs) == 4
                obs = obs / 4.0 # normalize
            # make sure the target network is in eval mode (no gradient calculation)
            obs = torch.from_numpy(obs).unsqueeze(0).type(torch.FloatTensor).to(self.device)
            self.Q_target.eval()
            with torch.no_grad():
                q_actions = self.Q_target(obs)
                action = torch.argmax(q_actions).item()
        else:
            action = np.random.choice(self.n_actions)
        return action

    def learn(self, batch_size):
        '''
        This method performs a training step for the agent.
        Parmeters:
            batch_size: number of samples to perform the training step on (int)
        '''
        if (self.steps_count > self.steps_to_start_learn and
            self.replay_buffer.can_sample(batch_size)):
            # Use the replay buffer to sample a batch of transitions
            # Note: done_mask[i] is 1 if the next state corresponds to the end of an episode,
            # in which case there is no Q-value at the next state; at the end of an
            # episode, only the current state reward contributes to the target
            obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = self.replay_buffer.sample(batch_size)
            obs_batch = torch.from_numpy(obs_batch).type(torch.FloatTensor).to(self.device)
            act_batch = torch.from_numpy(act_batch).long().to(self.device)
            rew_batch = torch.from_numpy(rew_batch).to(self.device)
            next_obs_batch = torch.from_numpy(next_obs_batch).type(torch.FloatTensor).to(self.device)
            if self.obs_represent == 'locations':
                obs_batch = obs_batch / 4.0 # normalize
                next_obs_batch = next_obs_batch / 4.0 # noramalize
            elif self.obs_represent == 'state-int':
                obs_batch = obs_batch / 499.0 # normalize
                next_obs_batch = next_obs_batch / 499.0 # noramalize
            not_done_mask = torch.from_numpy(1 - done_mask).to(self.device)
            # Compute current Q value, q_func takes only state and output value for every state-action pair
            # We choose Q based on action taken.
            current_Q_values = self.Q_train(obs_batch.type(torch.FloatTensor).to(self.device)).gather(1, act_batch.unsqueeze(1))
            # Compute next Q value based on which action gives max Q values
            # Detach variable from the current graph since we don't want gradients for next Q to propagated
            next_max_q = self.Q_target(next_obs_batch.type(torch.FloatTensor).to(self.device)).detach().max(1)[0]
            next_Q_values = not_done_mask * next_max_q
            # Compute the target of the current Q values
            target_Q_values = rew_batch + (self.gamma * next_Q_values)
            # loss
            loss = self.loss_criterion(current_Q_values, target_Q_values.unsqueeze(1))
            # add regularozation
            if self.use_l1_regularizer:
                lam = torch.tensor(self.l1_lambda)
                l1_reg = torch.tensor(0.)
                for param in self.Q_train.parameters():
                    l1_reg += torch.norm(param, 1)
                loss += lam * l1_reg
            # optimize model
            self.optimizer.zero_grad()
            loss.backward()
            if (self.clip_grads):
                for param in self.Q_train.parameters():
                    param.grad.data.clamp_(-1, 1)
            self.optimizer.step()
            self.num_param_updates += 1
            # copy weights to target network and save network state
            if self.num_param_updates % self.target_update_freq == 0:
                self.Q_target.load_state_dict(self.Q_train.state_dict())
                # save network state
                self.save_agent_state()

    def predict_action(self, obs):
        '''
        Predict action for inference or playing.
        Parameters:
            obs: a state observation from the environment (int/tuple/np.array)
        Returns:
            action: action for which the Q-value of the current observation is the highest (int)
        '''
        with torch.no_grad():
            obs = torch.from_numpy(obs).unsqueeze(0).type(torch.FloatTensor).to(self.device)
            q_actions = self.Q_target(obs)
            action = torch.argmax(q_actions).item()
        return action

    def save_agent_state(self):
        '''
        This function saves the current state of the DQN (the weights) to a local file.
        '''
        filename = "taxi_agent_" + self.name + ".pth"
        dir_name = './taxi_agent_ckpt'
        full_path = os.path.join(dir_name, filename)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        torch.save({
            'model_state_dict': self.Q_train.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_count': self.steps_count,
            'episodes_seen': self.episodes_seen,
            'epsilon': self.epsilon,
            'num_param_updates': self.num_param_updates
            }, full_path)
        print("Saved Taxi Agent checkpoint @ ", full_path)

    def load_agent_state(self, path=None, copy_to_target_network=False, load_optimizer=True):
        '''
        This function loads an agent checkpoint.
        Parameters:
            path: path to a checkpoint, e.g `/path/to/dir/ckpt.pth` (str)
            copy_to_target_network: whether or not to copy the loaded training
                DQN parameters to the target DQN, for manual loading (bool)
            load_optimizer: whether or not to restore the optimizer state
        '''
        if path is None:
            filename = "taxi_agent_" + self.name + ".pth"
            dir_name = './taxi_agent_ckpt'
            full_path = os.path.join(dir_name, filename)
        else:
            full_path = path
        exists = os.path.isfile(full_path)
        if exists:
            if not torch.cuda.is_available():
                checkpoint = torch.load(full_path, map_location='cpu')
            else:
                checkpoint = torch.load(full_path)
            self.Q_train.load_state_dict(checkpoint['model_state_dict'])
            if load_optimizer:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.steps_count = checkpoint['steps_count']
            self.episodes_seen = checkpoint['episodes_seen']
            self.epsilon = checkpoint['epsilon']
            self.num_param_update = checkpoint['num_param_updates']
            print("Checkpoint loaded successfully from ", full_path)
            # for manual loading a checkpoint
            if copy_to_target_network:
                self.Q_target.load_state_dict(self.Q_train.state_dict())
        else:
            print("No checkpoint found...")

def preprocess_frame(env, mode='atari', render=False):
    '''
    This function preprocess the current frame of the environment.
    Prameters:
        env: the environment (gym.Env)
        mode: processing mode to use (str)
            options: 'atari' - 1 channel, 'control' - 3 channels
        render: whetheState-Integerr or not to render the screen, which opens a window (bool)
            * in ClassicControl problems, even when setting render mode to 'rgb_array',
                a window is opened. Setting this to False will close this window each time.
            * Performance is better when set to True, less overhead.
    Returns:
        frame: the processed (reshaped, scaled, adjusted) frame (np.array, np.uint8)
    '''
    screen = env.render(mode='rgb_array')
    if not render:
        env.close() # on Windows, must close the opened window
    if mode =='atari':
        screen = np.reshape(screen, [500, 500, 3]).astype(np.float32)
        screen = screen[:, :, 0] * 0.299 + screen[:, :, 1] * 0.587 + screen[:, :, 2] * 0.114 # dimension reduction, contrast
        screen = Image.fromarray(screen)
        resized_screen = screen.resize((84, 84), Image.BILINEAR)
        resized_screen = np.array(resized_screen)
        x_t = np.reshape(resized_screen, [84, 84, 1])
    else:
        screen = np.reshape(screen, [500, 500, 3]).astype(np.uint8)
        screen = Image.fromarray(screen)
        resized_screen = screen.resize((84, 84), Image.BILINEAR)
        resized_screen = np.array(resized_screen)
        x_t = np.reshape(resized_screen, [84, 84, 3])
    return x_t.astype(np.uint8)

class AcrobotAgent():
    '''
    This class implements a DQN agent for OpenAi's Acrobot-v1 environment (https://gym.openai.com/envs/Acrobot-v1/)
    Environment details:
         Observations:
                The state consists of the sin() and cos() of the two rotational joint
                angles and the joint angular velocities :
                [cos(theta1) sin(theta1) cos(theta2) sin(theta2) thetaDot1 thetaDot2].
                For the first link, an angle of 0 corresponds to the link pointing downwards.
                The angle of the second link is relative to the angle of the first link.
                An angle of 0 corresponds to having the same angle between the two links.
                A state of [1, 0, 1, 0, ..., ...] means that both links point downwards
         Actions:
             The action is either applying +1, 0 or -1 torque on the joint between
             the two pendulum links.
            There are 3 discrete deterministic actions:
            - 0: -1
            - 1: 0
            - 2: +1
        Rewards:
            There is a reward of -1 for each action and 0 on terminal state
        Rendering:
            two modes: 'human', 'rgb_array'
    '''
    def __init__(
        self,
        env,
        name='',
        optimizer='RMSprop',
        momentum=0.9,
        loss='SmoothL1',
        exploration=None,
        use_l1_regularizer=False,
        l1_lambda=1.0,
        replay_buffer_size=500000,
        frame_history_len=4,
        learning_freq=4,
        gamma=0.99,
        learning_rate=0.0003,
        steps_to_start_learn=50000,
        target_update_freq=10000,
        obs_represent='frame_seq',
        clip_grads=False,
        use_batch_norm=False,
        use_dropout=False,
        dropout_rate=0.5
        ):
        '''
        Initialize a Deep Q-learning algorithm agent for Acrobot-v1 environment.

        Parameters:
            env: gym environment to train on (gym.Env)
            name: name of the model, used for saving and loading the model (str)
            optimizer: Optimizer for the DQN (str)
                options: 'RMSprop', 'Adam', 'Nesterov'
            momentum: momentum parameter for the optimizer,
                        used for 'Nesterov' (float)
            loss: Loss function for the DQN (str)
                options: 'MSE', 'SmoothL1'
            exploration: Exploration scheduling strategy (Schedule)
            regularizer: Add regularization to the loss (str)
                options: 'l1'
            l1_lambda: Lambda parameter for the regularization (float)
            replay_buffer_size: Number of memories to store in the replay buffer (int)
            frame_history_len: How many frames represent a state (int)
            learning_freq: How many steps to take between performing a learning steps (int)
            gamma: Discount Factor (float)
            learning_rate: Learning rate for the optimizer (float)
            steps_to_start_learn: Number of environment steps from which to start replaying experiences (int)
            target_update_freq: Number of experience replay rounds (not steps!) to perform between
                each update to the target Q network (int)
            clip_grads: Whether or not the gradients should be clipped (bool)
            obs_represent: Representation of the states for the Acrobot agent (str)
                options: 'frame_seq' (a sequence of rgb frames represent an observation),
                            'frame_diff' (one rgb frame)
            use_batch_norm: Use Batch Normalizaton between DQN layers (bool)
            use_dropout: Use Dropout Regularization on the first layer (bool)
            dropout_rate: Probability to drop each neurin in the first layer if `use_dropout` is True
        '''

        assert type(env.observation_space) == gym.spaces.Box
        assert type(env.action_space) == gym.spaces.Discrete
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Hyper-params
        self.name = name
        self.gamma = gamma
        self.n_actions = env.action_space.n
        self.replay_buffer_size = replay_buffer_size
        self.frame_history_len = frame_history_len
        self.learning_freq = learning_freq
        self.steps_to_start_learn = steps_to_start_learn
        self.target_update_freq = target_update_freq
        self.clip_grads=clip_grads
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self.use_l1_regularizer = use_l1_regularizer
        self.l1_lambda = l1_lambda

        if loss == 'MSE':
            self.loss_criterion = nn.MSELoss()
        else:
            self.loss_criterion = nn.SmoothL1Loss()

        self.learning_rate = learning_rate
        if exploration is None:
            self.explore_schedule = ExponentialSchedule()
        else:
            self.explore_schedule = exploration

        self.epsilon = self.explore_schedule.value(0)
        # Different modes
        # get observation shape
        env.reset()
        self.obs_represent = obs_represent
        if self.obs_represent == 'frame_seq':
            sample_frame = preprocess_frame(env, mode='atari')
            img_h, img_w, img_c = sample_frame.shape
            self.n_obs = frame_history_len * img_c
            self.replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)
            print("Using Frame-Sequence representation for states")

            # Initialize DQN's and optimizer
            self.Q_train = DQN_CNN(
                self.n_actions,
                self.n_obs,
                self.use_batch_norm,
                self.use_dropout,
                self.dropout_rate,
                mode='atari').to(self.device)

            self.Q_target = DQN_CNN(
                self.n_actions,
                self.n_obs,
                self.use_batch_norm,
                self.use_dropout,
                self.dropout_rate,
                mode='atari').to(self.device)
        else:
            # 'frame_diff'
            sample_frame = preprocess_frame(env, mode='control')
            img_h, img_w, img_c = sample_frame.shape
            self.n_obs = img_c
            self.replay_buffer = ReplayBuffer(replay_buffer_size, 1)
            print("Using Frame-Difference representation for states")

            # Initialize DQN's and optimizer
            self.Q_train = DQN_CNN(
                self.n_actions,
                self.n_obs,
                self.use_batch_norm,
                self.use_dropout,
                self.dropout_rate,
                mode='control').to(self.device)

            self.Q_target = DQN_CNN(
                self.n_actions,
                self.n_obs,
                self.use_batch_norm,
                self.use_dropout,
                self.dropout_rate,
                mode='control').to(self.device)

        # set modes
        self.Q_train.train()
        self.Q_target.eval()

        # optimizer
        self.momentum = momentum
        if optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.Q_train.parameters(), lr=self.learning_rate)
        elif optimizer == 'Nesterov':
            self.optimizer = torch.optim.SGD(self.Q_train.parameters(), lr=self.learning_rate, momentum=self.momentum, nesterov=True)
        else:
            self.optimizer = torch.optim.RMSprop(self.Q_train.parameters(), lr=self.learning_rate)

        # statistics
        self.steps_count = 0
        self.episodes_seen = 0
        self.num_param_updates = 0

        # load checkpoint if it exists
        self.load_agent_state()

        # init target network with the same weights
        self.Q_target.load_state_dict(self.Q_train.state_dict())

        print("Created Agent for Acrobot-v1")

    def select_greedy_action(self, obs, use_episode=True):
        '''
        This method picks an action to perform according to an epsilon-greedy policy.
        Parameters:
            obs: current state or observation from the environment (np.array)
            use_episode: whether to use the episodes count as value for decay (bool)
        Returns:
            action (int)
        '''
        if use_episode:
            self.epsilon = self.explore_schedule.value(self.episodes_seen)
        else:
            self.epsilon = self.explore_schedule.value(self.steps_count)
        threshold = self.epsilon
        rand_num = random.random()
        if (rand_num > threshold):
            # make sure the target network is in eval mode (no gradient calculation)
            obs = torch.from_numpy(obs).unsqueeze(0).type(torch.FloatTensor).to(self.device) / 255.0
            self.Q_target.eval()
            with torch.no_grad():
                q_actions = self.Q_target(obs)
                action = torch.argmax(q_actions).item()
        else:
            action = np.random.choice(self.n_actions)
        return action

    def learn(self, batch_size):
        '''
        This method performs a training step for the agent.
        Parmeters:
            batch_size: number of samples to perform the training step on (int)
        '''
        if (self.steps_count > self.steps_to_start_learn and
            self.steps_count % self.learning_freq == 0 and
            self.replay_buffer.can_sample(batch_size)):
            # Use the replay buffer to sample a batch of transitions
            # Note: done_mask[i] is 1 if the next state corresponds to the end of an episode,
            # in which case there is no Q-value at the next state; at the end of an
            # episode, only the current state reward contributes to the target
            obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = self.replay_buffer.sample(batch_size)
            obs_batch = torch.from_numpy(obs_batch).type(torch.FloatTensor).to(self.device) / 255.0
            act_batch = torch.from_numpy(act_batch).long().to(self.device)
            rew_batch = torch.from_numpy(rew_batch).to(self.device)
            next_obs_batch = torch.from_numpy(next_obs_batch).type(torch.FloatTensor).to(self.device) / 255.0
            not_done_mask = torch.from_numpy(1 - done_mask).to(self.device)
            # Compute current Q value, q_func takes only state and output value for every state-action pair
            # We choose Q based on action taken.
            current_Q_values = self.Q_train(obs_batch.type(torch.FloatTensor).to(self.device)).gather(1, act_batch.unsqueeze(1))
            # Compute next Q value based on which action gives max Q values
            # Detach variable from the current graph since we don't want gradients for next Q to propagated
            next_max_q = self.Q_target(next_obs_batch.type(torch.FloatTensor).to(self.device)).detach().max(1)[0]
            next_Q_values = not_done_mask * next_max_q
            # Compute the target of the current Q values
            target_Q_values = rew_batch + (self.gamma * next_Q_values)
            # loss
            loss = self.loss_criterion(current_Q_values, target_Q_values.unsqueeze(1))
            # add regularozation
            if self.use_l1_regularizer:
                lam = torch.tensor(self.l1_lambda)
                l1_reg = torch.tensor(0.)
                for param in self.Q_train.parameters():
                    l1_reg += torch.norm(param, 1)
                loss += lam * l1_reg
            # optimize model
            self.optimizer.zero_grad()
            loss.backward()
            if (self.clip_grads):
                for param in self.Q_train.parameters():
                    param.grad.data.clamp_(-1, 1)
            self.optimizer.step()
            self.num_param_updates += 1
            # copy weights to target network and save network state
            if self.num_param_updates % self.target_update_freq == 0:
                self.Q_target.load_state_dict(self.Q_train.state_dict())
                # save network state
                self.save_agent_state()

    def predict_action(self, obs):
        '''
        Predict action for inference or playing.
        Parameters:
            obs: a state observation from the environment (int/tuple/np.array)
        Returns:
            action: action for which the Q-value of the current observation is the highest (int)
        '''
        if (self.obs_represent == 'frame_diff'):
            obs = obs.transpose(2, 0, 1)
        obs = torch.from_numpy(obs).unsqueeze(0).type(torch.FloatTensor).to(self.device) / 255.0
        self.Q_target.eval()
        with torch.no_grad():
            q_actions = self.Q_target(obs)
            action = torch.argmax(q_actions).item()
        return action

    def save_agent_state(self):
        '''
        This function saves the current state of the DQN (the weighif not torch.cuda.is_available():
                checkpoint = torch.load(full_path, map_location='cpu')
            else:
                checkpoint = torch.load(full_path)ts) to a local file.
        '''
        filename = "acrobot_agent_" + self.name + ".pth"
        dir_name = './acrobot_agent_ckpt'
        full_path = os.path.join(dir_name, filename)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        torch.save({
            'model_state_dict': self.Q_train.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_count': self.steps_count,
            'episodes_seen': self.episodes_seen,
            'epsilon': self.epsilon,
            'num_param_updates': self.num_param_updates
            }, full_path)
        print("Saved Acrobot Agent checkpoint @ ", full_path)

    def load_agent_state(self, path=None, copy_to_target_network=False, load_optimizer=True):
        '''
        This function loads an agent checkpoint.
        Parameters:
            path: path to a checkpoint, e.g `/path/to/dir/ckpt.pth` (str)
            copy_to_target_network: whether or not to copy the loaded training
                DQN parameters to the target DQN, for manual loading (bool)
            load_optimizer: whether or not to restore the optimizer state
        '''
        if path is None:
            filename = "acrobot_agent_" + self.name + ".pth"
            dir_name = './acrobot_agent_ckpt'
            full_path = os.path.join(dir_name, filename)
        else:
            full_path = path
        exists = os.path.isfile(full_path)
        if exists:
            if not torch.cuda.is_available():
                checkpoint = torch.load(full_path, map_location='cpu')
            else:
                checkpoint = torch.load(full_path)
            self.Q_train.load_state_dict(checkpoint['model_state_dict'])
            if load_optimizer:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.steps_count = checkpoint['steps_count']
            self.episodes_seen = checkpoint['episodes_seen']
            self.epsilon = checkpoint['epsilon']
            self.num_param_update = checkpoint['num_param_updates']
            print("Checkpoint loaded successfully from ", full_path)
            # for manual loading a checkpoint
            if copy_to_target_network:
                self.Q_target.load_state_dict(self.Q_train.state_dict())
        else:
            print("No checkpoint found...")
