'''
    This file implements the DQN for the Deep Q-Learning algorithm.
    The DQN has 3 different architectures based on the mission at hand.
    Toy Text - Regular DNN with FC layers, where we have one hidden layer.
    Classic Control - CNN for Control problems
    Atari - CNN for Atari games.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class DQN_DNN(nn.Module):
    '''
    Initialize the DQN network with a DNN architecture.
    Parmaeters:
        n_features: number of features of the states (int)
        n_hidden: number of hidden units in the hidden layer (int)
        n_actions: number of possible actions in the environment, the number of Q-values (int)
        use_batch_norm: use batch normalization agter the first layer (bool)
        use_dropout: add dropout regulariztion (bool)
        dropout_rate: probability of a neuron to be dropped during training (float)
    '''
    def __init__(self, n_features ,n_hidden, n_actions, use_batch_norm=False, use_dropout=False, dropout_rate=0.5):
        super(DQN_DNN, self).__init__()
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate

        if (self.use_batch_norm):
            fc1_layers = OrderedDict([('fc1', nn.Linear(n_features,n_hidden)),
               ('bn1', nn.BatchNorm1d(n_hidden)),  
               ('relu1', nn.ReLU())])
            self.fc1 = nn.Sequential(fc1_layers)
        else:
            fc1_layers = OrderedDict([('fc1', nn.Linear(n_features,n_hidden)),  
               ('relu1', nn.ReLU())])
            self.fc1 = nn.Sequential(fc1_layers)

        self.fc2 = nn.Linear(n_hidden, n_actions)

        if (self.use_dropout):
            # putting the model in model.eval() will disable dropout
            self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, x):
        '''
        Must be defined!
        Passes X through the layers, returns the output
        Linear: y=xA^T + b (applies linear tranformation)
        '''
        if (self.use_dropout):
            x = self.dropout(self.fc1(x))
        else:
            x = self.fc1(x)
        x = self.fc2(x.reshape(x.shape[0], -1))
        return x

class DQN_CNN(nn.Module):
    def __init__(self, n_actions, in_channels=4, use_batch_norm=False, use_dropout=False, dropout_rate=0.5, mode='control'):
        '''
        Parmaeters:
            in_channels: number of input channels (int)
                e.g. The number of most recent frames stacked together as describe in the paper: 
                    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
            n_actions: number of possible actions in the environment, the number of Q-values (int)
            use_batch_norm: use batch normalization agter the first layer (bool)
            use_dropout: add dropout regulariztion (bool)
            dropout_rate: probability of a neuron to be dropped during training (float)
            mode: the architecture is determined according to the task at hand (str)
                'control' - Acrobot-v1, stacked frames / difference of frames as state
                'atari' - Atari games, stacked frames based
        '''
        super(DQN_CNN, self).__init__()
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self.mode = mode
        # Conv layers
        # Dimensions formula:
        # Hout = floor[(Hin + 2*padding[0] - dialation[0] * (kernel_size[0] -
        # 1) - 1)/(stride[0]) + 1]
        # Wout = floor[(Win + 2*padding[1] - dialation[1] * (kernel_size[1] -
        # 1) - 1)/(stride[1]) + 1]
        if self.mode == 'control':
            if self.use_batch_norm:
                conv1_layers = OrderedDict([('conv1', nn.Conv2d(in_channels, 16, kernel_size=5, stride=2)),
                    ('bn1', nn.BatchNorm2d(16)),  
                    ('relu1', nn.ReLU())])
                self.conv1 = nn.Sequential(conv1_layers)

                conv2_layers = OrderedDict([('conv2', nn.Conv2d(16, 32, kernel_size=5, stride=2)),
                    ('bn2', nn.BatchNorm2d(32)),  
                    ('relu2', nn.ReLU())])
                self.conv2 = nn.Sequential(conv2_layers)

                conv3_layers = OrderedDict([('conv3', nn.Conv2d(32, 32, kernel_size=5, stride=2)),
                    ('bn3', nn.BatchNorm2d(32)),  
                    ('relu3', nn.ReLU())])
                self.conv3 = nn.Sequential(conv3_layers)

                self.fc1 = nn.Linear(32 * 7 * 7, n_actions)
            else:
                conv1_layers = OrderedDict([('conv1', nn.Conv2d(in_channels, 16, kernel_size=5, stride=2)), 
                    ('relu1', nn.ReLU())])
                self.conv1 = nn.Sequential(conv1_layers)

                conv2_layers = OrderedDict([('conv2', nn.Conv2d(16, 32, kernel_size=5, stride=2)), 
                    ('relu2', nn.ReLU())])
                self.conv2 = nn.Sequential(conv2_layers)

                conv3_layers = OrderedDict([('conv3', nn.Conv2d(32, 32, kernel_size=5, stride=2)),
                    ('relu3', nn.ReLU())])
                self.conv3 = nn.Sequential(conv3_layers)

                self.fc1 = nn.Linear(32 * 7 * 7, n_actions)
        else: # mode == 'atari'
            if self.use_batch_norm:
                conv1_layers = OrderedDict([('conv1', nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)),
                    ('bn1', nn.BatchNorm2d(32)),  
                    ('relu1', nn.ReLU())])
                self.conv1 = nn.Sequential(conv1_layers)

                conv2_layers = OrderedDict([('conv2', nn.Conv2d(32, 64, kernel_size=4, stride=2)),
                    ('bn2', nn.BatchNorm2d(64)),  
                    ('relu2', nn.ReLU())])
                self.conv2 = nn.Sequential(conv2_layers)

                conv3_layers = OrderedDict([('conv3', nn.Conv2d(64, 64, kernel_size=3, stride=1)),
                    ('bn3', nn.BatchNorm2d(64)),  
                    ('relu3', nn.ReLU())])
                self.conv3 = nn.Sequential(conv3_layers)

                fc1_layers = OrderedDict([('fc1', nn.Linear(64 * 7 * 7, 512)),
                                          ('bn4', nn.BatchNorm1d(512)),
                                          ('relu4', nn.ReLU())])
                self.fc1 = nn.Sequential(fc1_layers)
                self.fc2 = nn.Linear(512, n_actions)
            else:
                conv1_layers = OrderedDict([('conv1', nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)),
                    ('relu1', nn.ReLU())])
                self.conv1 = nn.Sequential(conv1_layers)

                conv2_layers = OrderedDict([('conv2', nn.Conv2d(32, 64, kernel_size=4, stride=2)), 
                    ('relu2', nn.ReLU())])
                self.conv2 = nn.Sequential(conv2_layers)

                conv3_layers = OrderedDict([('conv3', nn.Conv2d(64, 64, kernel_size=3, stride=1)),
                    ('relu3', nn.ReLU())])
                self.conv3 = nn.Sequential(conv3_layers)

                fc1_layers = OrderedDict([('fc1', nn.Linear(64 * 7 * 7, 512)),
                    ('relu4', nn.ReLU())])
                self.fc1 = nn.Sequential(fc1_layers)
                self.fc2 = nn.Linear(512, n_actions)

        if self.use_dropout:
            self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, x):
        '''
        Forward pass.
        '''
        if self.mode == 'control':
            if self.use_dropout:
                x = self.conv1(x)
                x = self.dropout(x)
                x = self.conv2(x)
                x = self.conv3(x)
                x = self.fc1(x.reshape(x.shape[0], -1))
                return x
            else:
                x = self.conv1(x)
                x = self.conv2(x)
                x = self.conv3(x)
                x = self.fc1(x.reshape(x.shape[0], -1))
                return x
        else:
            # mode == 'atari'
            if self.use_dropout:
                x = self.conv1(x)
                x = self.dropout(x)
                x = self.conv2(x)
                x = self.conv3(x)
                x = self.fc1(x.reshape(x.shape[0], -1))
                x = self.fc2(x)
                return x
            else:
                x = self.conv1(x)
                x = self.conv2(x)
                x = self.conv3(x)
                x = self.fc1(x.reshape(x.shape[0], -1))
                x = self.fc2(x)
                return x