'''
This file holds scheduling classes for the epsilon-greedy exploration strategy
'''

# imports:
import numpy as np

class ExponentialSchedule():
    '''
    Exponential scheduling strategy.
    eps(t) = end_val + (start_val - end_val) * exp(-1.0 * t / decay_rate)
    Parameters:
        start_val: initial value (float)
        end_val: final value (float)
        decay_rate: rate of exponential decaying (int), usually steps or episodes
    '''
    def __init__(self, start_val = 1.0, end_val = 0.05, decay_rate = 200):
        self.start = start_val
        self.end = end_val
        self.decay = decay_rate

    def value(self, t):
        '''
        Calculates the current value at time t
        Parameters:
            t: current time/step/episode (int)
        '''
        return (self.end + (self.start - self.end) * np.exp(-1.0 * t / self.decay))

class LinearSchedule():
    def __init__(self, start_val = 1.0, end_val = 0.05, total_timesteps = 200):
        '''
        Linear scheduling strategy.
        eps(t) = start_val + (end_val - start_val) * min(t / total_time_steps, 1)
        Parameters:
            start_val: initial value (float)
            end_val: final value (float)
            total_timesteps: time steps to get from start to end (until saturation)
        '''

        self.start = start_val
        self.end = end_val
        self.total_timesteps = total_timesteps

    def value(self, t):
        '''
        Calculates the current value at time t
        Parameters:
            t: current time/step/episode (int)
        '''
        return (self.start + (self.end - self.start) * min(1.0 * t / self.total_timesteps, 1.0))