'''
This file implements the replay buffer used for the learning step of the DQN.
It is inspired by DeepMind's replay buffer.
Original file @ https://github.com/berkeleydeeprlcourse/homework/tree/master/hw3
'''
# imports
import random
import numpy as np


class ReplayBuffer():
    def __init__(self, size, frame_history_len):
        '''
        Replay Buffer: Stores memories of the environment.
        The agent samples from this buffer in the learning process.
        Parameters:
            size: maximum number of transitions to store in the buffer. Memories are overridden on overflow.
            frame_history_len: number of frames per observation (some environemnts' states
                                are determined by more than one frame, e.g Pong).
        '''
        self.size = size
        self.frame_history_len = frame_history_len

        self.next_idx = 0
        self.current_buff_size = 0

        self.obs = None
        self.action = None
        self.reward = None
        self.done = None

    def can_sample(self, batch_size):
        '''
        Returns True if the buffer can supply `batch_size` of samples.
        '''
        return batch_size + 1 <= self.current_buff_size

    def encode_sample(self, indices):
        '''
        Encodes samples into batches.
        Parameters:
            indices: draws the samples at these positions
        '''
        obs_batch = np.concatenate([self.encode_observation(idx)[np.newaxis, :] for idx in indices], 0) # np.newaxis expands dimensions to allow batches
        action_batch = self.action[indices]
        reward_batch = self.reward[indices]
        next_obs_batch = np.concatenate([self.encode_observation(idx + 1)[np.newaxis, :] for idx in indices], 0)
        # Mask terminal states (1 if this is terminal state, 0 otherwise)
        done_mask = np.array([1.0 if self.done[idx] else 0.0 for idx in indices], dtype=np.float32)

        return obs_batch, action_batch, reward_batch, next_obs_batch, done_mask

    def generate_n_unique(self, n):
        '''
        Generates n uniuqe indices.
        Parameters:
            n: number of indices to generate (int)
        Returns:
            res: list of indices from the buffer
        '''
        res = []
        while len(res) < n:
            candidate = random.randint(0, self.current_buff_size - 2)
            if candidate not in res:
                res.append(candidate)
        return res


    def sample(self, batch_size):
        '''
        Sample `batch_size` different memories.
        memory[i] is a tuple (observations[i], actions[i], rewards[i], next_observations[i]) such that:
            observation[i]: the observed state of the environment
            actions[i]: the action taken as a result of being at observation[i]
            rewards[i]: the reward received from taking actions[i]
            next_observations[i]: the next observation as a result from taking actions[i]
                * unless the episode was done
        Parameters:
            batch_size: number of samples to draw from the buffer (int)
        Returns:
            obs_batch: np.array, (batch_size, img_channels * frame_history_len, img_h, img_w), dtype np.uint8
            action_batch: np.array, (batch_size,), dtype np.int32
            reward_batch: np.array, (batch_size,), dtype np.float32
            next_obs_batch: np.array, (batch_size, img_channels * frame_history_len, img_h, img_w), dtype np.uint8
        '''
        assert self.can_sample(batch_size)
        indices = self.generate_n_unique(batch_size)
        return self.encode_sample(indices)

    def encode_recent_observation(self):
        '''
        Returns the most recent `frame_history_len` frames.
        Returns:
            obs: np.array, (img_h, img_w, img_c * frame_history_len), dtype np.uint8
        '''
        assert self.current_buff_size > 0
        return self.encode_observation((self.next_idx - 1) % self.size)

    def encode_observation(self, idx):
        '''
        Encodes an observation at memory[idx].
        Parameters:
            idx: poistion in the buffer (int)
        Returns:
            for CNN: np.array, (img_c * frame_history_len, img_h, img_w), dtype np.uint8
            for DNN: np.array, (env.observations.n,), dtype np.float32
        '''
        end_idx   = idx + 1 # make noninclusive
        start_idx = end_idx - self.frame_history_len
        # this checks if we are using low-dimensional observations, such as one-hot
        # state, in which case we just directly return the latest obs.
        if len(self.obs.shape) <= 2:
            return self.obs[end_idx-1]
        # check boundries and overflow
        if start_idx < 0 and self.current_buff_size != self.size:
            start_idx = 0
        for idx in range(start_idx, end_idx - 1):
            # skip terminal states
            if self.done[idx % self.size]:
                start_idx = idx + 1
        missing_context = self.frame_history_len - (end_idx - start_idx)
        # if zero padding is needed for missing context
        # or we are on the boundry of the buffer
        if start_idx < 0 or missing_context > 0:
            frames = [np.zeros_like(self.obs[0]) for _ in range(missing_context)]
            for idx in range(start_idx, end_idx):
                frames.append(self.obs[idx % self.size])
            return np.concatenate(frames, 0)
        else:
            # this optimization has potential to saves about 30% compute time \o/ (!!)
            img_h, img_w = self.obs.shape[2], self.obs.shape[3]
            return self.obs[start_idx:end_idx].reshape(-1, img_h, img_w)

    def store_frame(self, frame):
        '''
        Store a single frame in the buffer at the next available index and overwrite
        old frames if necessary. Process the frame to comply with the DQN format.
        Parameters:
            frame: np.array, (img_h, img_w, img_c), dtype np.uint8
                    or int
        Returns:
            pos: Index of the position in the buffer for the stored frame (int)
        '''
        # check what type of states we work with
        if type(frame) is int:
            if self.obs is None:
                self.obs = np.empty([self.size], dtype=np.int32) # (buffer size, img_channels, img_h, img_w)
                self.action = np.empty([self.size], dtype=np.int32)
                self.reward = np.empty([self.size], dtype=np.float32)
                self.done = np.empty([self.size], dtype=np.bool)
        else:
            # make sure we are not using low-dimensional observations, such as one-hot
            if len(frame.shape) > 1:
                # transpose image frame into (img_c, img_h, img_w)
                frame = frame.transpose(2, 0, 1)

            # in case the buffer is empty:
            if self.obs is None:
                self.obs = np.empty([self.size] + list(frame.shape), dtype=np.uint8) # (buffer size, img_channels, img_h, img_w)
                self.action = np.empty([self.size], dtype=np.int32)
                self.reward = np.empty([self.size], dtype=np.float32)
                self.done = np.empty([self.size], dtype=np.bool)

        self.obs[self.next_idx] = frame

        pos = self.next_idx
        self.next_idx = (self.next_idx + 1) % self.size
        self.current_buff_size = min(self.size, self.current_buff_size + 1)

        return pos

    def store_effect(self, idx, action, reward, done):
        '''
        Store the effects of the action taken after obeserving frame stored
        at index idx.
        `store_frame` MUST be called before this.
        Note:`store_frame` and `store_effect` are broken into two functions 
        so that one can call `encode_recent_observation` in between.
        Parameters:
            idx: Index in the buffer of the corresponding observation (int)
            action: The action that was taken (int)
            reward: The corresponding reward from taking `action` (float)
            done: True if the episode was finished after taking `action` (bool)
        '''
        self.action[idx] = action
        self.reward[idx] = reward
        self.done[idx]   = done