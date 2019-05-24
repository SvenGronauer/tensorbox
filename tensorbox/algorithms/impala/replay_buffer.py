import numpy as np
from collections import namedtuple

Trajectory = namedtuple('Trajectory', ['obs', 'actions', 'rewards', 'dones', 'action_probs'])


class TrajectoryBuffer:
    """ Buffer stores trajectories which can be sampled"""

    def __init__(self, obs_shape, batch_size, horizon, sequence_length, size=500, dtype=np.float32):
        """

        :param obs_shape: tuple, shape of observations
        :param batch_size: int, batch size
        :param horizon: int, maximum length of one trajectory
        :param sequence_length: int, length of sampled experience
        :param size: int, buffer size
        """
        self.bs = batch_size
        self.horizon = horizon
        self.sequence_length = sequence_length
        self.idx = 0
        self.count = 0
        self.min_fill_level = 100
        self.size = size
        self.dtype = dtype

        self.obs = np.zeros(shape=(size, horizon)+obs_shape, dtype=self.dtype)
        self.actions = np.zeros(shape=(size, horizon),  dtype=np.int32)
        self.action_probs = np.zeros(shape=(size, horizon), dtype=self.dtype)
        self.rewards = np.zeros(shape=(size, horizon), dtype=self.dtype)
        self.dones = np.zeros(shape=(size, horizon), dtype=self.dtype)
        self.bootstrap_states = np.zeros(shape=(size, *obs_shape), dtype=self.dtype)
        # TODO you can kick the bootstrap

    def add_trajectory(self, trajectory):
        obs, actions, rewards, dones, action_probs, bootstrap_state, _ = trajectory
        self.obs[self.idx] = obs
        self.actions[self.idx] = actions
        self.rewards[self.idx] = rewards
        self.dones[self.idx] = dones
        self.action_probs[self.idx] = action_probs
        self.bootstrap_states[self.idx] = bootstrap_state
        
        # TODO # comment: IMPLEMENTED in sub-classes
        # self.idx = (self.idx + 1) % self.size  
        # 
        # if self.count < self.size:
        #     self.count += 1
        
    def sample_trajectories(self, batch_size=None):
        raise NotImplementedError

    @property
    def filled(self):
        if self.count >= self.min_fill_level:  # self.size / 50:
            return True
        else:
            return False


class UniformBuffer(TrajectoryBuffer):
    
    def add_trajectory(self, trajectory):
        """ adds a trajectory to replay buffer """
        super().add_trajectory(trajectory)
        self.idx = (self.idx + 1) % self.size
        if self.count < self.size:
            self.count += 1
            
    def sample_trajectories(self, batch_size=None):
        """ generate random indices and return data of shape (batch_size, sequence_length) """

        bs = batch_size if batch_size else self.bs
        seq_len = self.sequence_length

        s = np.random.randint(low=0, high=self.count, size=bs)  # generate random indices for to choice trajectories
        n = np.random.randint(low=0, high=self.horizon - seq_len - 1)

        return (self.obs[s, n:n+seq_len], self.actions[s, n:n+seq_len], self.rewards[s, n:n+seq_len],
                self.dones[s, n:n+seq_len], self.action_probs[s, n:n+seq_len], self.obs[s, n+seq_len])


class PrioritizedBuffer(TrajectoryBuffer):
    def __init__(self, obs_shape, batch_size, horizon, sequence_length, size, alpha=0.2):
        super().__init__(obs_shape, batch_size, horizon, sequence_length, size)
        self.ranks = np.zeros(size, dtype=self.dtype)
        self.alpha = alpha
        
    def add_trajectory(self, trajectory):
        """ adds a trajectory to replay buffer """
        super().add_trajectory(trajectory)
        self.ranks -= 1
        self.ranks[self.idx] = 0

        self.idx = (self.idx + 1) % self.size
        if self.count < self.size:
            self.count += 1

    def sample_trajectories(self, batch_size=None):
        """ draw trajectoreis with respect to ranks"""
        bs = batch_size if batch_size else self.bs
        seq_len = self.sequence_length

        # exponential treatment of ranks
        y = np.exp(self.ranks[:self.count] * self.alpha / self.size)
        sample_probs = y / np.sum(y)  # normalize to 1.

        s = np.random.choice(self.count, size=bs, p=sample_probs)  # choose random indices w.r.t distribution p
        n = np.random.randint(low=0, high=self.horizon - seq_len - 1)

        return (self.obs[s, n:n+seq_len], self.actions[s, n:n+seq_len], self.rewards[s, n:n+seq_len],
                self.dones[s, n:n+seq_len], self.action_probs[s, n:n+seq_len], self.obs[s, n+seq_len])
