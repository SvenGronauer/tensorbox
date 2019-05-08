import datetime
from abc import ABC, abstractmethod
import os
import tensorflow as tf
import numpy as np
import gym
from collections import namedtuple

import tensorbox.common.utils as U
from tensorbox.common.trainer import ReinforcementTrainer
from tensorbox.algorithms.ppo.gae import calculate_target_returns, \
    calculate_gae_advantages

from tensorbox.common.probability_distributions import GaussianDistribution, \
    CategoricalDistribution


Trajectory = namedtuple('Trajectory', ['observations',
                                       'actions',
                                       'rewards',
                                       'dones',
                                       'values',
                                       'horizon'])


class PPOTrainer(ReinforcementTrainer):
    def __init__(self, net, opt, env, log_path, horizon=64, debug_level=0, **kwargs):
        super(PPOTrainer, self).__init__(net=net,
                                         opt=opt,
                                         env=env,
                                         log_path=log_path,
                                         debug_level=debug_level,
                                         **kwargs)
        self.behavior_net = net  # todo copy structure form policy network
        self.horizon = horizon

        if isinstance(env.action_space, gym.spaces.Discrete):
            self.policy_distribution = CategoricalDistribution(num_classes=env.action_space.n)
        elif isinstance(env.action_space, gym.spaces.Box):
            raise NotImplementedError
        else:
            raise NotImplementedError

        # create nodes to fill later
        self.entropy = None
        self.total_loss = None

    def build_ppo_loss(self, dataset, clip_value=0.2):

        """
        """

        """ actions = from dataset
            advantages = from dataset
            observations = from dataset
            target_returns = from dataset
        """
        for n_batch, batch in dataset:
            observations, actions, advantages, target_returns = batch

        policy_logits,  values = self.net(observations)
        behavior_logits,  target_values = self.net(observations)  # todo stop gradient here

        ratio = tf.divide(tf.math.softmax(policy_logits, axis=-1), tf.math.softmax(behavior_logits, axis=-1))


        un_clipped_loss = advantages * ratio
        clipped_loss = advantages * tf.clip_by_value(ratio, 1.0 - clip_value, 1.0 + clip_value)
        policy_surrogate = - tf.reduce_mean(tf.minimum(un_clipped_loss, clipped_loss))  # make minimization problem

        # ===== Value Loss + Entropy =====
        value_loss = 0.5 * tf.reduce_mean(tf.square(target_returns - values))
        self.entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.policy_probs,
                                                                  logits=self.policy_logits)
        entropy_loss = -0.01 * tf.reduce_mean(self.entropy)
        total_loss = policy_surrogate + value_loss + entropy_loss

    def train(self, epochs):
        for i in range(10):
            print(i)

    def get_action_and_value(self, x):
        action_logits, value = self.behavior_net(x)
        action = self.policy_distribution.sample(action_logits, use_logits=True)

        # action = action.astype(np.int32)
        # val = value.numpy()
        # action = action.numpy()
        return action.astype(np.int32), np.squeeze(value.numpy())

    def get_trajectories(self):
        """ run behavior-policy roll-outs to obtain trajectories"""

        obs = self.env.reset()
        num_env = obs.shape[0] if len(obs.shape) >= 2 else 1
        actions = np.zeros(shape=(self.horizon, num_env), dtype=np.int64)
        obs_shape = tuple([self.horizon+1, num_env] + list(self.env.observation_space.shape))

        obs_t_plus_1 = np.zeros(shape=obs_shape, dtype=np.float64)
        rewards = np.zeros(shape=(self.horizon, num_env), dtype=np.float64)
        values_t_plus_1 = np.zeros(shape=(self.horizon+1, num_env), dtype=np.float64)
        target_returns = np.zeros(shape=(self.horizon, num_env), dtype=np.float64)
        advantages = np.zeros(shape=(self.horizon, num_env), dtype=np.float64)
        dones = np.zeros(shape=(self.horizon, num_env), dtype=np.float64)

        # episode_rewards = []
        # episode_steps = []

        for t in range(self.horizon):
            # acs, val = self.behavior_net(obs)
            if len(obs.shape) == 1:
                obs = obs.reshape((-1, obs.shape[0]))
            acs, val = self.get_action_and_value(obs)
            new_obs, r, ds, _ = self.env.step(actions=acs)
            i = t % self.horizon
            obs_t_plus_1[i] = obs
            rewards[i] = r
            values_t_plus_1[i] = val
            actions[i] = acs
            dones[i] = ds
            obs = new_obs

        # perform bootstrap
        if len(obs.shape) == 1:
            obs = obs.reshape((-1, obs.shape[0]))
        _, bootstrap_values = self.behavior_net(obs)
        obs_t_plus_1[self.horizon] = obs
        values_t_plus_1[self.horizon] = np.squeeze(bootstrap_values)

        return Trajectory(observations=obs_t_plus_1,
                          actions=actions,
                          rewards=rewards,
                          dones=dones,
                          values=values_t_plus_1,
                          horizon=self.horizon)

    @staticmethod
    def get_dataset(trajectory, batch_size=32, buffer_size=64):
        """

        :param trajectory:
        :param batch_size:
        :param buffer_size:
        :return: tf.data.Dataset(), unzip to obs, acs, adv, t_ret = batch
        """
        gamma = 0.99
        adv = calculate_gae_advantages(trajectory=trajectory,
                                       gamma=gamma)
        target_returns = calculate_target_returns(trajectory=trajectory,
                                                  gamma=gamma)
        advantages = U.normalize(adv)  # Normalize the advantages
        obs = trajectory.observations[:-1]  # remove t_plus_1 state to match

        # Reshape and flatten stacked interactions
        new_shape = tuple([-1] + list(obs.shape[2:]))
        obs = obs.reshape(new_shape)  # shape into [self.horizon * N, obs.shape]

        ds = tf.data.Dataset.from_tensor_slices((obs,
                                                trajectory.actions.flatten(),
                                                advantages.flatten(),
                                                target_returns.flatten()))
        return ds.shuffle(buffer_size).batch(batch_size)
