import tensorflow as tf
from tensorflow.python import keras
import numpy as np
import time

import tensorbox.common.utils as utils
from tensorbox.common.trainer import ReinforcementTrainer
from tensorbox.algorithms.ppo.gae import calculate_target_returns, \
    calculate_gae_advantages
from tensorbox.common.probability_distributions import get_probability_distribution, GaussianDistribution
from tensorbox.common.classes import Trajectory
from tensorbox.methods import GradientDescent


class DQNTrainer(ReinforcementTrainer):
    def __init__(self,
                 method=GradientDescent,
                 *args,
                 **kwargs):
        super(DQNTrainer, self).__init__(method=method,
                                         *args,
                                         **kwargs)
        self.batch_size = 32
        self.dataset_buffer_size = self.batch_size * 8

        self.gamma = 0.99

        self.policy_distribution = get_probability_distribution(self.env.action_space)
        # self.action_shape = env.get_action_shape()
        self.action_shape = self.env.action_space.shape
        print('action_shape =', self.env.action_space)
        # self.summary_writer = tf.summary.create_file_writer(self.log_path)
        self.summary_writer = None

        # self.value_loss_metric = keras.metrics.Mean(name='value_loss_metric')
        # self.entropy_metric = keras.metrics.Mean(name='entropy_metric')
        # self.entropy_loss_metric = keras.metrics.Mean(name='entropy_loss_metric')
        # self.policy_loss_metric = keras.metrics.Mean(name='policy_loss_metric')
        # self.total_loss_metric = keras.metrics.Mean(name='total_loss_metric')
        # self.mean_policy_ratio = keras.metrics.Mean(name='mean_policy_ratio')
        # self.approximate_kl_divergence = keras.metrics.Mean(name='approximate_kl_divergence')
        # self.clip_fraction = keras.metrics.Mean(name='clip_fraction')

    def evaluate(self):
        raise NotImplementedError
        print('mean reward =', utils.safe_mean(trajectory.rewards))

    def get_action_and_value(self, x):
        """ get actions and values w.r.t. behavior network as numpy arrays"""
        action_logits, value = self.behavior_net(x)
        action = self.policy_distribution.get_action(action_logits)
        # return np.squeeze(action.numpy()), np.squeeze(value.numpy())
        return action, np.squeeze(value.numpy())

    def value_iteration(self, data_set, epochs):
        a_dim = 3

        loss_metric = tf.keras.metrics.Mean(name='test_loss')
        for epoch in range(epochs):
            losses = []
            for (s, a, r, s_prime, a_prime) in data_set:
                with tf.GradientTape() as tape:
                    # make VI
                    q_s = self.net(s)
                    q_s_prime = self.net(s)

                    q_s_prime_max_a = tf.reduce_max(q_s_prime, axis=1)  # TT.max(Q_s_prime, axis=1)
                    one_hot_mask = tf.one_hot(a, a_dim, on_value=True, off_value=False,
                                              dtype=tf.bool)
                    q_s_a = tf.boolean_mask(q_s, one_hot_mask)

                    pi_s = tf.argmax(q_s, axis=1)
                    # TODO test with and without stop gradient
                    # TD_tgt = tf.stop_gradient(r + gamma * q_s_prime_max_a)
                    TD_tgt = r + self.gamma * q_s_prime_max_a
                    loss = tf.reduce_mean(tf.square(TD_tgt - q_s_a))

                gradients = tape.gradient(loss, self.net.trainable_variables)
                self.opt.apply_gradients(zip(gradients, self.net.trainable_variables))
                loss_metric(loss)
                losses.append(loss_metric.result())
            print('Epoch: {}\t Loss: {}'.format(epoch, np.mean(losses)))

    def train(self, epochs):
        raise NotImplementedError

    @tf.function
    def train_step(self,
                   batch,
                   clip_value,
                   **kwargs):
        raise NotImplementedError
        with tf.GradientTape() as tape:
            loss = self.build_ppo_loss(batch, clip_value=clip_value)
        gradients = tape.gradient(loss, self.net.trainable_variables)
        self.opt.apply_gradients(zip(gradients, self.net.trainable_variables))
