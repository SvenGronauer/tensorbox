import tensorflow as tf
from tensorflow.python import keras
import numpy as np
import time

import tensorbox.common.utils as utils
from tensorbox.common.trainer import ReinforcementTrainer
from tensorbox.common.probability_distributions import get_probability_distribution, GaussianDistribution
from tensorbox.methods import GradientDescent


class DQNTrainer(ReinforcementTrainer):
    def __init__(self,
                 data_wrapper,
                 method=GradientDescent,
                 *args,
                 **kwargs):
        super(DQNTrainer, self).__init__(method=method,
                                         *args,
                                         **kwargs)
        self.batch_size = 32
        self.dataset_buffer_size = self.batch_size * 8

        self.gamma = 0.99
        self.dw = data_wrapper

        self.target_net = self.net.clone_net()

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

    def get_action_and_value(self, x):
        """ get actions and values w.r.t. behavior network as numpy arrays"""
        action_logits, value = self.behavior_net(x)
        action = self.policy_distribution.get_action(action_logits)
        # return np.squeeze(action.numpy()), np.squeeze(value.numpy())
        return action, np.squeeze(value.numpy())

    def q_factor_value_iteration(self, data_set, epochs):
        a_dim = 3

        loss_metric = tf.keras.metrics.Mean(name='test_loss')
        for epoch in range(epochs):
            losses = []
            ts = time.time()
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
                    loss = tf.reduce_max(tf.square(TD_tgt - q_s_a))

                gradients = tape.gradient(loss, self.net.trainable_variables)
                self.opt.apply_gradients(zip(gradients, self.net.trainable_variables))
                loss_metric(loss)
                losses.append(loss_metric.result())
            print('Epoch: {}\t Loss: {}\t Time: {}'.format(epoch,
                                                           np.mean(losses),
                                                           time.time() - ts))

    def value_iteration(self, data_set, epochs):
        """ Learn J* trough value iteration"""

        loss_metric = tf.keras.metrics.Mean(name='test_loss')
        a_dim = 3
        k = 5

        for epoch in range(epochs):
            losses = []
            ts = time.time()
            # number_samples = 128
            # if epoch % k == 0:  # copy weights to target network
            self.target_net.set_weights(self.net.get_weights())
            for (s, a, r, s_prime, a_prime) in data_set:

                batch_size = s.shape[0]
                s_primes = []
                rewards = []

                # s = self.dw.env.sample_observation()

                for ac in range(a_dim):
                    actions = np.full((batch_size, ), ac)
                    s_prime = self.dw.env.batch_f(s.numpy(), actions)
                    rews = self.dw.env.batch_r(s.numpy(), actions, s_prime)
                    rewards.append(rews)
                    s_primes.append(s_prime)

                stacked_s_prime = np.array(s_primes)
                rewards = np.array(rewards)
                stacked_s_prime = np.reshape(s_primes, (-1, stacked_s_prime.shape[-1]))

                with tf.GradientTape(persistent=True) as tape:
                    j_s = self.net(s)
                    # j_s_prime = self.target_net(stacked_s_prime)
                    j_s_prime = self.net(stacked_s_prime)
                    j_s_prime = tf.reshape(j_s_prime, shape=rewards.shape)

                    targets = rewards + self.gamma * j_s_prime
                    # max_target = tf.stop_gradient(tf.reduce_max(targets, axis=0))
                    target = tf.reduce_mean(targets, axis=0)

                    loss_a = 0.5 * tf.reduce_mean(tf.square(tf.stop_gradient(target) - j_s))
                    loss_b = 0.5 * tf.reduce_mean(tf.square(target - tf.stop_gradient(j_s)))
                    loss_c = 0.5 * tf.reduce_mean(tf.square(target - j_s))
                    loss_d = 0.5 * tf.reduce_mean(tf.square(tf.stop_gradient(target) - j_s))

                gradients_a = tape.gradient(loss_a, self.net.trainable_variables)
                gradients_b = tape.gradient(loss_b, self.net.trainable_variables)
                gradients_c = tape.gradient(loss_c, self.net.trainable_variables)
                gradients_d = tape.gradient(loss_d, self.net.trainable_variables)
                del tape

                delta = 0.05
                # delta = epoch / (epochs - 1)
                # weighted_grads = gradients_a + delta * gradients_b

                weighted_grads = [g[0] + delta * g[1] for g in zip(gradients_a, gradients_b)]
                self.opt.apply_gradients(zip(gradients_c, self.net.trainable_variables))
                loss_metric(loss_a)
                # print('Batch Loss:', loss_metric.result())
                losses.append(loss_metric.result())
            print('Epoch: {}\t Loss: {}\t Time: {}'.format(epoch,
                                                           np.mean(losses),
                                                           time.time() - ts))

    def train(self, epochs):
        raise NotImplementedError

    @tf.function
    def train_step(self,
                   batch,
                   clip_value,
                   **kwargs):
        raise NotImplementedError

