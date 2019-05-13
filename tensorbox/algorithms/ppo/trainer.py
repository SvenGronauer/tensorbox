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


class PPOTrainer(ReinforcementTrainer):
    def __init__(self, net, opt, env, log_path, horizon=1024, debug_level=0, **kwargs):
        super(PPOTrainer, self).__init__(net=net,
                                         opt=opt,
                                         env=env,
                                         log_path=log_path,
                                         debug_level=debug_level,
                                         **kwargs)
        self.behavior_net = net.clone_net()  # clones the network model, but weight values differ
        self.horizon = horizon
        self.batch_size = 32
        self.dataset_buffer_size = self.batch_size * 8

        """ ppo parameters """
        self.K = 1
        self.gamma = 0.99
        self.clip_value = 0.2
        self.start_clip_value = 0.2

        self.policy_distribution = get_probability_distribution(env.action_space)
        # self.action_shape = env.get_action_shape()
        self.action_shape = env.action_space.shape
        self.summary_writer = tf.summary.create_file_writer(self.log_path)

        self.latest_trajectory = None

        self.value_loss_metric = keras.metrics.Mean(name='value_loss_metric')
        self.entropy_metric = keras.metrics.Mean(name='entropy_metric')
        self.entropy_loss_metric = keras.metrics.Mean(name='entropy_loss_metric')
        self.policy_loss_metric = keras.metrics.Mean(name='policy_loss_metric')
        self.total_loss_metric = keras.metrics.Mean(name='total_loss_metric')
        self.mean_policy_ratio = keras.metrics.Mean(name='mean_policy_ratio')
        self.approximate_kl_divergence = keras.metrics.Mean(name='approximate_kl_divergence')
        self.clip_fraction = keras.metrics.Mean(name='clip_fraction')

    def build_ppo_loss(self, batch, clip_value=0.2):
        """ build the surrogate objective function according to Schulman et al. 2017
        :param batch: iterator over tf.data.Dataset()
        :param clip_value: float, bound for surrogate objective
        :return:
        """
        observations, actions, advantages, old_values, target_returns = batch

        pi_logits, values = self.net(observations)
        b_logits,  _ = self.behavior_net(observations)
        # Note: BUG in TF2.0, must cast explicitly to float32
        pi_logits = tf.cast(pi_logits, tf.float32)
        b_logits = tf.cast(b_logits, tf.float32)
        values = tf.cast(values, tf.float32)

        neg_log_pi = self.policy_distribution.neg_log(actions, pi_logits)
        neg_log_b = self.policy_distribution.neg_log(actions, b_logits)

        # new_policy / old_policy = exp(log_new - log_old) = exp(neg_log_old - neg_log_new)
        ratio = tf.exp(tf.stop_gradient(neg_log_b) - neg_log_pi)

        un_clipped_loss = advantages * ratio
        clipped_loss = advantages * tf.clip_by_value(ratio, 1.0 - clip_value, 1.0 + clip_value)
        # make minimization problem: max J = min -J
        policy_loss = -tf.reduce_mean(tf.minimum(un_clipped_loss, clipped_loss))

        # clipped value loss
        values_clipped = old_values + tf.clip_by_value(values - old_values, - clip_value, clip_value)
        value_loss_1 = tf.square(values - target_returns)
        value_loss_2 = tf.square(values_clipped - target_returns)
        value_loss = 0.5 * tf.reduce_mean(tf.maximum(value_loss_1, value_loss_2))
        # value_loss = 0.5 * tf.reduce_mean(value_loss_1)

        entropy = self.policy_distribution.entropy(pi_logits, from_logits=True)
        entropy_loss = -0.01 * tf.reduce_mean(entropy)
        total_loss = policy_loss + value_loss + entropy_loss

        # feed metrics
        self.value_loss_metric(value_loss)
        self.entropy_loss_metric(entropy_loss)
        self.entropy_metric(entropy)
        self.policy_loss_metric(policy_loss)
        self.total_loss_metric(total_loss)
        self.mean_policy_ratio(tf.reduce_mean(ratio))
        self.approximate_kl_divergence(0.5 * tf.reduce_mean(tf.square(neg_log_pi - neg_log_b)))
        self.clip_fraction(tf.cast(tf.greater(tf.abs(ratio - 1.0), clip_value), tf.float32))

        return total_loss

    def evaluate(self):
        trajectory = self.get_trajectories()
        print('mean reward =', utils.safe_mean(trajectory.rewards))

    def get_action_and_value(self, x):
        """ get actions and values w.r.t. behavior network as numpy arrays"""
        action_logits, value = self.behavior_net(x)
        action = self.policy_distribution.sample(action_logits)
        return np.squeeze(action.numpy()), np.squeeze(value.numpy())

    def get_trajectories(self, dtype=np.float32):
        """ run behavior-policy roll-outs to obtain trajectories"""
        obs = self.env.reset()
        num_env = obs.shape[0] if len(obs.shape) >= 2 else 1
        if isinstance(self.policy_distribution, GaussianDistribution):
            a_shape = (self.horizon, num_env) + self.env.get_action_shape()
            actions = np.zeros(shape=a_shape, dtype=dtype)
        else:  # make discrete actions
            actions = np.zeros(shape=(self.horizon, num_env), dtype=np.int32)  # todo adjust actions
        obs_shape = tuple([self.horizon+1, num_env] + list(self.env.observation_space.shape))
        obs_t_plus_1 = np.zeros(shape=obs_shape, dtype=dtype)
        rewards = np.zeros(shape=(self.horizon, num_env), dtype=dtype)
        values_t_plus_1 = np.zeros(shape=(self.horizon+1, num_env), dtype=dtype)
        dones = np.zeros(shape=(self.horizon, num_env), dtype=dtype)
        episode_return = np.zeros((num_env, ))
        mean_episode_returns = tf.zeros(1)
        count = tf.zeros(1)

        for t in range(self.horizon):
            # acs, val = self.behavior_net(obs)
            if len(obs.shape) == 1:
                obs = obs.reshape((-1, obs.shape[0]))
            acs, val = self.get_action_and_value(obs)
            new_obs, r, ds, _ = self.env.step(actions=acs)
            ds = tf.cast(ds, tf.float32)
            i = t % self.horizon
            obs_t_plus_1[i] = obs
            rewards[i] = r
            values_t_plus_1[i] = val
            actions[i] = acs
            dones[i] = ds
            obs = new_obs

            # track episode return
            episode_return += r
            mean_episode_returns += tf.reduce_sum(episode_return * ds)
            count += tf.reduce_sum(ds)
            episode_return *= (1. - ds)

        _, bootstrap_values = self.behavior_net(obs)
        obs_t_plus_1[self.horizon] = obs
        values_t_plus_1[self.horizon] = np.squeeze(bootstrap_values)

        a = np.squeeze((mean_episode_returns/count).numpy())
        return Trajectory(observations=obs_t_plus_1,
                          actions=actions,
                          rewards=rewards,
                          dones=dones,
                          values=values_t_plus_1,
                          mean_episode_return=a,
                          horizon=self.horizon)

    def get_dataset(self, trajectory):
        """ create tf dataset from trajectory data
        :param trajectory: Trajectory(), holding policy roll-out data
        :return: tf.data.Dataset(), unzip to obs, acs, adv, t_ret = batch
        """
        adv = calculate_gae_advantages(trajectory=trajectory,
                                       gamma=self.gamma)
        target_returns = calculate_target_returns(trajectory=trajectory,
                                                  gamma=self.gamma)
        advantages = utils.normalize(adv)  # Normalize the advantages
        obs = trajectory.observations[:-1]  # remove t_plus_1 state to match shapes
        values = trajectory.values[:-1]

        # Reshape and flatten stacked interactions
        if isinstance(self.policy_distribution, GaussianDistribution):
            new_actions_shape = tuple([-1] + list(trajectory.actions.shape[2:]))
            actions = trajectory.actions.reshape(new_actions_shape)
        else:
            actions = trajectory.actions.flatten()
        obs = obs.reshape(tuple([-1] + list(obs.shape[2:])))  # shape into (horizon * N, obs.shape)
        ds = tf.data.Dataset.from_tensor_slices((obs,
                                                actions,
                                                advantages.flatten(),
                                                values.flatten(),
                                                target_returns.flatten()))
        return ds.shuffle(self.dataset_buffer_size).batch(self.batch_size)

    def logging(self, step):
        # t = self.opt.step  # todo make me opt step
        t = step
        with self.summary_writer.as_default():
            # t = self.opt.iterations
            tf.summary.scalar('clip value', self.clip_value, step=t)
            tf.summary.scalar('value loss', self.value_loss_metric.result(), step=t)
            tf.summary.scalar('entropy', self.entropy_metric.result(), step=t)
            tf.summary.scalar('policy loss', self.policy_loss_metric.result(), step=t)
            tf.summary.scalar('entropy loss', self.entropy_loss_metric.result(), step=t)
            tf.summary.scalar('total_loss_metric', self.total_loss_metric.result(), step=t)
            tf.summary.scalar('approximate_kl_divergence', self.approximate_kl_divergence.result(), step=t)
            tf.summary.scalar('clip_fraction', self.clip_fraction.result(), step=t)
            tf.summary.scalar('mean policy ratio', self.mean_policy_ratio.result(), step=t)
            tf.summary.scalar('mean episode return', self.latest_trajectory.mean_episode_return, step=t)

    def train(self, epochs):
        print('Start training for {} epochs'.format(epochs))
        for epoch in range(epochs):
            ts = time.time()
            self.clip_value = self.start_clip_value * (1.0 - epoch/epochs)
            value_losses = []
            if epoch % self.K == 0:
                self.behavior_net.set_weights(self.net.get_weights())
            self.latest_trajectory = self.get_trajectories()
            ds = self.get_dataset(self.latest_trajectory)
            for n_batch, batch in enumerate(ds):
                self.train_step(batch, self.clip_value)
                value_losses.append(self.value_loss_metric.result())
            # print('Value loss=', utils.safe_mean(value_losses))
            # self.evaluate()
            print('Episode {} \t episode return: {:0.3f} \t took: {:0.2f}s'.format(epoch,
                self.latest_trajectory.mean_episode_return,
                time.time() - ts))
            self.logging(epoch)

    @tf.function
    def train_step(self, batch, clip_value, **kwargs):
        with tf.GradientTape() as tape:
            loss = self.build_ppo_loss(batch, clip_value=clip_value)
        gradients = tape.gradient(loss, self.net.trainable_variables)
        self.opt.apply_gradients(zip(gradients, self.net.trainable_variables))
