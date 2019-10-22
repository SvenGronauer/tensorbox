import tensorflow as tf
from tensorflow.python import keras
import numpy as np
import time
from copy import deepcopy

import tensorbox.common.utils as utils
from tensorbox.networks import SharedMLPNet, MLPNet
from tensorbox.common.trainer import ReinforcementTrainer
from tensorbox.algorithms.ppo.gae import calculate_target_returns, \
    calculate_gae_advantages
from tensorbox.common.probability_distributions import get_probability_distribution, \
    GaussianDistribution
from tensorbox.common.classes import Trajectory


class PPOTrainer(ReinforcementTrainer):
    def __init__(self, horizon=1024,
                 *args,
                 **kwargs):
        super(PPOTrainer, self).__init__(*args,
                                         **kwargs)
        # clones only the network structure, the weights still differ!
        self.behavior_net = self.net.clone_net_structure()
        # Info: parameter sharing means that the instance SharedMLPNet() is used to share weights
        # between the policy and the value network
        self.parameter_sharing = True
        if not isinstance(self.net, SharedMLPNet):  # create a value network
            self.value_net = MLPNet(in_dim=self.net.in_dim,
                                    out_dim=1,
                                    units=self.net.units,
                                    activation=self.net.activation)
            self.parameter_sharing = False
            self.old_value_net = self.value_net.clone_net_structure()
        self.value_opt = deepcopy(self.opt)

        """ PPO parameters """
        self.K = 15
        self.gamma = 0.99               # default: 0.99
        self.horizon = horizon
        self.batch_size = 512
        self.clip_value = 0.2
        self.start_clip_value = 0.2     # default: 0.2
        self.entropy_loss_factor = 0.   # default: 0.01
        self.dataset_buffer_size = self.batch_size * 4

        self.policy_distribution = get_probability_distribution(self.env.action_space)
        self.action_shape = self.env.action_space.shape
        if self.logger:
            self.summary_writer = tf.summary.create_file_writer(self.log_dir)
            print('Create TF event files at:', self.log_dir)

        self.latest_train_trajectory = None
        self.time_start = time.time()

        self.value_loss_metric = keras.metrics.Mean(name='value_loss_metric')
        self.entropy_metric = keras.metrics.Mean(name='entropy_metric')
        self.entropy_loss_metric = keras.metrics.Mean(name='entropy_loss_metric')
        self.policy_loss_metric = keras.metrics.Mean(name='policy_loss_metric')
        self.total_loss_metric = keras.metrics.Mean(name='total_loss_metric')
        self.mean_policy_ratio = keras.metrics.Mean(name='mean_policy_ratio')
        self.approximate_kl_divergence = keras.metrics.Mean(name='approximate_kl_divergence')
        self.clip_fraction = keras.metrics.Mean(name='clip_fraction')

    def build_ppo_loss(self, batch, clip_value=0.2, **kwargs):
        """ build the surrogate objective function according to Schulman et al. 2017
        :param batch: iterator over tf.data.Dataset()
        :param clip_value: float, bound for surrogate objective
        :return:
        """
        observations, actions, advantages, old_values, target_returns = batch

        if self.parameter_sharing:
            pi_logits, values = self.net(observations)
            b_logits,  _ = self.behavior_net(observations)
        else:
            pi_logits = self.net(observations)
            values = self.value_net(observations)
            b_logits = self.behavior_net(observations)

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
        value_loss_1 = tf.square(values - tf.stop_gradient(target_returns))
        value_loss_2 = tf.square(values_clipped - tf.stop_gradient(target_returns))
        value_loss = 0.5 * tf.reduce_mean(tf.maximum(value_loss_1, value_loss_2))

        entropy = self.policy_distribution.entropy(pi_logits, from_logits=True)
        entropy_loss = -self.entropy_loss_factor * tf.reduce_mean(entropy)
        total_loss = policy_loss + value_loss + entropy_loss
        policy_net_loss = policy_loss  # + entropy_loss

        # feed metrics
        self.value_loss_metric(value_loss)
        self.entropy_loss_metric(entropy_loss)
        self.entropy_metric(entropy)
        self.policy_loss_metric(policy_loss)
        self.total_loss_metric(total_loss)
        self.mean_policy_ratio(tf.reduce_mean(ratio))
        self.approximate_kl_divergence(0.5 * tf.reduce_mean(tf.square(neg_log_pi - neg_log_b)))
        self.clip_fraction(tf.cast(tf.greater(tf.abs(ratio - 1.0), clip_value), tf.float32))

        return policy_net_loss, value_loss

    def copy_weights(self):
        if self.parameter_sharing:
            self.behavior_net.set_weights(self.net.get_weights())
        else:
            self.behavior_net.set_weights(self.net.get_weights())
            self.old_value_net.set_weights(self.value_net.get_weights())

    def sample_action_and_value(self, x):
        """ get actions and values w.r.t. behavior network as numpy arrays"""
        if self.parameter_sharing:
            action_logits, value = self.behavior_net(x)
        else:
            action_logits = self.behavior_net(x)
            value = self.old_value_net(x)
        action = self.policy_distribution.sample(action_logits, as_numpy=True)
        # return np.squeeze(action.numpy()), np.squeeze(value.numpy())
        return action, np.squeeze(value.numpy())
    
    def get_action_and_value(self, x):
        if self.parameter_sharing:
            action_logits, value = self.behavior_net(x)
        else:
            action_logits = self.behavior_net(x)
            value = self.old_value_net(x)
        action = self.policy_distribution.mode(action_logits, as_numpy=True)
        # return np.squeeze(action.numpy()), np.squeeze(value.numpy())
        return action, np.squeeze(value.numpy())
    
    def get_trajectories(self, dtype=np.float32):
        """ run behavior-policy roll-outs to obtain trajectories"""
        obs = self.env.reset()
        # ac = self.env.action_space.sample()
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
            if self.training:  # sample from distribution during training
                acs, val = self.sample_action_and_value(obs)
            else:
                acs, val = self.sample_action_and_value(obs)
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

        if self.parameter_sharing:
            _, bootstrap_values = self.behavior_net(obs)
        else:
            bootstrap_values = self.old_value_net(obs)
        obs_t_plus_1[self.horizon] = obs
        values_t_plus_1[self.horizon] = np.squeeze(bootstrap_values)

        if count > 0:
            ret = np.squeeze((mean_episode_returns / count).numpy())
        else:  # # if horizons of trajectories are infinite
            ret = (tf.reduce_sum(episode_return) / num_env).numpy()

        return Trajectory(observations=obs_t_plus_1,
                          actions=actions,
                          rewards=rewards,
                          dones=dones,
                          values=values_t_plus_1,
                          mean_episode_return=ret,
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
        """ logging result values from training iterations into files
        :param step: int, current iteration
        :return: None
        """
        if not self.training:
            return

        self.training = False
        evaluation_returns = self.policy_evaluation()
        self.training = True
        
        if self.logger:
            write_dic = {'value_loss': self.value_loss_metric.result().numpy(),
                         'Approx KL Divergence': self.approximate_kl_divergence.result().numpy(),
                         'Training Return': self.latest_train_trajectory.mean_episode_return,
                         'Evaluation Return': evaluation_returns,
                         'clip Value': self.clip_value,
                         'Entropy': self.entropy_metric.result().numpy(),
                         'total_loss_metric': self.total_loss_metric.result().numpy(),
                         'clip_fraction': self.clip_fraction.result().numpy(),
                         'policy loss': self.policy_loss_metric.result().numpy(),
                         'mean policy ratio': self.mean_policy_ratio.result().numpy(),
                         'Entropy Loss': self.entropy_loss_metric.result().numpy(),
                         'Time': time.time() - self.time_start}
            self.logger.write(write_dic, step)

    def policy_evaluation(self):
        self.training = False
        trajectory = self.get_trajectories()
        # print('mean reward =', trajectory.mean_episode_return)
        self.training = True
        return trajectory.mean_episode_return

    def restore(self):
        super(PPOTrainer, self).restore()  # restores only policy net
        self.behavior_net.set_weights(self.net.get_weights())  # copy weights
        print('copied net.weights -> behavior_net.weights')

    def train(self, epochs):
        self.training = True
        print('Start training for {} epochs'.format(epochs))
        for epoch in range(epochs):
            self.time_start = time.time()
            percentage_of_total = epoch/epochs
            self.clip_value = self.start_clip_value * (1.0 - percentage_of_total)
            value_losses = []
            if epoch % self.K == 0:
                self.copy_weights()
                self.policy_distribution.update(percentage_of_total)
            self.latest_train_trajectory = self.get_trajectories()
            ds = self.get_dataset(self.latest_train_trajectory)
            for n_batch, batch in enumerate(ds):
                self.train_step(batch, self.clip_value)
                value_losses.append(self.value_loss_metric.result())
            self.logging(epoch)
        # self.training = False
        self.behavior_net.set_weights(self.net.get_weights())  # copy weights

    # @tf.function
    def train_step(self, batch, clip_value, **kwargs):
        with tf.GradientTape(persistent=True) as tape:
            policy_loss, value_loss = self.build_ppo_loss(batch, clip_value=clip_value, **kwargs)
        policy_gradients = tape.gradient(policy_loss, self.net.trainable_variables)
        value_gradients = tape.gradient(value_loss, self.value_net.trainable_variables)
        self.opt.apply_gradients(zip(policy_gradients, self.net.trainable_variables))
        self.value_opt.apply_gradients(zip(value_gradients, self.value_net.trainable_variables))
        del tape
