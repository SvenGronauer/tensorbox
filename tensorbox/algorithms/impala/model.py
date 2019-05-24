import tensorflow as tf
from abc import ABC, abstractmethod
import numpy as np
import time 
from queue import Empty
from impala.vtrace import calculate_v_trace
import utils as U


class ImpalaModel:
    def __init__(self, observation_shape, n_actions, learning_rate, entropy_scale, dtype=tf.float32):
        # class variables
        self.n_actions = n_actions
        self.global_step = tf.train.get_or_create_global_step()
        self.learning_rate = learning_rate
        self.entropy_scale = entropy_scale

        # Placeholders
        self.observations = tf.placeholder(dtype=dtype, shape=[None] + list(observation_shape), name='inputs')
        self.v_trace_targets = tf.placeholder(dtype=dtype, shape=[None], name='v_trace_targets')
        self.q_targets = tf.placeholder(dtype=dtype, shape=[None], name='q_targets')
        self.behavior_policy_probs = tf.placeholder(dtype=dtype, shape=[None], name='behavior_policy_prob')
        self.actions = tf.placeholder(dtype=tf.int32, shape=[None], name='actions')
        self.activation_func = tf.nn.relu

        # call class methods
        self.logits, self.values = self.build_mlp_graph()
        self.policy_probs = tf.nn.softmax(self.logits)

        # fake nodes, overwritten by learner
        self.entropy_loss = None
        self.policy_loss = None
        self.value_loss = None
        self.entropy = None
        self.loss = None
        self.optimizer = None
        self.train_op = None

    def build_mlp_graph(self, layers=(64, 64)):
        """ build multi-layer perceptron"""
        x = self.observations
        for layer, units in enumerate(layers):
            x = tf.layers.dense(x,
                                units=units,
                                activation=self.activation_func,
                                name="dense_{}".format(layer))

        policy_logits = tf.layers.dense(x, units=self.n_actions, activation=None, name="logits")
        values = tf.layers.dense(x, units=1, activation=None, name="values")

        # with tf.variable_scope('policy'):
        #     c = self.observations
        #     for layer, units in enumerate(layers):
        #         c = tf.layers.dense(c,
        #                             units=units,
        #                             activation=self.activation_func,
        #                             name="dense_{}".format(layer))
        #     policy_logits = tf.layers.dense(c, units=self.n_actions, activation=None, name="logits")
        #
        # with tf.variable_scope('values'):
        #     v = self.observations
        #     for layer, units in enumerate(layers):
        #         v = tf.layers.dense(v,
        #                             units=units,
        #                             activation=self.activation_func,
        #                             name="dense_{}".format(layer))
        #     values = tf.layers.dense(v, units=1, activation=None, name="logits")

        return policy_logits, values

    def build_loss(self):
        """ learner must call this method"""

        self.value_loss = 0.5 * tf.reduce_mean(tf.square(self.values - self.v_trace_targets))
        # self.value_loss = tf.losses.huber_loss(predictions=self.values, labels=self.v_trace_targets, delta=1.0)

        log_policy = tf.nn.log_softmax(self.logits)  # make the log calculation numerical stable (avoid log of zero)
        self.entropy = tf.reduce_mean(tf.reduce_sum(-self.policy_probs * log_policy, axis=-1))
        self.entropy_loss = -self.entropy_scale * self.entropy

        # policy loss
        neg_log_pi = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.actions, logits=self.logits)
        self.policy_loss = tf.reduce_mean(neg_log_pi * self.q_targets)

        self.loss = self.policy_loss + self.value_loss + self.entropy_loss

    def build_trainer(self, max_grad_norm=None):
        """ learner must call this method"""

        # params = tf.trainable_variables()
        # grads = tf.gradients(self.loss, params)
        # if max_grad_norm is not None:  # Clip the gradients (normalize)
        #     grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        # grads_and_vars = list(zip(grads, params))

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)

        # lr = tf.train.polynomial_decay(self.learning_rate, self.global_step, 100000, 0)
        # decay = 0.99
        # momentum = 0.0
        # epsilon = 0.1
        # self.optimizer = tf.train.RMSPropOptimizer(lr, decay, momentum, epsilon)
        # if self.args.debug:
        #     for grad, var in grads_and_vars:
        #         if grad is not None:  # must be explicitely stated like this, to avoid error
        #             tf.summary.histogram(var.name + '/gradient', grad)

        # self.train_op = self.optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)
        self.train_op = self.optimizer.minimize(loss=self.loss, global_step=self.global_step)

    def get_action_and_prob(self, session, observation):
        """ forward pass to get action, expects single observation"""
        assert observation.ndim == 1
        observation = observation[None, :]  # make dimension = 2 to allow forward pass
        policy_prob, values = session.run([self.policy_probs, self.values],
                                          feed_dict={self.observations: observation})
        policy_prob = np.squeeze(policy_prob)

        action = np.random.choice(self.n_actions, p=policy_prob)
        action = int(np.squeeze(action))
        return action, policy_prob[action]

    def get_policy_probs_and_values(self, session, observation):
        """ forward pass to get policy probabilities and values"""
        policy_probs, values = session.run([self.policy_probs, self.values],
                                           feed_dict={self.observations: observation})
        return policy_probs, values

    @property
    def train_ops(self):
        return [self.train_op, self.global_step, self.loss, self.value_loss,
                self.policy_loss, self.entropy_loss, self.entropy]


class Worker:
    def work(self, session):
        raise NotImplementedError


class Learner(Worker):
    def __init__(self, model, queue, buffer, logger, norm, rho_clip, **kwargs):
        self.model = model
        self.queue = queue
        self.buffer = buffer
        self.logger = logger
        self.normalizer = norm
        self.returns = []
        self.rho_clip = rho_clip

    def work(self, session):
        return self.learn_from_buffer(session=session)
    
    def learn_from_buffer(self, session):
        try:  # fetch as many trajectories as possible from queue and add to experience buffer
            while True:
                trajectory = self.queue.get(block=True, timeout=0.05)
                self.returns.append(trajectory[-1])
                self.buffer.add_trajectory(trajectory)
        except Empty:
            # start training Loop
            if self.buffer.filled:
                prev_gstep = session.run(self.model.global_step)
                start_time = time.time()

                # Epoch training
                values = np.zeros(5)
                steps_per_epoch = 100
                for step in range(steps_per_epoch):
                    trajectory = self.buffer.sample_trajectories()
                    obs, actions, rewards, dones, behavior_action_probs, bootstrap_state = trajectory

                    B, H, *obs_shape = obs.shape  # (batch_size, horizon)
                    normed_flat_obs, _ = self.normalizer.normalize(obs, rewards, update_internal_with_session=session)

                    # determine π(a|s) for each state in trajectory batch  # TODO changed
                    # _policy_action_probs, flat_values = self.model.get_policy_probs_and_values(session=session,
                    #                                                                            observation=normed_flat_obs)
                    _policy_action_probs, flat_values = self.model.get_policy_probs_and_values(session=session,
                                                                                               observation=obs.reshape((B * H, *obs_shape)))
                    # normed_bootstrap_state, normed_rews = self.normalizer.normalize(bootstrap_state, rewards)
                    _policy_action_probs = _policy_action_probs.reshape((B, H, -1))
                    bootstrap_value = session.run(self.model.values,
                                                  # feed_dict={self.model.observations: normed_bootstrap_state})  TODO changed
                                                  feed_dict={self.model.observations: bootstrap_state})

                    policy_action_probs = np.zeros(shape=(B, H), dtype=np.float32)

                    for b in range(B):  # get π(a|s) for each (s, a)-pair in trajectory batch
                        for h in range(H):
                            policy_action_probs[b, h] = _policy_action_probs[b, h, actions[b, h]]

                    # print('np.max =', np.max(policy_action_probs))
                    # print('np.min =', np.min(policy_action_probs))

                    v_trace = calculate_v_trace(policy_action_probs=policy_action_probs,
                                                values=flat_values.reshape((B, H)),
                                                rewards=rewards.reshape((B, H)),  # TODO changed
                                                bootstrap_value=bootstrap_value,
                                                behavior_action_probs=behavior_action_probs,
                                                rho_bar=self.rho_clip)
                    # normed_adv = U.normalize(v_trace.policy_adv.flatten())
                    feed = {
                        self.model.observations: normed_flat_obs,
                        self.model.v_trace_targets: v_trace.v_s.flatten(),
                        self.model.q_targets: v_trace.policy_adv.flatten(),
                        self.model.actions: actions.flatten(),
                        self.model.behavior_policy_probs: behavior_action_probs.flatten()
                    }
                    trained_ops = session.run(self.model.train_ops, feed_dict=feed)
                    _, gstep, loss, value, policy, entropy_loss, entropy = trained_ops
                    values += np.array([loss, value, policy, entropy_loss, entropy])

                values /= steps_per_epoch
                summary = {'Debug/rho_mean': np.mean(v_trace.clipped_rho),
                           'Debug/rho_std': np.std(v_trace.clipped_rho),
                           'Debug/rho_min': np.min(v_trace.clipped_rho),
                           'Debug/rho_max': np.max(v_trace.clipped_rho),
                           'Debug/policy_adv_mean': np.mean(v_trace.policy_adv),
                           'Debug/policy_adv_std': np.std(v_trace.policy_adv),
                           'Debug/policy_adv_min': np.min(v_trace.policy_adv),
                           'Debug/policy_adv_max': np.max(v_trace.policy_adv),
                           'Training/Loss': values[0],
                           'Training/Value Loss': values[1],
                           'Training/Policy Loss': values[2],
                           'Training/Entropy Loss': values[3],
                           'Training/Entropy': values[4],
                           'Training/Buffer Fill Level': self.buffer.count,
                           'Training/Gsteps_per_sec': (gstep - prev_gstep) / (time.time() - start_time)}
                if self.returns:
                    summary['Training/Episode Return'] = U.safe_mean(self.returns)
                self.logger.write(summary, global_step=gstep)
                print('Global step {}: Returns={:0.1f}, Entropy={:0.2f}, Value Loss={:0.2f}'.format(
                    gstep, U.safe_mean(self.returns), values[4], values[1]))
                self.returns = []
            else:
                print('Learner waits until self.buffer is filled ({}/{})'.format(self.buffer.count,
                                                                                 self.buffer.min_fill_level))
                time.sleep(5)
    
    
class Actor(Worker):
    def __init__(self, model, env, queue, horizon, normalizer, **kwargs):
        self.model = model
        self.env = env
        self.queue = queue
        self.horizon = horizon
        self.normalizer = normalizer
        self.warm_up = True
        self.warm_up_threshold = 50
        self.count = 0
    
    def work(self, session):
        self.count += 1
        if self.count >= self.warm_up_threshold:  # generate random trajectories until warm_up = False
            self.warm_up = False
        return self.roll_out_trajectories(session=session)

    def roll_out_trajectories(self, session):
        x = self.env.reset()
        dones = np.zeros(self.horizon, dtype=np.float32)
        obs = np.zeros(shape=(self.horizon,) + self.env.observation_space.shape, dtype=np.float32)
        actions = np.zeros(self.horizon, dtype=np.int32)
        action_probs = np.zeros(self.horizon, dtype=np.float32)
        rewards = np.zeros(self.horizon, dtype=np.float32)
        step = 0
        done = False
        episode_return = 0

        self.normalizer.get_values_from_tf_graph(session=session)  # fetch newest mean/std values

        while not done and step < self.horizon and not session.should_stop():
            if self.warm_up:  # sample random during warm-up phase
                action = np.random.randint(low=0, high=self.env.action_space.n)
                action_prob = 1.0 / self.env.action_space.n
            else:
                normed_obs = self.normalizer.normalize_observation(x)
                # action, action_prob = self.model.get_action_and_prob(session=session, observation=normed_obs)
                action, action_prob = self.model.get_action_and_prob(session=session, observation=x)  # TODO changed
            y, reward, done, info = self.env.step(action)
            # fill trajectory arrays
            obs[step] = x
            actions[step] = action
            rewards[step] = reward
            dones[step] = 1.0 if done else 0.0
            action_probs[step] = action_prob
            episode_return += reward
            x = y
            step += 1

        # Note that trajectory takes un-processed rewards and observations
        trajectory = (obs, actions, rewards, dones, action_probs, x, episode_return)  # , values)
        self.normalizer.update_ops(obs, rewards, session=session)  # actors update mean/std nodes for obs and rewards
        self.queue.put(trajectory)
        # print('newest internals:')
        # print(self.normalizer.ob_rms.mean)
        # print(self.normalizer.ret_rms.mean)
        return episode_return
