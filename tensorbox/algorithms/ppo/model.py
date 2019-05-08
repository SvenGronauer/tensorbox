import tensorflow as tf
import numpy as np
import os
import tensorbox.common.utils as U


class Model(object):
    def __init__(self, scope, observation_shape, n_actions, args, mode="training"):
        self.scope = scope
        self.n_actions = n_actions
        self.units = 64
        self.activation_func = tf.nn.relu
        self.activation_func_policy = tf.nn.relu
        self.args = args
        self.mode = mode

        # input placeholders
        with tf.variable_scope(self.scope):
            self.observations = tf.placeholder(tf.float32, shape=[None] + list(observation_shape), name='inputs')
            self.advantages = tf.placeholder(dtype=tf.float32, shape=[None], name='advantages')
            self.target_returns = tf.placeholder(dtype=tf.float32, shape=[None], name='target_returns')  # Empirical return
            self.actions = tf.placeholder(dtype=tf.int32, shape=[None], name='actions')
            if args.network == 'mlp':
                self.policy_logits, self.old_policy_logits, self.values, self.old_values = self.build_mlp_graph()
            elif args.network == 'cnn':
                raise NotImplementedError
            else:
                raise NotImplementedError

            # make logits to probabilities
            self.policy_probs = tf.nn.softmax(self.policy_logits)
            self.old_policy_probs = tf.nn.softmax(self.old_policy_logits)
            self.output = self.policy_probs

    def build_mlp_graph(self):

        with tf.variable_scope('policy'):
            c = tf.layers.dense(self.observations, units=self.units, activation=self.activation_func_policy, name="common_dense_1")
            c = tf.layers.dense(c, units=self.units, activation=self.activation_func_policy, name="common_dense_2")
            policy_logits = tf.layers.dense(c, units=self.n_actions, activation=None, name="logits")

        with tf.variable_scope('old_policy'):
            o = tf.layers.dense(self.observations, units=self.units, activation=self.activation_func_policy, name="dense_1")
            o = tf.layers.dense(o, units=self.units, activation=self.activation_func_policy, name="dense_2")
            old_policy_logits = tf.layers.dense(o, units=self.n_actions, activation=None, name="logits")

        with tf.variable_scope('values'):
            v = tf.layers.dense(self.observations, units=self.units, activation=self.activation_func, name="dense_1")
            v = tf.layers.dense(v, units=self.units, activation=self.activation_func, name="dense_2")
            values = tf.layers.dense(v, units=1, activation=None, name="logit")
        with tf.variable_scope('old_values'):
            o = tf.layers.dense(self.observations, units=self.units, activation=self.activation_func, name="dense_1")
            o = tf.layers.dense(o, units=self.units, activation=self.activation_func, name="dense_2")
            old_values = tf.layers.dense(o, units=1, activation=None, name="logit")
        return policy_logits, old_policy_logits, values, old_values

    def get_vec_action_and_value(self, session, observation, eps_greedy=None):
        # take old policy probabilities for off-policy roll outs
        action_probs, values = session.run([self.old_policy_probs, self.old_values],
                                           feed_dict={self.observations: observation})
        # print(action_probs)
        actions = np.zeros(shape=(len(observation)), dtype=np.int32)
        for i in range(len(observation)):
            if eps_greedy:
                if np.random.random() < eps_greedy:
                    actions[i] = np.random.randint(self.n_actions)
                else:
                    actions[i] = np.argmax(action_probs[i])
            else:
                # print(action_probs)
                actions[i] = np.random.choice(self.n_actions, p=action_probs[i])
            if self.args.env == 'cart':
                actions = np.squeeze(actions)

        return np.array(actions), np.squeeze(values)


class GlobalModel(Model):
    def __init__(self, scope,
                 observation_shape,
                 n_actions,
                 args,
                 log_directory,
                 mode='training'):

        with tf.device("/device:GPU:1"):
            super().__init__(scope=scope,
                             observation_shape=observation_shape,
                             n_actions=n_actions,
                             args=args,
                             mode=mode)

        # self.writer_prefix = writer_prefix
        self.global_step = tf.train.create_global_step()
        self.epoch = tf.get_variable(shape=[], dtype=tf.int64, initializer=tf.zeros_initializer(),
                                     trainable=False, name='number_epoch')
        self.increment_epoch = tf.assign(self.epoch, self.epoch+1)

        self.learning_rate = tf.placeholder(dtype=tf.float32, shape=(), name='learning_rate')
        self.steps_in_env = tf.placeholder(dtype=tf.float32, shape=(), name='steps_in_env')

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.ratio = None
        self.clipped_loss = None
        self.value_loss = None
        self.approximate_kl_divergence = None
        self.mean_policy_ratio = None
        self.clipping_multiplier = tf.placeholder(dtype=tf.float32, shape=[], name='clipping_multiplier')

        self.total_loss = self.build_loss()
        self.grads = self.build_gradients()
        self.train_op = self.get_train_op()

        # paths
        self.tmp_path = log_directory
        # if args.master:
        #     self.tmp_path = os.path.join('/tmp/ppo', args.env, 'master_controller',
        #                                  args.algorithm, args.network+'_'+str(self.units))
        # elif args.n_players >= 2:
        #     self.tmp_path = os.path.join('/tmp/ppo', args.env, 'multi_agent_'+str(args.n_players), args.algorithm,
        #                                  args.network+'_'+str(self.units))
        # else:
        #     self.tmp_path = os.path.join('/tmp/ppo', args.env, 'single_agent', args.algorithm,
        #                                  args.network+'_'+str(self.units))

        self.checkpoints_path = os.path.join(self.tmp_path, 'ckpts')
        self.ckpt_file_path = self.checkpoints_path + '/checkpoint'

        # Summaries
        self.new_to_old_policy = U.update_target_graph(source=self.scope+'/policy', target=self.scope+'/old_policy')
        self.new_to_old_values = U.update_target_graph(source=self.scope+'/values', target=self.scope+'/old_values')
        self.saver = tf.train.Saver(max_to_keep=10)

        if mode == 'training':
            
            self.train_summary = self.merge_all_summaries()
            self.train_writer = tf.summary.FileWriter(self.tmp_path, tf.get_default_graph())
            self.eval_return = tf.placeholder(dtype=tf.float32, shape=[], name='average_return_epoch')
            self.eval_average_steps = tf.placeholder(dtype=tf.float32, shape=[], name='avr_steps_in_env')
            self.gsteps_per_sec = tf.placeholder(dtype=tf.float32, shape=[], name='gsteps_per_sec')
            with tf.variable_scope('evaluation'):
                sum_reward = tf.summary.scalar('average_return', self.eval_return)
                average_steps = tf.summary.scalar('average_steps_in_env', self.eval_average_steps)
                global_steps_per_sec = tf.summary.scalar('global_steps_per_sec', self.gsteps_per_sec)
                self.eval_summary = tf.summary.merge([sum_reward, average_steps, global_steps_per_sec])

                # ==== build summary for global step ======
                mean_reward = tf.summary.scalar('mean_return_gstep', self.eval_return)
                steps_in_env = tf.summary.scalar('steps_in_env', self.steps_in_env)
                self.g_step_summary = tf.summary.merge([mean_reward, steps_in_env])

        print('built global model')

    def build_loss(self):
        """
        Create Loss function as a Clipped Surrogate Objective
        see Schulman et al., Proximal Policy Optimization, 2017

        :return: total loss
        """
        with tf.variable_scope(self.scope +'/loss'):

            # Calculate log pi(a|s)
            new_log_pi = -tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.actions,
                                                                         logits=self.policy_logits)
            old_log_pi = -tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.actions,
                                                                         logits=self.old_policy_logits)
            # ratio = new_policy / old_policy = exp(log_new - log_old)
            ratio = tf.exp(new_log_pi - old_log_pi)
            self.approximate_kl_divergence = 0.5 * tf.reduce_mean(tf.square(new_log_pi - old_log_pi))
            self.mean_policy_ratio = tf.reduce_mean(ratio)

            # ===== Surrogate Loss ======
            # PPO's pessimistic surrogate (J^CLIP) => take the minimum as a pessimistic assumption
            # assign minus to loss surrogate to change maximization to a minimization problem
            clip_param = 0.2
            self.clip_value = clip_value = clip_param * self.clipping_multiplier
            # clip fraction in [0, 1]: how often is clipping applied to objective function?
            self.clip_fraction = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), clip_value)))

            un_clipped_loss = self.advantages * ratio
            clipped_loss = self.advantages * tf.clip_by_value(ratio, 1.0 - clip_value, 1.0 + clip_value)
            policy_surrogate = - tf.reduce_mean(tf.minimum(un_clipped_loss, clipped_loss))  # make minimization problem

            # ===== Value Loss + Entropy =====
            value_loss = 0.5 * tf.reduce_mean(tf.square(self.target_returns - self.values))
            self.entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.policy_probs,
                                                                      logits=self.policy_logits)
            entropy_loss = -0.01 * tf.reduce_mean(self.entropy)
            total_loss = policy_surrogate + value_loss + entropy_loss

        with tf.device('/cpu:0'):
            with tf.variable_scope('loss'):
                tf.summary.scalar('entropy_loss', entropy_loss)
                tf.summary.scalar('value_loss', value_loss)
                tf.summary.scalar('policy_surrogate_loss', policy_surrogate)
                tf.summary.scalar('total_loss', total_loss)
        return total_loss

    def build_gradients(self, max_grad_norm=0.5):
        """
        Get gradients by applying automatic differentiation

        :param max_grad_norm: float, clip gradients with respect to gradient norm
        :return: tf.gradients()
        """
        local_params = tf.trainable_variables()
        # build gradients except for old policy nodes
        grads = tf.gradients(self.total_loss, local_params, stop_gradients=self.old_policy_logits)
        # grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)

        # max_grad_norm = 0.5
        # max_grad_norm = None
        if max_grad_norm is not None:
            # Clip the gradients (normalize)
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        return grads

    def merge_all_summaries(self):
        """ Merge all defined summaries

        :return: tf.summary node
        """
        with tf.variable_scope('training'):
            tf.summary.scalar('clip_fraction', self.clip_fraction)
            tf.summary.scalar('mean_policy_ratio', self.mean_policy_ratio)
            tf.summary.scalar('approximate_KL_divergence', self.approximate_kl_divergence)
            tf.summary.scalar('clip_value', self.clip_value)
            tf.summary.scalar('learning_rate', self.learning_rate)
            tf.summary.scalar('entropy', tf.reduce_mean(self.entropy))
            tf.summary.scalar('global_steps', tf.reduce_mean(self.global_step))

        return tf.summary.merge_all()

    def get_train_op(self):

        local_params = tf.trainable_variables()
        grads_and_vars = list(zip(self.grads, local_params))

        if self.args.debug:
            for grad, var in grads_and_vars:
                if grad is not None:  # must be explicitely stated like this, to avoid error
                    tf.summary.histogram(var.name + '/gradient', grad)
        train_op = self.optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)
        return train_op

    def restore_from_checkpoint(self, session, directory):

        ckpt_file_path= os.path.join(directory, 'ckpts') + '/checkpoint'
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(ckpt_file_path))
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(session, ckpt.model_checkpoint_path)
            print('Restore model from {} ... successful!'.format(ckpt_file_path))
            step = self.global_step.eval()
            print('Global step:', step)
            print('Trained epochs:', self.epoch.eval())
            return step
        else:
            print('Restore model from {} ... failed!'.format(ckpt_file_path))
            print('run global init')
            session.run(tf.global_variables_initializer())
        return 0

    def save_as_checkpoint(self, session):
        self.saver.save(sess=session, save_path=self.ckpt_file_path, global_step=self.epoch)
        print('Saved model to:', self.checkpoints_path)

