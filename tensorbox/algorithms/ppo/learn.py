import numpy as np
import time
import tensorflow as tf
import os
import utils as U

from ppo.model import GlobalModel
import envs.vecenv
import pandas
from ppo.gae import calculate_gae_advantages, calculate_target_returns
from common.logger import CSVWriter


def get_default_parameters():
    return {
        'discount_factor': 0.99,
        'horizon': 1024,
        'learning_rate': 3.0e-4,
        'K': 1,
        'max_epochs': 500,
        'num_envs': 16
    }


def parameter_grid_search():
    """
    generator function that yields sets of parameters

    Perform a grid search on hyper parameters
    Graphs are store as event files that are plotted with tensor board

    Usage:
    for param_kws in parameter_search():
        print(param_kws)

    :return: yield, dict
    """
    learning_rates = [3.0e-4]
    update_cycles = [1]
    horizons = [1024]
    discount_factors = [0.99]
    num_envs = 16
    max_epochs = 1000
    number_runs_per_param_config = 5

    for hor in horizons:
        for update_cycle in update_cycles:
            for lr in learning_rates:
                for gamma in discount_factors:
                    for n in range(number_runs_per_param_config):
                        yield dict(horizon=hor, learning_rate=lr, K=update_cycle, run_id=n, create_checkpoints=False,
                                   save_interval=100, num_envs=num_envs, discount_factor=gamma, max_epochs=max_epochs)


class DataSet(object):
    """
    Stores sampled trajectories from environments, shuffles data, and provides mini-batches for SGD training
    """
    def __init__(self, a, b, c, d, batch_size=512):
        self.step = 0
        self.batch_size = batch_size
        assert len(a) == len(b) == len(c) == len(d), 'not even length'
        self.a, self.b, self.c, self.d = self.shuffle_data(a, b, c, d)

    def get_batch(self):
        self.step += 1
        if len(self.a) >= self.batch_size * self.step:
            low = (self.step - 1) * self.batch_size
            high = low + self.batch_size
            return self.a[low:high], self.b[low:high], self.c[low:high], self.d[low:high]
        else:
            raise IndexError

    @staticmethod
    def shuffle_data(a, b, c, d):
        p = np.random.permutation(len(a))
        return a[p], b[p], c[p], d[p]


def evaluate_current_model_parameters(session, env, model, n_players,
                                      evaluation_iterations=5, centralized_training=True):
    """
    Evaluate current model in environment

    :param session:
    :param env: Env(), environment in which to perform evaluation
    :param model:
    :param evaluation_iterations:
    :return:
    """
    mean_reward = []
    mean_steps = []

    def process_observation(obs):
        if n_players >= 2 and centralized_training:
            shape = tuple([-1]+env.observation_space.shape)
            return obs.reshape(shape)
        else:
            return obs[None, :]

    for iteration in range(evaluation_iterations):
        x = env.reset()
        done = False
        rewards = 0.0
        step = 0
        while not done:
            # act with epsilon-greedy policy (eps = 10%)
            actions, _ = model.get_vec_action_and_value(session=session,
                                                        observation=process_observation(x),
                                                        eps_greedy=None)
                                                        # eps_greedy=0.1)
            x, r, done, _ = env.step(actions)
            # env.render()
            step += 1
            rewards += r
        mean_reward.append(rewards)
        mean_steps.append(step)
    return np.mean(mean_reward), np.mean(mean_steps)


def get_dataset(sess, global_model, horizon, batch_size, server, writer, gamma, n_players=1, master_mode=False):
    """ run policy roll-outs to obtain a dataset for batch updating"""

    obs = server.reset()
    N = obs.shape[0]  # = num_envs * n_players
    actions = np.zeros(shape=(horizon, N), dtype=np.int64)
    obs_shape = tuple([horizon, N] + list(server.observation_space.shape))

    observations = np.zeros(shape=obs_shape, dtype=np.float64)
    rewards = np.zeros(shape=(horizon, N), dtype=np.float64)
    values = np.zeros(shape=(horizon, N), dtype=np.float64)
    target_returns = np.zeros(shape=(horizon, N), dtype=np.float64)
    advantages = np.zeros(shape=(horizon, N), dtype=np.float64)
    dones = np.zeros(shape=(horizon, N), dtype=np.float64)

    episode_rewards = []
    episode_steps = []

    for t in range(horizon):
        acs, vals = global_model.get_vec_action_and_value(session=sess, observation=obs)
        if master_mode or n_players == 1:
            actions_for_step = acs
        else:
            actions_for_step = acs.reshape((-1, n_players))  # reshape into [num_envs, n_players] for roll-outs

        new_obs, r, ds, infos = server.step(actions_for_step)
        i = t % horizon
        observations[i] = obs
        rewards[i] = r
        values[i] = vals
        actions[i] = acs
        dones[i] = ds
        obs = new_obs

        # get episode rewards from roll outs => calculate average reward over all trajectories
        for info in infos:
            if 'episode_reward' in info:
                episode_rewards.append(info['episode_reward'])
            if 'episode_steps' in info:
                episode_steps.append(info['episode_steps'])

    mean_dict = pandas.DataFrame(list(infos)).mean(axis=0).to_dict()
    writer.write(mean_dict)

    _, last_values = global_model.get_vec_action_and_value(session=sess, observation=obs)

    for n in range(N):
        adv = calculate_gae_advantages(rewards=rewards[:, n],
                                       values=values[:, n],
                                       dones=dones[:, n],
                                       last_value=last_values[n],
                                       gamma=gamma)
        target_r = calculate_target_returns(rewards=rewards[:, n],
                                            dones=dones[:, n],
                                            last_value=last_values[n],
                                            gamma=gamma)
        target_returns[:, n] = target_r
        advantages[:, n] = adv
    advantages = U.normalize(advantages)  # Normalize the advantages

    new_shape = tuple([-1] + list(observations.shape[2:]))  # Reshape and flatten stacked interactions
    o = observations.reshape(new_shape)  # shape into [horizon * N, obs.shape]
    acs = actions.flatten()
    adv = advantages.flatten()
    tars = target_returns.flatten()

    return DataSet(o, acs, adv, tars, batch_size=batch_size), U.safe_mean(episode_rewards), U.safe_mean(episode_steps)


def vec_training(parsed_args, horizon, K, learning_rate, num_envs, log_directory, schedule='decay', max_epochs=2000,
                 create_checkpoints=True, restore_model=False, save_interval=100, discount_factor=0.99,
                 batch_size=128, run_id=0, **settings):
    """
    Vectorized environments (parallel sub-processes roll out agent-environment interactions)
    Inference and train graph in one graph!

    :param parsed_args: parsed arguments from main function
    :param horizon: int, number of roll outs in episode
    :param K: int, update cycle of old policy, defined as K in Schulman et al., 2017
    :param learning_rate: float, step size of gradient optimizer
    :param schedule: str, 'decay' for annealing clipping rate or 'const'
    :param max_epochs: int, maximum of performed update epochs
    :param create_checkpoints: bool
    :param restore_model: bool
    :param save_interval: int, save model parameters every n steps
    :param num_envs: int, number of available CPU cores
    :param log_directory: string, path where log files are written to
    :param settings: dict, configuration parameters for setup, parsed to sub-functions
    :return:
    """
    print('==============================')
    print('Parsed Parameters for Training')
    print('============================== \n')
    print('Learning Rate:  {}'.format(learning_rate))
    print('K:              {}'.format(K))
    print('Horizon:        {}'.format(horizon))
    print('Discount:       {}'.format(discount_factor))
    print('Schedule:       {}'.format(schedule))
    print('Maximum Epochs: {}'.format(max_epochs))
    print('Cores:          {}'.format(num_envs))
    print('Restore Model:  {}'.format(restore_model))
    print('Create ckpts:   {}'.format(create_checkpoints))
    print('Save Interval:  {}'.format(save_interval))
    print('Run ID:         {}'.format(run_id))
    print('Agent Setting:  {}'.format(settings['agent_setting']))
    print('Log directory:  {}'.format(log_directory))
    print('\n==============================')
    args = parsed_args
    print('Parsed args: {}'.format(parsed_args))

    tf.reset_default_graph()
    train_graph = tf.Graph()

    with train_graph.as_default():

        # init and run vectorized environments trainer thread
        server = envs.vecenv.VecEnv(num_envs=num_envs, **settings)
        server = envs.vecenv.VecNormalize(server)  # normalize observations and rewards of env

        global_model = GlobalModel(scope='global',
                                   observation_shape=server.observation_space.shape,
                                   n_actions=server.action_space.n,
                                   args=args,
                                   log_directory=log_directory)

    writer = CSVWriter(dir=log_directory)
    tf_config = tf.ConfigProto(allow_soft_placement=True,  # soft placement to allow flexible training on CPU/GPU
                               intra_op_parallelism_threads=os.cpu_count(),  # speed up training time
                               inter_op_parallelism_threads=os.cpu_count())
    with tf.Session(graph=train_graph, config=tf_config) as sess:
        # if restore_model:
        #     global_model.restore_from_checkpoint(session=sess)
        # else:
        sess.run(tf.global_variables_initializer())
        server.setup()  # copy values from tf.graph into classes
        print('==============================')
        epoch = global_model.epoch.eval()
        steps_in_env = 0
        try:
            sess.run(global_model.new_to_old_policy)  # after init: copy new policy params to old policy
            sess.run(global_model.new_to_old_values)

            while True:
                start_time = time.time()
                previous_g_step = global_model.global_step.eval()

                if epoch >= max_epochs:
                    break
                if epoch % K == 0:  # copy new policy params to old policy
                    sess.run(global_model.new_to_old_policy)
                    sess.run(global_model.new_to_old_values)

                # perform policy roll outs and pack them into data set
                ds, mean_reward, mean_steps = get_dataset(sess=sess,
                                                          global_model=global_model,
                                                          horizon=horizon,
                                                          batch_size=batch_size,
                                                          server=server,
                                                          master_mode=args.master,
                                                          gamma=discount_factor,
                                                          n_players=args.n_players,
                                                          writer=writer)
                steps_in_env += horizon * num_envs
                # perform policy updates
                while True:
                    try:
                        o, acs, adv, tars = ds.get_batch()
                        feed = {
                            global_model.observations: o,
                            global_model.actions: acs,
                            global_model.advantages: adv,
                            global_model.target_returns: tars,
                            global_model.clipping_multiplier: (1.0 - epoch/max_epochs),
                            global_model.learning_rate: learning_rate * (1.0 - epoch/max_epochs)
                        }
                        _, train_summary = sess.run([global_model.train_op, global_model.train_summary], feed_dict=feed)
                    except IndexError:
                        break

                # ====== Evaluation and debug prints ======
                sess.run(global_model.increment_epoch)
                current_g_step = global_model.global_step.eval()
                epoch = global_model.epoch.eval()
                train_time = time.time() - start_time
                gsteps_per_sec = (current_g_step - previous_g_step) / train_time
                global_model.train_writer.add_summary(summary=train_summary,
                                                      global_step=epoch)
                print('Trained epoch {} \t took: {:0.2f} secs \t Mean reward: {:0.2f}, \t mean steps: {:0.2f}'.format(
                    epoch, train_time, mean_reward, mean_steps))
                eval_summary = sess.run(global_model.eval_summary,
                                        feed_dict={global_model.eval_return: mean_reward,
                                                   global_model.eval_average_steps: mean_steps,
                                                   global_model.gsteps_per_sec: gsteps_per_sec
                                                   })
                global_model.train_writer.add_summary(summary=eval_summary,
                                                      global_step=epoch)
                g_step_summary = sess.run(global_model.g_step_summary,
                                          feed_dict={global_model.eval_return: mean_reward,
                                                     global_model.steps_in_env: steps_in_env})
                global_model.train_writer.add_summary(summary=g_step_summary,
                                                      global_step=global_model.global_step.eval())

                if epoch % save_interval == 0 and create_checkpoints:
                    global_model.save_as_checkpoint(session=sess)

        except KeyboardInterrupt:
            print('vec_training(): got KeyboardInterrupt')
        finally:
            if create_checkpoints:
                global_model.save_as_checkpoint(session=sess)
            writer.close()
            server.close()


def play(args, **settings):
    """
    Restore model parameters from checkpoint file and outputs game play as graphical output
    :param parsed_args:
    :param settings:
    :return:
    """

    tf.reset_default_graph()
    print(args.dir)
    assert args.dir, 'Please provide directory where checkpoint file is located'

    env = U.make_env(**settings)

    def process_observation(obs):

        if settings['n_players'] == 1 or settings['master_controller']:
            return obs[None, :]
            # shape = tuple([-1]+env.observation_space.shape)
            # return obs.reshape(shape)
        else:
            return obs

    global_model = GlobalModel(scope='global',
                               observation_shape=env.observation_space.shape,
                               n_actions=env.action_space.n,
                               args=args,
                               mode='testing',
                               log_directory='/tmp')
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=tf_config) as sess:

        global_step = global_model.restore_from_checkpoint(session=sess, directory=args.dir)
        assert global_step > 0, 'Did not load model'

        env.setup()
        x = env.reset()
        done = False
        info = dict()
        count = np.zeros(5)
        step = 0

        while not done:
            env.render()
            actions, value = global_model.get_vec_action_and_value(sess, process_observation(x), eps_greedy=None)

            x, reward, dones, info = env.step(actions)

            count[actions] += 1
            if settings['n_players'] >= 2:
                done = np.array(dones).any()
            else:
                done = dones

    print('Took {} steps'.format(step))
    print('Info:', info)
    print('Count:', count)
