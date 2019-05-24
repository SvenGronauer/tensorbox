import tensorflow as tf
from multiprocessing import Process, Queue
import os
import datetime
import numpy as np
from impala.model import ImpalaModel, Learner, Actor
from impala.replay_buffer import UniformBuffer, PrioritizedBuffer
from common.logger import TensorBoardWriter, CSVWriter, CombinedWriter
import utils as U
from impala.py_process import PyProcessHook


def get_default_parameters():
    return {
        'batch_size': 2,
        'entropy_scale': 0.1,
        'horizon': 256,
        'learning_rate': 2.0e-4,
        'max_steps': 50000,
        'rho_clip': 2.0,
        'sequence_length': 128
    }


def parameter_grid_search():
    """
    generator function that yields sets of parameters

    Usage:
    for param_kws in parameter_search():
        print(param_kws)

    :return: yield, dict
    """
    horizons = [256]
    batch_sizes = [16]
    sequence_lengths = [16]
    learning_rates = [2.0e-4, 4.0e-4]
    entropy_scales = [1e-1, 1e-2, 1e-3]

    for bs in batch_sizes:
        for hor in horizons:
            for seq_len in sequence_lengths:
                for es in entropy_scales:
                    for lr in learning_rates:
                        yield dict(batch_size=bs, entropy_scale=es, horizon=hor,
                                   learning_rate=lr, sequence_length=seq_len)


class NormalizeObservationsRewards:
    def __init__(self, observation_space, clip_value=10.0, epsilon=1e-8):
        self.obs_shape = observation_space.shape
        self.ob_rms = U.TfRunningMeanStd(shape=observation_space.shape, scope='RunningMeanStd/Obs')
        self.ret_rms = U.TfRunningMeanStd(shape=(), scope='RunningMeanStd/Rew')
        self.clip = clip_value
        self.epsilon = epsilon

    def normalize_and_update(self, obs, rewards):
        """ normalize inputs and update internal running mean/std parameters """

        return self._ob_filter(obs), self._reward_filter(rewards.flatten())

    def normalize(self, obs, rewards, update_internal_with_session=None):
        """ only normalize inputs and rewards """
        if update_internal_with_session:
            self.get_values_from_tf_graph(session=update_internal_with_session)
        flatten_obs = obs.reshape((-1, *self.obs_shape))
        ob = np.clip((flatten_obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clip, self.clip)
        rew = np.clip((rewards.flatten() - self.ret_rms.mean) / np.sqrt(self.ret_rms.var + self.epsilon), -self.clip, self.clip)
        return ob, rew

    def normalize_observation(self, obs):
        """ only normalize inputs"""
        return np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clip, self.clip)

    def _ob_filter(self, obs):
        # flatten observations for calculating mean and std along axis=0
        flatten_obs = obs.reshape((-1, *self.obs_shape))

        self.ob_rms.update(flatten_obs)
        normed_flat_obs = np.clip((flatten_obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clip, self.clip)
        return normed_flat_obs

    def _reward_filter(self, rewards):
        self.ret_rms.update(rewards)
        rew = np.clip((rewards - self.ret_rms.mean) / np.sqrt(self.ret_rms.var + self.epsilon), -self.clip, self.clip)
        return rew

    def setup(self, session=None):
        """ get_values_from_tf_graph and copy into classes"""
        self.ob_rms.get_values_from_tf_graph(session)
        self.ret_rms.get_values_from_tf_graph(session)

    def get_values_from_tf_graph(self, session=None):
        """ get_values_from_tf_graph and copy into classes"""
        self.ob_rms.get_values_from_tf_graph(session)
        self.ret_rms.get_values_from_tf_graph(session)

    def update_ops(self, obs, rewards, session):
        flatten_obs = obs.reshape((-1, *self.obs_shape))
        if not session.should_stop():
            self.ob_rms.update(flatten_obs, session=session)
            self.ret_rms.update(rewards.flatten(), session=session)


def training(cluster,
             job_name,
             task_index,
             queue,
             kwargs,
             horizon,
             sequence_length,
             learning_rate,
             entropy_scale,
             max_steps,
             batch_size):
    """
    Trains the architecture

    learner updates the parameters of the NN according to Actor-Critic
    actors roll-out policy and put trajectories into queue
    :param cluster:
    :param job_name:
    :param task_index:
    :param queue:
    :param kwargs:
    :param horizon:
    :param sequence_length:
    :param learning_rate:
    :param entropy_scale:
    :param max_steps:
    :param batch_size:
    :return:
    """
    print('==============================')
    print('Parsed Parameters for Training')
    print('============================== \n')
    print('Learning Rate:   {}'.format(learning_rate))
    print('Horizon:         {}'.format(horizon))
    print('Sequence Length: {}'.format(sequence_length))
    print('Entropy Scale:   {}'.format(entropy_scale))
    print('Maximum Steps:   {}'.format(max_steps))
    print('\n==============================')
    print('Started server: job_name={}, task_index={}'.format(job_name, task_index))
    date_string = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M")
    log_directory = os.path.join('/tmp/impala', date_string)

    is_chief = (job_name == 'learner')
    server = tf.train.Server(cluster, job_name=job_name, task_index=task_index)

    device_name = '/job:{}/task:{}'.format(job_name, task_index)
    print('Place on tf model on device:', device_name)
    with tf.device(tf.train.replica_device_setter(worker_device=device_name, cluster=cluster)):
        # create model ...
        env = U.make_env(**kwargs)  # share running mean ops across devices
        normalizer = NormalizeObservationsRewards(observation_space=env.observation_space)

        if is_chief:  # only learners needs to build loss
            with tf.device('/gpu'):
                model = ImpalaModel(observation_shape=env.observation_space.shape,
                                    n_actions=env.action_space.n,
                                    learning_rate=learning_rate,
                                    entropy_scale=entropy_scale)
                model.build_loss()
                model.build_trainer()
                trajectory_buffer = PrioritizedBuffer(obs_shape=env.observation_space.shape,
                                                     batch_size=batch_size,
                                                     horizon=horizon,
                                                     sequence_length=sequence_length,
                                                     size=1000)
                logs = CombinedWriter(dir=log_directory)
                print('Logging to', log_directory)
                U.dump_dict_as_json(kwargs, directory=log_directory, file_name='configuration')
            worker = Learner(model=model, queue=queue, buffer=trajectory_buffer, logger=logs, norm=normalizer, **kwargs)
        else:
            with tf.device('/cpu'):  # pin workers to CPU
                model = ImpalaModel(observation_shape=env.observation_space.shape,
                                    n_actions=env.action_space.n,
                                    learning_rate=learning_rate,
                                    entropy_scale=entropy_scale)
            worker = Actor(model=model, env=env, queue=queue, normalizer=normalizer, **kwargs)

    # The StopAtStepHook handles stopping after running given steps.
    # max_steps = 10000
    hooks = [tf.train.StopAtStepHook(last_step=max_steps)]  # , PyProcessHook()]

    # TODO adjust tf.Config for flexible node placement on GPUs
    tf_config = tf.ConfigProto(allow_soft_placement=True,  # soft placement to allow flexible training on CPU/GPU
                               intra_op_parallelism_threads=1,  # speed up training time
                               inter_op_parallelism_threads=1)  # number of physical cores
    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when done
    # or an error occurs.
    with tf.train.MonitoredTrainingSession(master=server.target,
                                           is_chief=is_chief,
                                           checkpoint_dir=os.path.join("/tmp/impala/", date_string),
                                           config=tf_config,
                                           save_checkpoint_secs=120,
                                           hooks=hooks) as mon_sess:
        normalizer.setup(session=mon_sess)

        while not mon_sess.should_stop():
            # learner batches from experience buffer and updates policy network
            # actors only enqueue trajectories into the FIFO queue
            worker.work(session=mon_sess)

        print('{}:{} wants to join ... Training finished!'.format(job_name, task_index))
        if is_chief:
            logs.close()
            server.join()


def play(args, **kwargs):
    """ play mode """

    print(args.dir)
    assert args.dir, 'Please provide directory where checkpoint file is located'

    kwargs['normalize'] = True
    normed_env = U.make_env(**kwargs)  # use env.setup() after session creation to apply mean/std to obs and rewards

    model = ImpalaModel(observation_shape=normed_env.observation_space.shape,
                        n_actions=normed_env.action_space.n, learning_rate=0.01, entropy_scale=0.0)

    # max_steps = 10000
    # hooks = [tf.train.StopAtStepHook(last_step=max_steps)]  # , PyProcessHook()]

    print('Restore from:', args.dir)
    with tf.train.SingularMonitoredSession(checkpoint_dir=args.dir) as sess:

        normed_env.setup(session=sess)  # restore values for running mean/std
        print('Restored from global step:', sess.run(model.global_step))

        try:
            done = False
            obs = normed_env.reset()
            print(obs)

            while not done:
                normed_env.render()
                action, _ = model.get_action_and_prob(session=sess, observation=obs)
                obs, reward, done, info = normed_env.step(action)

        except KeyboardInterrupt:
            print('got KeyboardInterrupt')
        finally:
            pass


def main(args, **kwargs):
    print('--> Using the following configuration:')
    print(kwargs)
    num_actors = 2

    cluster = tf.train.ClusterSpec({
        "worker": ['localhost:{}'.format(8000 + i) for i in range(num_actors)],
        "learner": ["localhost:9000"]
    })

    queue = Queue(maxsize=100)

    bs = kwargs['batch_size']
    horizon = kwargs['horizon']
    lr = kwargs['learning_rate']
    es = kwargs['entropy_scale']
    max_steps = kwargs['max_steps']
    seq_len = kwargs['sequence_length']

    processes = []
    # define processes as daemon so that children terminate when parent crashes
    params = (cluster, 'learner', 0, queue, kwargs, horizon, seq_len, lr, es, max_steps, bs)
    p = Process(target=training, args=params)
    p.daemon = True
    p.start()
    processes.append(p)

    for actor_id in range(num_actors):  # create worker processes
        params = (cluster, 'worker', actor_id, queue, kwargs, horizon, seq_len, lr, es, max_steps, bs)
        p = Process(target=training, args=params)
        p.daemon = True
        p.start()
        processes.append(p)
    print('ALL PROCESSES STARTED')
    # time.sleep(5)
    for p in processes:
        p.join()
    print('ALL JOINED')


if __name__ == '__main__':

    shared_job_device = '/job:learner/task:0'
    main(shared_job_device)
