import numpy as np
import multiprocessing as mp
import tensorbox.common.utils as utils
import time
from abc import ABC, abstractmethod


def worker(worker_remote, parent_remote, env_name, config):
    """
    Worker that runs an RL environment as sub-process
    Communication to parent process (server) via pipes

    :param worker_remote: mp.Pipe(), communication channel
    :param parent_remote: mp.Pipe(), to be closed at beginning of this function
    :param env_name:
    :param config: dict()
    :return:
    """

    parent_remote.close()  # child still has a copy open and the underlying kernel object is not being released.
    config['seed'] = np.int32(time.time()*100)  # place random seed for training
    env = utils.make_env(env_name, **config)

    try:
        while True:
            cmd, data = worker_remote.recv()
            if cmd == 'step':
                ob, reward, done, info = env.step(data)
                if done or 'episode_reward' in info:  # check if horizon of episode is reached
                        ob = env.reset()
                worker_remote.send((ob, reward, done, info))
            elif cmd == 'reset':
                ob = env.reset()
                worker_remote.send(ob)
            elif cmd == 'render':
                worker_remote.send(env.render(mode='rgb_array'))
            elif cmd == 'close':
                worker_remote.close()
                break
            elif cmd == 'get_spaces':
                print('env.observation_space', env.observation_space)
                worker_remote.send((env.observation_space, env.action_space))
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')
    finally:
        env.close()


class VecEnv(object):
    """
    VecEnv that runs multiple environments in parallel in sub-proceses and communicates with them via pipes.
    Recommended to use when num_envs > 1 and step() can be a bottleneck.

    source: https://github.com/openai/baselines/blob/master/baselines/common/vec_env/__init__.py
    """
    def __init__(self, env_name, num_envs=4, observation_space=None, action_space=None, **config):
        """
        Arguments:
        env_fns: iterable of callables -  functions that create environments to run in subprocesses. Need to be cloud-pickleable
        """
        self.waiting = False
        self.closed = False
        # reader, writer = Pipe()
        self.remotes, self.work_remotes = zip(*[mp.Pipe(duplex=True) for _ in range(num_envs)])

        # self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
        self.ps = [mp.Process(target=worker, args=(work_remote, remote, env_name, config))
                   for (work_remote, remote) in zip(self.work_remotes, self.remotes)]

        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
            time.sleep(0.05)  # sleep some milliseconds to force different seeds in environments
        for remote in self.work_remotes:
            remote.close()

        if observation_space and action_space:
            self.observation_space = observation_space
            self.action_space = action_space
        else:   # determine spaces from environment
            self.remotes[0].send(('get_spaces', None))
            self.observation_space, self.action_space = self.remotes[0].recv()
        print('observation_space {} of type {}'.format(self.observation_space.shape, self.observation_space))
        print('action_space n={} of type {}'.format(self.action_space.n, self.action_space))
        self.viewer = None
        self.num_envs = num_envs

    def close(self):
        if self.closed:
            return
        if self.viewer is not None:
            self.viewer.close()
        self.close_extras()

        if mp.active_children() is not []:  # check for active child processes
            print('Active children:')
            print(mp.active_children())
        self.closed = True
        print('Server closed')

    def get_viewer(self):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.SimpleImageViewer()
        return self.viewer

    def setup(self):
        pass

    def step(self, actions):
        """
        Step the environments synchronously.
        This is available for backwards compatibility.
        """
        self.step_async(actions)
        return self.step_wait()

    def step_async(self, actions):
        """
        Step asynchronously in each environment

        :param actions: list or np.array, holding action for each environment
        :raises: AssertionError, if pipes are already closed
        :return:
        """
        self._assert_not_closed()
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        self._assert_not_closed()
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)

        flattend_rew = np.array(rews).flatten()
        flatten_dones = np.array(dones).flatten()
        return np.vstack(obs), flattend_rew, flatten_dones, infos

    def reset(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('reset', None))
        v_stacked_obs = np.vstack([remote.recv() for remote in self.remotes])
        return v_stacked_obs

    def close_extras(self):
        self.closed = True
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    def get_images(self):
        self._assert_not_closed()
        for pipe in self.remotes:
            pipe.send(('render', None))
        imgs = [pipe.recv() for pipe in self.remotes]
        return imgs

    def _assert_not_closed(self):
        assert not self.closed, "Trying to operate on a SubprocVecEnv after calling close()"


class VecEnvWrapper(VecEnv):
    """
    An environment wrapper that applies to an entire batch
    of environments at once.
    """

    def __init__(self, venv):
        self.venv = venv
        self.num_envs = venv.num_envs,
        # self.observation_space = venv.observation_space,
        # self.action_space = venv.action_space

        self.action_space = venv.action_space
        self.observation_space = venv.observation_space

    def step_async(self, actions):
        self.venv.step_async(actions)

    @abstractmethod
    def reset(self):
        pass

    def close(self):
        return self.venv.close()

    def render(self, mode='human'):
        return self.venv.render(mode=mode)

    def get_images(self):
        return self.venv.get_images()


class VecNormalize(VecEnvWrapper):
    """
    A vectorized wrapper that normalizes the observations
    and returns from an environment.
    """

    def __init__(self, venv, ob=True, ret=True, clip=10., gamma=0.99, epsilon=1e-8):
        print(venv.action_space)
        print('init VecNormalize')
        VecEnvWrapper.__init__(self, venv)
        self.ob_rms = utils.TfRunningMeanStd(shape=self.observation_space.shape, scope='RunningMeanStd/Obs') if ob else None
        self.ret_rms = utils.TfRunningMeanStd(shape=(), scope='RunningMeanStd/Rew') if ret else None
        self.clip = clip
        self.ret = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon

    def run_tf_init(self, sess):
        """
        Setup mean and variance as TF nodes
        :return:
        """
        if isinstance(self.ob_rms, utils.TfRunningMeanStd):
            self.ob_rms.run_tf_init(sess)
        if isinstance(self.ret_rms, utils.TfRunningMeanStd):
            self.ret_rms.run_tf_init(sess)

    @DeprecationWarning
    def get_values_from_tf_graph(self):
        # s\ess = tf.get_default_graph()
        self.ob_rms.get_values_from_tf_graph()
        self.ret_rms.get_values_from_tf_graph()

    def setup(self):
        """ get_values_from_tf_graph and copy into classes"""
        self.ob_rms.get_values_from_tf_graph()
        self.ret_rms.get_values_from_tf_graph()

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()

        normed_obs = self._obfilt(obs)
        normed_rewards = self._reward_filter(rews)
        return normed_obs, normed_rewards, news, infos

    def _obfilt(self, obs):
        if self.ob_rms:
            self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clip, self.clip)
            return obs
        else:
            return obs

    def _reward_filter(self, rewards):
        self.ret_rms.update(rewards)
        rew = np.clip((rewards - self.ret_rms.mean) / np.sqrt(self.ret_rms.var + self.epsilon), -self.clip, self.clip)
        return rew

    def reset(self):
        self.ret = np.zeros(self.num_envs)
        obs = self.venv.reset()
        return self._obfilt(obs)


if __name__ == '__main__':
    print('Test VEC ENV')
    # env = utils.make_env('fruit')

    server = VecEnv(env_name='fruit')
    ob = server.reset()
    print(ob)

    d = False
    while not d:
        actions = np.random.randint(low=0, high=2, size=(2,))
        ob, reward, done, info = server.step(actions)
        print(ob, reward, done, info)
        d = np.any(done)
    server.close()
