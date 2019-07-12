import numpy as np
import gym
import matplotlib.pyplot as plt
from tensorbox.common.pid_controller import PIDController
from abc import ABC, abstractmethod


class BaseSignalGenerator(ABC):
    def __init__(self, frequency=512, *args):
        self.frequency = frequency
        self.counter = 0

    def __call__(self, *args, **kwargs):
        return self.get_next_value()

    def __next__(self):
        return self.get_next_value()

    @abstractmethod
    def get_next_value(self):
        pass


class JumpSignalGenerator(BaseSignalGenerator):
    def __init__(self):
        super(JumpSignalGenerator, self).__init__()
        self.magnitude = 1.

    def get_next_value(self):
        value = 0. if self.counter <= 2 / 2 else 1.
        self.counter += 1
        return value * self.magnitude


class SawtoothSignalGenerator(BaseSignalGenerator):
    def __init__(self):
        super(SawtoothSignalGenerator, self).__init__()
        self.magnitude = 1.
        self.signal_slope = 2. * self.magnitude / self.frequency

    def get_next_value(self):
        it = (self.counter + self.frequency // 2) % self.frequency
        self.counter = (self.counter + 1) % self.frequency
        return it * self.signal_slope - self.magnitude


class RectifierSignalGenerator(BaseSignalGenerator):
    def __init__(self):
        super(RectifierSignalGenerator, self).__init__()
        self.magnitude = 5.

    def get_next_value(self):
        value = 0. if self.counter <= self.frequency // 2 else 1.
        self.counter = (self.counter + 1) % self.frequency
        return value * self.magnitude


class BasePlant(object):
    def __init__(self, dt):
        # self.input = 0.
        self.output = 0.
        self.dt = dt  # time interval


class PT1Plant(BasePlant):
    def __init__(self, dt, time_constant=.15):
        super(PT1Plant, self).__init__(dt=dt)
        self.time_constant = time_constant

    def __call__(self, *args, **kwargs):
        return self.update(*args, **kwargs)

    def update(self, input):
        # input ~ error * pid_controller
        # diff = input - self.output
        self.output += input / self.time_constant * self.dt
        return self.output


class PT2Plant(BasePlant):
    def __init__(self, dt, T1=.03, T2=.4):
        super(PT2Plant, self).__init__(dt=dt)
        self.T1 = T1
        self.T2 = T2

    def __call__(self, *args, **kwargs):
        return self.update(*args, **kwargs)

    def update(self, input):
        # input ~ error * pid_controller
        # diff = input - self.output
        self.output += input / self.T1 * self.dt
        self.output += input / self.T2 * self.dt
        return self.output


def get_plant_from_name(name, dt):
    if name == 'pt1':
        return PT1Plant(dt=dt)
    if name == 'pt2':
        return PT2Plant(dt=dt)
    else:
        raise ValueError('No Plant named `{}`'.format(name))


def get_signal_from_name(signal_name):
    if signal_name == 'rectifier':
        return RectifierSignalGenerator()
    else:
        raise ValueError('Did not find signal `{}`'.format(signal_name))


class ClosedLoopSystemEnv(gym.Env):
    def __init__(self, dt=.01, signal='rectifier', plant='pt2'):
        super(ClosedLoopSystemEnv, self).__init__()
        self.dt = dt
        self.plant = get_plant_from_name(plant, dt)
        self.reference_generator = get_signal_from_name(signal)
        # self.reference_generator = SawtoothSignalGenerator()
        self.iteration = 0
        self.horizon = 1024
        self.endless_mode = True

        self.y = 0.  # output value
        self.w = 0.  # reference value
        self.error = 0.

        # self.pid = PIDController(k_p=3., k_i=1., k_d=0.02)
        self.pid = PIDController(k_p=5., k_i=1.2, k_d=0.1)

        self.error_bound = 10.
        output_limit = 10.
        k_p_limit = 5.
        k_i_limit = 5.
        k_d_limit = 5.
        # Upper and lower limit for observation space
        self.o_high = np.array([self.error_bound,  # e: error limit
                                output_limit,  # y: output limit
                                self.pid.windup_limit,
                                k_p_limit,  # k_p limit
                                k_i_limit,  # k_i limit
                                k_d_limit  # k_d limit
                                ])
        self.o_low = np.array([-self.error_bound,  # e: error limit
                               -output_limit,  # y: output limit
                               -self.pid.windup_limit,
                               -k_p_limit,  # k_p limit
                               -k_i_limit,  # k_i limit
                               -k_d_limit  # k_d limit
                               ])
        # Upper and lower limit for action space
        self.a_high = np.array([1., 1., 1.])
        self.a_low = np.array([-1., -1., -1.])

        # Create spaces
        self.observation_space = gym.spaces.Box(self.o_low, self.o_high, dtype=np.float32)
        self.action_space = gym.spaces.Box(self.a_low, self.a_high, dtype=np.float32)

    @property
    def state(self):
        return np.concatenate(([self.error, self.y, self.pid.sum_of_errors], self.pid.parameters))

    def step(self, ac):
        new_params = ac
        self.pid.set_from_policy_mean(new_params)

        self.w = self.reference_generator()
        self.error = np.clip(self.w - self.y,
                             -self.error_bound,
                             self.error_bound)  # clip error bound
        u = self.pid(self.error)
        self.y = self.plant.update(u)

        self.iteration += 1
        r = 1. / np.e**np.abs(10*self.error)
        done = False if self.endless_mode else self.iteration >= self.horizon
        info = dict()
        return self.state, r, done, info

    def reset(self):
        self.iteration = 0
        return self.state

    def render(self, mode='human'):
        pass


if __name__ == '__main__':
    env = ClosedLoopSystemEnv()
    x = env.reset()
    horizon = 2048
    done = False
    ret = 0.
    i = 0
    ws = np.zeros(horizon)
    ys = np.zeros(horizon)
    ts = np.arange(horizon)
    while not done and i < horizon:
        action = np.array([0., 0, 0])
        ws[i] = env.w
        ys[i] = env.y
        x, reward, done, _ = env.step(action)
        ret += reward
        i += 1
    print('Return:', ret)

    print('Plot jump')

    plt.figure()
    plt.plot(ts, ws, 'r-', ts, ys, 'b-')
    plt.show()
    env.close()

