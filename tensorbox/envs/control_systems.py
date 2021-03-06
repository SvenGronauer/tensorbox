import numpy as np
import gym
import matplotlib.pyplot as plt
from tensorbox.common.pid_controller import PIDController
from abc import ABC, abstractmethod
import unittest


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
    def __init__(self, jump_time, dt):
        super(JumpSignalGenerator, self).__init__()
        self.magnitude = 1.
        self.dt = dt
        self.jump_time = jump_time
        self.time = 0.

    def get_next_value(self):
        self.time += self.dt
        value = 1. if self.time >= self.jump_time else 0.
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


class StateModel:
    def __init__(self,
                 dt,
                 x_init,
                 A,
                 b,
                 c,
                 d):
        self.x = x_init
        self.A = A
        self.b = b
        self.c = c
        self.d = d
        self.dt = dt
        self.current_u = None

    def __call__(self, u, *args, **kwargs):
        return self.update_internal_state(u)

    def update_internal_state(self, u):
        self.current_u = u
        dx = (self.A @ self.x + self.b * u) * self.dt
        self.x += dx
        return self.y

    @property
    def y(self):
        y = self.c.T @ self.x + self.d * self.current_u
        return np.squeeze(y)


class PT1Plant(StateModel):
    def __init__(self,
                 x_init=np.array([0.]),
                 A=np.array([-10]),
                 b=np.array([10]),
                 c=np.array([1.]),
                 d=np.array([0.]),
                 dt=0.01):
        """
        R = 100  # resistance in Ohm
        C = 1.0e-3  # capacity in H
        T = 1 / (R * C) = 10.0
        """
        super(PT1Plant, self).__init__(dt, x_init, A, b, c, d)

    @property
    def y(self):
        y = super().y  # squeeze 1D-array to float number
        return np.squeeze(y)


class PT2Plant(StateModel):
    def __init__(self,
                 dt,
                 x_init=np.array([0., 0.]),
                 A=np.array([[-75, -10],    # -R/L, -1/L
                             [100., 0.]]),  # 1/C,  0
                 b=np.array([10, 0.]),
                 c=np.array([0., 1.]),
                 d=np.array([0.])):
        """
        RLC Network, where R = 7.5 , C = 1e-2, L = 0.1
        => T_1 = 0.0577, T_2 = 0.0173
        from: Unbehauen, Regelungstechnik I, p. 47
        for asymptotic stability: R >= 2 * sqrt(L/C)
        """
        super(PT2Plant, self).__init__(dt, x_init, A, b, c, d)


class StateSystemEnv(gym.Env):
    def __init__(self,
                 dt,
                 clip_error=False,
                 pid=None,
                 *args,
                 **kwargs):
        """

        :param dt: float, time step in seconds
        :param args:
        """
        super(StateSystemEnv, self).__init__()
        self.reference_generator = JumpSignalGenerator(jump_time=0.3, dt=dt)
        self.plant = PT2Plant(dt=dt)
        self.dt = dt
        # self.reference_generator = SawtoothSignalGenerator()
        self.iteration = 0
        self.horizon = 1024
        self.endless_mode = True
        self.clip_error = clip_error
        self.error = 0.
        self.y = 0.
        self.w = 0.

        # self.pid = PIDController(k_p=3., k_i=1., k_d=0.02)
        if pid is None:
            self.pid = PIDController(dt, k_p=25., k_i=25, k_d=.0,
                                     limit_parameters=False,
                                     anti_windup=False)
        else:
            self.pid = pid
        # self.pid = PIDController(k_p=.7, k_i=1.4, k_d=0.)

        self.error_bound = 10.
        output_limit = 10.
        parameter_observation_limit = 5.
        # Upper and lower limit for observation space
        self.o_high = np.array([self.error_bound,  # e: error limit
                                output_limit,  # y: output limit
                                self.pid.windup_limit,
                                parameter_observation_limit,  # k_p limit
                                parameter_observation_limit,  # k_i limit
                                parameter_observation_limit  # k_d limit
                                ])
        self.o_low = np.array([-self.error_bound,  # e: error limit
                               -output_limit,  # y: output limit
                               -self.pid.windup_limit,
                               -parameter_observation_limit,  # k_p limit
                               -parameter_observation_limit,  # k_i limit
                               -parameter_observation_limit  # k_d limit
                               ])
        # Upper and lower limit for action space
        self.a_high = np.array([5., 5., 5.])
        self.a_low = np.array([-5., -5., -5.])

        # Create spaces
        self.observation_space = gym.spaces.Box(self.o_low, self.o_high, dtype=np.float32)
        self.action_space = gym.spaces.Box(self.a_low, self.a_high, dtype=np.float32)

    @property
    def state(self):
        return np.concatenate(([self.error, self.y, self.pid.sum_of_errors],
                               self.pid.get_scaled_parameters()))

    def step(self, ac):
        new_params = ac
        self.pid.set_from_policy_mean(new_params)

        self.w = self.reference_generator()
        self.error = self.w - self.y
        if self.clip_error:   # clip error bound
            self.error = np.clip(self.error,
                                 -self.error_bound,
                                 self.error_bound)
        u = self.pid(self.error)
        self.y = self.plant(u)

        self.iteration += 1
        r = 1. / np.e**np.abs(10*self.error)
        d = False if self.endless_mode else self.iteration >= self.horizon
        info = dict()
        return self.state, r, d, info

    def reset(self):
        self.iteration = 0
        return self.state

    def render(self, mode='human'):
        pass


class PT2SystemEnv(StateSystemEnv):
    def __init__(self, dt=0.001, *args, **kwargs):
        pid = PIDController(dt=dt,
                            k_p=25.,
                            k_i=25.,
                            k_d=0.,
                            limit_parameters=True,
                            anti_windup=False)
        # adjust limits and scales for PID control of PT-2 Plant
        pid.max_limits_parameters = np.array([50., 50., 0.])
        pid.min_limits_parameters = np.array([0., 0., 0.])
        pid.scales = np.array([5., 5., 1.])
        super(PT2SystemEnv, self).__init__(dt=dt,
                                           pid=pid,
                                           *args, **kwargs)
        self.reference_generator = JumpSignalGenerator(jump_time=0.3, dt=dt)
        self.plant = PT2Plant(dt=dt)


class PT1SystemEnv(StateSystemEnv):
    def __init__(self, dt=0.001, *args, **kwargs):
        super(PT1SystemEnv, self).__init__(dt, *args, **kwargs)
        self.reference_generator = JumpSignalGenerator(jump_time=0.3, dt=dt)
        self.plant = PT1Plant(dt=dt)


if __name__ == '__main__':
    # test_state_model()
    dt = 0.001
    env = PT2SystemEnv(dt=dt, clip_error=False)
    x = env.reset()
    stop_time = 3.
    done = False
    ret = 0.
    i = 0
    ts = np.arange(0, stop_time, dt)
    ws = np.zeros_like(ts)
    ys = np.zeros_like(ts)
    while not done and i < len(ts):
        action = np.array([0., 0, 0])
        ws[i] = env.w
        ys[i] = env.y
        x, reward, done, _ = env.step(action)
        ret += reward
        i += 1
    print('Return:', ret)

    print('Plot jump')
    #
    plt.figure()
    plt.plot(ts, ws, 'r-', ts, ys, 'b-')
    plt.show()
    env.close()

