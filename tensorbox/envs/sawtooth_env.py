import numpy as np
import gym
import matplotlib.pyplot as plt
from tensorbox.common.pid_controller import PIDController


class SawtoothWaveEnv(gym.Env):
    def __init__(self):
        super(SawtoothWaveEnv, self).__init__()
        self.iteration = 0
        self.horizon = 1024
        self.endless_mode = True
        self.signal_counter = 0
        self.delta_steps = 128
        self.magnitude = 1.
        self.signal_slope = 2. * self.magnitude / self.delta_steps
        self.y = 0.  # output
        self.error = 0.

        self.pid = PIDController(k_p=0.7)

        # Upper and lower limit for observation space
        self.o_high = np.array([10.,  # e: error limit
                                10.,  # y: output limit
                                self.pid.windup_limit,
                                5.,  # k_p limit
                                5.,  # k_i limit
                                5.  # k_d limit
                                ])
        self.o_low = np.array([-10.,  # e: error limit
                               -10.,  # y: output limit
                               -self.pid.windup_limit,
                               0.,  # k_p limit
                               0.,  # k_i limit
                               0.  # k_d limit
                               ])
        # Upper and lower limit for action space
        self.a_high = np.array([1., 1., 1.])
        self.a_low = np.array([-1., -1., -1.])

        # Create spaces
        self.observation_space = gym.spaces.Box(self.o_low, self.o_high, dtype=np.float32)
        self.action_space = gym.spaces.Box(self.a_low, self.a_high, dtype=np.float32)

    @staticmethod
    def plant_dynamics(a):
        return a  # these are linear!

    def get_w_value(self):
        it = (self.signal_counter + self.delta_steps // 2) % self.delta_steps
        self.signal_counter += 1
        return it * self.signal_slope - self.magnitude

    @property
    def state(self):
        return np.concatenate(([self.error, self.y, self.pid.sum_of_errors], self.pid.parameters))

    def step(self, action):
        new_params = action
        self.pid.set_from_policy_mean(new_params)

        w = self.get_w_value()
        self.error = w - self.y
        u = self.pid(self.error)
        self.y += self.plant_dynamics(u)

        r = 1. / np.e**np.abs(10*self.error)
        done = False if self.endless_mode else self.iteration >= self.horizon
        self.iteration += 1
        info = dict()
        return self.state, r, done, info

    def reset(self):
        self.signal_counter = 0
        self.iteration = 0
        return self.state

    def render(self, mode='human'):
        pass


if __name__ == '__main__':
    env = SawtoothWaveEnv()
    x = env.reset()
    done = False
    ret = 0.
    while not done:
        action = np.array([0., 0, 0])
        x, reward, done, _ = env.step(action)
        ret += reward
    print('Return:', ret)

    print('Plot jump')
    saw = SawtoothWaveEnv()
    N = 1024
    ys = np.zeros(N)
    ts = np.arange(N)
    print(saw.pid.parameters)
    for i in range(N):
        ys[i] = saw.get_w_value()
        if i == 100:
            saw.pid.set(np.array([.1, .1, .1]))
            print(saw.pid.parameters)
            print(saw.pid.k_d)
    plt.figure()
    plt.plot(ts, ys, 'b-')
    plt.show()
