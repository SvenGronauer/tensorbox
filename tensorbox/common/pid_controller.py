import numpy as np
import matplotlib.pyplot as plt


class PIDController(object):
    """
    Proportional–integral–derivative controller (PID controller) for closed-loop control
    """

    def __init__(self,
                 dt,
                 k_p=1.2,
                 k_i=1.,
                 k_d=.005,
                 limit_parameters=True,
                 anti_windup=True):
        self.parameters = np.array([k_p, k_i, k_d])
        self.init_parameters = np.array([k_p, k_i, k_d])
        self.max_limits_parameters = np.array([50., 50., 0.])
        self.min_limits_parameters = np.array([0., 0., 0.])
        self.scales = np.array([5., 5., 1.])
        self.dt = dt  # time step
        self.last_error = 0.
        self.limit_parameters = limit_parameters  # limit the internal states and outputs
        self.sum_of_errors = 0.
        self.anti_windup = anti_windup
        self.windup_limit = 1000.

    def __call__(self, error, *args, **kwargs):
        """ calculate control u"""
        d_slope = (self.last_error - error) / self.dt
        u = self.k_p * error + self.k_i * self.sum_of_errors + self.k_d * d_slope
        self.last_error = error
        self.sum_of_errors += error * self.dt
        self.apply_anti_windup()
        return u

    def apply_anti_windup(self):
        if self.anti_windup:
            if self.sum_of_errors > self.windup_limit:
                self.sum_of_errors = self.windup_limit
            if self.sum_of_errors < -self.windup_limit:
                self.sum_of_errors = -self.windup_limit

    def get_parameters(self):
        return self.parameters

    def get_scaled_parameters(self):
        scaled_parameters = (self.parameters - self.init_parameters) / self.scales
        return scaled_parameters

    @property
    def k_p(self):
        return self.parameters[0]

    @property
    def k_i(self):
        return self.parameters[1]

    @property
    def k_d(self):
        return self.parameters[2]

    def set(self, params):
        self.parameters = params

    def set_from_policy_mean(self, mean):
        """ set mean output of neural network and apply a linear transformation function to
            adjust parameter range
        """
        new_params = mean * self.scales + self.init_parameters
        if self.limit_parameters:
            new_params = np.clip(new_params,
                                 self.min_limits_parameters,
                                 self.max_limits_parameters)
        self.set(new_params)

    def reset(self):
        self.last_error = 0.
        self.sum_of_errors = 0.

    def update(self, delta):
        new_params = self.parameters + delta
        self.set(new_params)


if __name__ == '__main__':
    print('Plot jump')
    N = 1000
    stop = np.pi * 4
    dt = stop / N
    ts = np.arange(0., stop, dt)
    w = (ts > 0.3).astype(float)
    # w = 0.5 * (-np.cos(ts) + 1)
    a = 5

    pid = PIDController(dt=dt)
    ys = np.zeros_like(ts)
    y = 0.

    for i in range(N):
        err = w[i] - y
        y += pid(err)
        ys[i] = y

    plt.figure()
    plt.plot(ts, w, 'b-', ts, ys, 'r')
    plt.show()
