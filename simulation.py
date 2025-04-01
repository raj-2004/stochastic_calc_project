from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch


class DerivableFunction:
    def __init__(self, callable_function):
        self.forward = callable_function

    def __call__(self, x_t1):
        if not isinstance(x_t1, torch.Tensor):
            x_t1 = torch.tensor(x_t1, requires_grad=True)
        return self.forward(x_t1)

    def derivative(self, x_t1):
        if not isinstance(x_t1, torch.Tensor):
            x_t1 = torch.tensor(x_t1, requires_grad=True)
        _y = self(x_t1)
        _y.backward()
        _grad = x_t1.grad
        return _grad

@dataclass
class InitialCondition:
    t: float = 0
    x_t: float = 0

class StochasticModel:
    def __init__(self, coeff_dt, coeff_dw, initial_condition):
        self.coeff_dt = coeff_dt
        self.coeff_dw = coeff_dw
        self.initial_condition = initial_condition


    def compute_next_xt(self, x_t1, t1, delta_t):
        x_t2 = x_t1 + delta_t * (self.coeff_dt(x_t1, t1)) + self.coeff_dw(x_t1, t1) * self.compute_dw(delta_t)
        return x_t2

    def simulate(self, t1, t2, steps=100):
        assert t1==self.initial_condition.t, (f"The initial condition must be {self.initial_condition.t}, "
                                              f"but found as {t1}")
        n = steps+1
        time_space = np.linspace(t1,t2,n)
        del_t = time_space[1]-time_space[0]

        x_space = np.zeros_like(time_space)
        x_space[0] = self.initial_condition.x_t
        for i, t in enumerate(time_space[1:]):
            x_space[i+1] = self.compute_next_xt(x_space[i], t, del_t)

        return time_space, x_space

    @staticmethod
    def compute_dw(delta_t):
        dw = np.random.normal(loc=0.0, scale=np.sqrt(delta_t))
        return dw

if __name__=='__main__':
    pass

    #region Checking DerivableFunction
    # coeff_dt_constant = 0.05
    # coeff_dw_constant = 0.1

    # def coeff_dw(x_t1, **kwargs):
    #     return coeff_dw_constant*x_t1
    # dw_func = DerivableFunction(coeff_dw)
    # dy = dw_func.derivative(x_t1=10.0)
    #endregion

    #region Checking StochasticModel
    # coeff_dt_constant = 0.05
    # coeff_dw_constant = 0.1
    # coeff_dt = lambda x_t1=None, t1=None: coeff_dt_constant
    # coeff_dw = lambda x_t1=None, t1=None: coeff_dw_constant*t1**2+t1
    # model = StochasticModel(coeff_dt, coeff_dw, InitialCondition(0, 0))
    # t, x = model.simulate(0, 4, 200)
    # print(t.shape, x.shape)
    # print(t)
    # print(x)
    # plt.plot(t, x)
    # plt.show()
    #endregion
