import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class InitialCondition:
    t: float = 0
    x_t: float = 0

class StochasticModel:
    def __init__(self, coeff_dt, coeff_dw, initial_condition):
        self.coeff_dt = coeff_dt
        self.coeff_dw = coeff_dw
        self.initial_condition = initial_condition

    def compute_dw(self, delta_t):
        dw = np.random.normal(loc=0.0, scale=np.sqrt(delta_t))
        return dw

    def compute_next_Xt(self, X_t1, t1, delta_t):
        X_t2 = X_t1 + delta_t*(self.coeff_dt(X_t1, t1)) + self.coeff_dw(X_t1, t1)*self.coeff_dw(X_t1, t1)
        return X_t2

    def simulate(self, t1, t2, steps=100):
        assert t1==self.initial_condition.t, (f"The initial condition must be {self.initial_condition.t}, "
                                              f"but found as {t1}")
        n = steps+1
        time_space = np.linspace(t1,t2,n)
        del_t = time_space[1]-time_space[0]

        X_space = np.zeros_like(time_space)
        X_space[0] = self.initial_condition.x_t
        for i, t in enumerate(time_space[1:]):
            X_space[i+1] = self.compute_next_Xt(X_space, t, del_t)

        return time_space, X_space

