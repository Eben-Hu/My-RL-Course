import numpy as np


class Get_Action:
    def __init__(self):
        self.EPSILON = 0.1

    def stochastic(self, obs=None):
        action = np.random.choice([-4, 4, -1, 1])
        return action






