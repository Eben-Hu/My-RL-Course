import numpy as np


class Get_Action:
    def stochastic(self, obs=None):
        action = np.random.choice([-4, 4, -1, 1])
        return action

    def 




