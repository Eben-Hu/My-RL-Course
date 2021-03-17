from abc import ABC
from gridworld import GridWorld
import numpy as np


class Cleaner(GridWorld, ABC):
    def __init__(self):
        self.obs = self.state
        self.action_pairs = {0: -4, 1: 4, 2: -1, 3: 1}
        self.neighbor_actions1 = {0: 3, 1: 2, 2: 2, 3: 0}
        self.neighbor_actions2 = {0: 2, 1: 3, 2: 0, 3: 2}

    @property
    def get_action(self):
        random_action = self.action_space.sample()
        p = np.random.random()
        if p < 0.8:
            decided_action = random_action
        elif p < 0.9:
            decided_action = self.neighbor_actions1[random_action]
        else:
            decided_action = self.neighbor_actions2[random_action]
        return self.action_pairs[decided_action]
