from abc import ABC
import gym
import numpy as np


class GridWorld(gym.Env, ABC):
    def __init__(self, n_width: int = 4,
                 n_height: int = 4,
                 normal_reward: float = -1,
                 final_reward: float = -9):
        self.n_width = n_width  # width of the env calculated by number of cells.
        self.n_height = n_height  # height...
        self.len = n_width * n_height
        self.action_space = gym.spaces.Discrete(4)  # {0:up} {1:down} {2:left} {3:right}
        self.observation_space = gym.spaces.Discrete(self.n_height * self.n_width)
        self.obs = None
        self.row = self.obs // self.n_width
        self.column = self.obs % self.n_height
        self.action = None
        self.normal_reward = normal_reward
        self.final_reward = final_reward
        self.end = 15  # (3, 3)
        self.action_pairs = {0: -4, 1: 4, 2: -1, 3: 1}
        self.neighbor_actions1 = {0: 3, 1: 2, 2: 2, 3: 0}
        self.neighbor_actions2 = {0: 2, 1: 3, 2: 0, 3: 2}

    def reset(self):
        self.obs = np.random.randint(range(0, self.n_width * self.n_height))
        return self.obs

    def step(self, action):  # action = {up:-4} {down:+4} {left:-1} {right:+1}
        done = 0
        info = {}
        temp = self.obs + action
        if temp == self.end:  # 15
            self.obs = self.end
            reward = self.final_reward
            done = 1
        elif temp in range(self.n_width * self.n_height - 1):  # 0~14
            self.obs = temp
            reward = self.normal_reward
        else:  # out of the range(16)
            reward = self.normal_reward
        return self.obs, reward, done, info

    def random_action(self):
        random_action = self.action_space.sample()
        p = np.random.random()
        if p < 0.8:
            decided_action = random_action
        elif p < 0.9:
            decided_action = self.neighbor_actions1[random_action]
        else:
            decided_action = self.neighbor_actions2[random_action]
        return self.action_pairs[decided_action]
