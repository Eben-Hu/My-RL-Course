from abc import ABC
import gym
import numpy as np


class GridWorld(gym.Env, ABC):
    def __init__(self, n_width: int = 4,
                 n_height: int = 4,
                 normal_reward: float = -1,
                 final_reward: float = 9):
        self.n_width = n_width  # width of the env calculated by number of cells.
        self.n_height = n_height  # height...
        self.len = n_width * n_height
        self.action_space = gym.spaces.Discrete(4)  # {0:up} {1:down} {2:left} {3:right}
        self.observation_space = gym.spaces.Discrete(self.n_height * self.n_width)
        self.action = None
        self.normal_reward = normal_reward
        self.final_reward = final_reward
        self.end = 15  # (3, 3)
        self.action_pairs = {0: -4, 1: 4, 2: -1, 3: 1}  # {up:-4, down:4, left:-1, right:1}
        self.actions_match = {-4: np.array([-1, 0]), 4: np.array([1, 0]), -1: np.array([0, -1]),
                              1: np.array([0, 1])}
        self.neighbor_actions1 = {-4: 1, 4: -1, -1: -4, 1: 4}  # {up:right, down:left, left:up, right:down}
        self.neighbor_actions2 = {-4: -1, 4: 1, -1: 4, 1: -4}    # {up:left, down:right, left:down, right:up}

    def reset(self):
        self.obs = np.random.randint(0, self.n_width * self.n_height - 1)  # [0, 15)
        return self.obs

    def step(self, action):  # action = -4, +4, -1 +1 / up, down, left, right, action is an integer
        done = 0
        info = {}
        p = np.random.random()  # the random behavior
        if p < 0.8:  # decide the real action
            self.action = action
        elif p < 0.9:
            self.action = self.neighbor_actions1[action]
        else:
            self.action = self.neighbor_actions2[action]
        self.row = self.obs // self.n_width  # x
        self.column = self.obs % self.n_height  # y
        obs = np.array([self.row, self.column])  # old state/obs [x, y]
        temp = obs + self.actions_match[self.action]  # new obs [x, y]
        if temp[0] * 4 + temp[1] == self.end:  # final state
            self.obs = self.end
            reward = self.final_reward
            done = 1
        elif temp[0] < 0 or temp[0] > 3 or temp[1] < 0 or temp[1] > 3:  # out of the range
            reward = self.normal_reward
        else:
            self.obs = temp[0] * 4 + temp[1]
            reward = self.normal_reward
        return self.obs, reward, done, info

def test_env():
    env = GridWorld()
    obs = env.reset()
    while True:
        print(obs)
        obs, _,_,_, = env.step(1)

if __name__ == '__main__':
    test_env()
