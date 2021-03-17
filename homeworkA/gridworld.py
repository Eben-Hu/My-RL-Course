import gym
import numpy as np


class GridWorld(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, n_width: int = 4,
                 n_height: int = 4,
                 u_size=40,
                 normal_reward: float = -1,
                 final_reward: float = -9):
        self.u_size = u_size  # size for each cell (pixels)
        self.n_width = n_width  # width of the env calculated by number of cells.
        self.n_height = n_height  # height...
        self.len = n_width * n_height
        self.width = u_size * n_width  # scenario width (pixels)
        self.height = u_size * n_height  # height (pixels)
        self.action_space = gym.spaces.Discrete(4)  # {0:up} {1:down} {2:left} {3:right}
        self.observation_space = gym.spaces.Discrete(self.n_height * self.n_width)
        self.normal_reward = normal_reward
        self.final_reward = final_reward
        self.end = [15]  # (3, 3)
        self.episodes = None
        self.state = self._reset()

    def _reset(self):
        self.episodes = []  # (s0, a1, s1), (s1, a1, s2),...
        return np.random.randint(range(0, self.n_width * self.n_height))

    # def get_action(self):  # this should belongs to policies
    #     random_action = self.action_space.sample()
    #     p = np.random.random()
    #     if p < 0.8:
    #         decided_action = random_action
    #     elif p < 0.9:
    #         decided_action = self.neighbor_actions1[random_action]
    #     else:
    #         decided_action = self.neighbor_actions2[random_action]
    #     return self.action_pairs[decided_action]

    def _step(self, action):
        old_state = self.state
        temp_state = self.state + action
        if temp_state in self.end:
            self.state = self.end
            reward = self.final_reward
        elif temp_state in range(self.n_width * self.n_height - 1):
            self.state = temp_state
            reward = self.normal_reward
        else:  # out of the range
            self.state = old_state
            reward = self.normal_reward
        self.episodes.append(old_state, action, self.state)  # (s0, a1, s1)
        if reward == self.normal_reward:
            return self.state, self.normal_reward, 0, {'state': (self.state // 4, self.state % 4)}
        else:
            return self.state, self.normal_reward, 1, {'state': (self.state // 4, self.state % 4)}

    # def render(self, mode='human'):
