import numpy as np
import random


class Grid:
    def __init__(self):
        self.dimension = 4
        self.terminate_state = [[3, 3]]
        self.terminate = [3, 3]
        self.states = [[i, j] for i in range(self.dimension) for j in range(self.dimension)]
        self.actions_pairs = {'up': np.array([0, -1]), 'down': np.array([0, 1]), 'left': np.array([-1, 0]),
                              'right': np.array([1, 0])}
        self.actions = ['up', 'down', 'left', 'right']
        self.neighbor_actions1 = {'up': 'right', 'down': 'left', 'right': 'down', 'left': 'up'}
        self.neighbor_actions2 = {'up': 'left', 'down': 'right', 'right': 'up', 'left': 'down'}
        self.discount = 0.9
        self.epsilon = 0.1
        self.alpha = 0.01  # learning rate
        self.final_reward = 9
        self.normal_reward = -1

    def step(self, state, action):
        """
        step calculate the new_state and the instant reward
        :param state: the current state
        :param action: the chosen action from the agent
        :return: next_state and reward
        """
        if list(state) in self.terminate_state:
            next_state = self.terminate
            return next_state, self.final_reward
        else:
            next_state = state + self.actions_pairs[action]
        if next_state[0] < 0 or next_state[0] > 3 or next_state[1] < 0 or next_state[1] > 3:
            # if move out of the grids
            next_state = state
            return next_state, self.normal_reward
        elif list(next_state) in self.terminate_state:
            return next_state, self.final_reward
        else:
            return next_state, self.normal_reward

    def get_action(self):
        """
        this function chooses the random action
        :return: the chosen action
        """
        random_action = np.random.choice(self.actions)
        p = np.random.random()
        if p < 0.8:
            return random_action
        elif 0.8 <= p < 0.9:
            return self.neighbor_actions1[random_action]
        else:
            return self.neighbor_actions2[random_action]

    def generate_episode(self):
        # this function is for Monte Carlo to generate episodes
        initial_state = random.choice(self.states[:-1])
        episode = []  # episode contains [s0, action0, reward1,s1, s1, action1,...sT-1, actionT-1, final_reward, sT]
        while True:
            if list(initial_state) in self.terminate_state:
                return episode
            else:
                action = self.get_action()
                final_state, current_reward = self.step(initial_state, action)
                episode.append([list(initial_state), action, current_reward, list(final_state)])
                initial_state = final_state

    def pair_to_sequence(self, state):
        return state[0]*4+state[1]

    def q_learning(self, q_tables):
        initial_state = random.choice(self.states[:-1])  # (x, y)
        episode = []
        while True:
            if list(initial_state) in self.terminate_state:
                return episode
            else:
                p = np.random.random()
                initial_sequence = self.pair_to_sequence(initial_state)  # initial_sequence = St
                if p < self.epsilon/4:  # epsilon
                    action = self.get_action()  # action = At, randomly selected
                    action_index = self.actions.index(action)
                    final_state, current_reward = self.step(initial_state, action)  # final_state = St+1, Rt+1
                    final_sequence = self.pair_to_sequence(final_state)
                    episode.append([list(initial_state), action, current_reward, list(final_state)])
                    next_action_index = np.argmax(q_tables[final_sequence])
                    next_action = self.actions_pairs[self.actions[next_action_index]]  # next_action = At+1
                    q_tables[initial_sequence][action_index] = q_tables[initial_sequence][action_index] + self.alpha *(current_reward + self.discount * (q_tables[final_sequence][next_action_index] - q_tables[initial_sequence][action_index]))
                    initial_state = final_state
                else:
                    action_index = np.argmax(q_tables[initial_sequence])  # action = At
                    action = self.actions[action_index]  # 'up'/'down'/'left'/'right'
                    final_state, current_reward = self.step(initial_state, action)  # final_state = St+1, Rt+1
                    final_sequence = self.pair_to_sequence(final_state)
                    episode.append([list(initial_state), action, current_reward, list(final_state)])
                    next_action_index = np.argmax(q_tables[final_sequence])
                    next_action = self.actions_pairs[self.actions[next_action_index]]  # next_action = At+1
                    q_tables[initial_sequence][action_index] = q_tables[initial_sequence][action_index] + self.alpha * (current_reward + self.discount * (q_tables[final_sequence][next_action_index] - q_tables[initial_sequence][action_index]))
                    initial_state = final_state
        return q_tables




