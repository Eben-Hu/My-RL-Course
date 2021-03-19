from gridworld import GridWorld
from utilities import Get_Action
import numpy as np


def generate_one_episode(env, policy, max_episode_number=1000):
    obs = env.reset()
    observations = [obs]
    rewards = []
    dones = []
    actions = []
    number = 0  # the number of steps in one episode which is no more than max_episode_number
    while True:
        action = policy.stochastic(obs)  # here the method depends on the policy
        next_obs, reward, done, info = env.step(action)
        observations.append(next_obs)
        rewards.append(reward)
        dones.append(done)
        actions.append(action)
        number += 1
        if done == 1 or number == max_episode_number:  # reach final state or max number step
            break
    return observations, actions, rewards, dones


def first_visit_monte_carlo_evaluate(discount=0.9):
    GAMMA = discount
    env = GridWorld()
    policy = Get_Action()
    values = np.zeros(16)
    returns = {state: list() for state in range(16)}
    number_of_episodes = 10000
    for episode in range(number_of_episodes):
        observations, _, rewards, _ = generate_one_episode(env, policy)
        observations.pop()  # exclude the sT
        G = 0
        for i, obs in enumerate(observations[::-1]):  # reverse the list observations and rewards
            G = GAMMA * G + rewards[::-1][i]
            if obs not in observations[::-1][i + 1:]:
                returns[obs].append(G)
                temp = np.average(returns[obs])
                values[obs] = temp
            values[15] = 0
    return values


def stochastic(obs=None):
    action = np.random.choice([-4, 4, -1, 1])
    return action


def epsilon_greedy(obs, q_tables):
    EPSILON = 0.1
    action_indexes = {0: -4, 1: 4, 2: -1, 3: 1}
    p = np.random.random()
    if p < EPSILON / 4:
        action = stochastic()
    else:
        action_index = np.argmax(q_tables[obs])
        action = action_indexes[action_index]
    return action


def q_learning(q_tables):
    ALPHA = 0.001
    GAMMA = 0.9
    policy_list = []  # contains the final approximate optimal policy
    actions = ['up', 'down', 'left', 'right']
    indexes_actions = {-4: 0, 4: 1, -1: 2, 1: 3}
    number_of_episodes = 10000
    for episode in range(number_of_episodes):
        max_step_number = 1000
        env = GridWorld()
        obs = env.reset()
        number = 0  # the number of steps in one episode which is no more than max_step_number
        while True:  # one episode
            action = epsilon_greedy(obs, q_tables)  # action = A
            action_index = indexes_actions[action]
            next_obs, reward, done, _ = env.step(action)  # next_obs = S', reward = R
            q_tables[obs][action_index] = q_tables[obs][action_index] + ALPHA * (
                        reward + GAMMA * max(q_tables[next_obs]) - q_tables[obs][action_index])
            obs = next_obs
            number += 1
            if done == 1 or number == max_step_number:  # reach final state or max number step
                break
    for row in range(len(q_tables)):
        policy_list.append(actions[np.argmax(q_tables[row])])
    optimal_policy = np.array(policy_list).reshape(4, 4)
    return q_tables, optimal_policy
