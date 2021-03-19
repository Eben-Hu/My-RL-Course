from gridworld import GridWorld
from utilities import Get_Action
import numpy as np
GAMMA = 0.9
ALPHA = 0.001


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


def first_visit_monte_carlo_evaluate(gamma=GAMMA, number_of_episodes=100000):
    env = GridWorld()
    policy = Get_Action()
    values = np.zeros(16)
    returns = {state: list() for state in range(16)}
    for episode in range(number_of_episodes):
        observations, _, rewards, _ = generate_one_episode(env, policy)
        observations.pop()  # exclude the sT
        G = 0
        for i, obs in enumerate(observations[::-1]):  # reverse the list observations and rewards
            G = gamma * G + rewards[::-1][i]
            if obs not in observations[::-1][i + 1:]:
                returns[obs].append(G)
                values[obs] = np.average(returns[obs])
            values[15] = 0
        if episode % 10000 == 0:
            print(f"In the No.{episode} the values are {values}.")
    return values


def stochastic(obs=None):
    action = np.random.choice([-4, 4, -1, 1])
    return action


def epsilon_greedy(obs, q_tables):
    EPSILON = 0.1
    action_indexes = {0: -4, 1: 4, 2: -1, 3: 1}
    p = np.random.random()
    if p < (1 - EPSILON):
        action_index = np.argmax(q_tables[obs])
        action = action_indexes[action_index]
    else:
        action = stochastic()
    return action


def q_learning(q_tables, gamma=GAMMA, alpha=0.001, number_of_episodes=10000, max_step_number=1000):
    policy_list = []  # contains the final approximate optimal policy
    env = GridWorld()
    actions = ['up', 'down', 'left', 'right']
    indexes_actions = {-4: 0, 4: 1, -1: 2, 1: 3}
    rewards = 0
    for episode in range(number_of_episodes):
        obs = env.reset()
        number = 0  # the number of steps in one episode which is no more than max_step_number
        while True:  # one episode
            action = epsilon_greedy(obs, q_tables)  # action = A
            action_index = indexes_actions[action]
            next_obs, reward, done, _ = env.step(action)  # next_obs = S', reward = R
            rewards += reward
            q_tables[obs][action_index] = q_tables[obs][action_index] + alpha * (
                        reward + gamma * max(q_tables[next_obs]) - q_tables[obs][action_index])
            obs = next_obs
            number += 1
            if done == 1 or number == max_step_number:  # reach final state or max number step
                break
    for row in range(len(q_tables)):
        policy_list.append(actions[np.argmax(q_tables[row])])
    performance = rewards/number_of_episodes
    optimal_policy = np.array(policy_list).reshape(4, 4)
    return q_tables, optimal_policy, performance
