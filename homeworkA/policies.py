from gridworld import GridWorld
from utilities import Get_Action
import numpy as np


def generate_one_episode(env, policy, max_episode_number=10000):
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
    number_of_episodes = 200000
    for episode in range(number_of_episodes):
        observations, _, rewards, _ = generate_one_episode(env, policy)
        observations.pop()
        G = 0
        for i, obs in enumerate(observations[::-1]):  # reverse the list observations and rewards
            G = GAMMA * G + rewards[::-1][i]
            if obs not in observations[::-1][i+2:]:
                returns[obs].append(G)
                temp = np.average(returns[obs])
                values[obs] = temp
            values[15] = 0
    return values

# def q_learning():
#     return None
