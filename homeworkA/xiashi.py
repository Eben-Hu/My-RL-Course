import numpy as np
#
# np.random.seed(1)
#
# # for i in range(10):
# #     a = np.random.random()
# #     print(a)
#
# # b = ['up', 'down', 'left', 'right']
# # actions = {'up': np.array([0, -1]), 'down': np.array([0, 1]), 'left': np.array([-1, 0]), 'right': np.array([1, 0])}
# # for j in range(5):
# #     c = np.random.choice(b)
# #     print(c)
# #     print(actions[c])
# #
# # def step(a, b):
# #     c = a + b
# #     d = a * b
# #     return c, d
# #
# # x, y = step(2,4)
# # print(x)
# # print(y)
#
#
# # a = list(range(5))
# # print(a)
# # print(a.index(3))
#
# import numpy as np
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set_style("darkgrid")
# import random
#
# # parameters
# gamma = 0.6 # discounting rate
# rewardSize = -1
# gridSize = 4
# terminationStates = [[0,0], [gridSize-1, gridSize-1]]
# actions = [[-1, 0], [1, 0], [0, 1], [0, -1]]
# numIterations = 10000
#
# V = np.zeros((gridSize, gridSize))
# returns = {(i, j):list() for i in range(gridSize) for j in range(gridSize)}
# print(returns)
# # deltas = {(i, j):list() for i in range(gridSize) for j in range(gridSize)}
# states = [[i, j] for i in range(gridSize) for j in range(gridSize)]
# print(states[:-1])
#
#
# # utils
# def generateEpisode():
#     initState = random.choice(states[1:-1])
#     episode = []
#     while True:
#         if list(initState) in terminationStates:
#             return episode
#         action = random.choice(actions)
#         finalState = np.array(initState)+np.array(action)
#         if -1 in list(finalState) or gridSize in list(finalState):
#             finalState = initState
#         episode.append([list(initState), action, rewardSize, list(finalState)])
#         initState = finalState
#
#
# for it in tqdm(range(numIterations)):
#     episode = generateEpisode()
#     G = 0
#     for i, step in enumerate(episode[::-1]):
#         G = gamma*G + step[2]
#         if step[0] not in [x[0] for x in episode[::-1][len(episode)-i:]]:
#             idx = (step[0][0], step[0][1])
#             returns[idx].append(G)
#             newValue = np.average(returns[idx])
#             # deltas[idx[0], idx[1]].append(np.abs(V[idx[0], idx[1]]-newValue))
#             V[idx[0], idx[1]] = newValue
#
# # using gamma = 1
# plt.figure(figsize=(20,10))
# all_series = [list(x)[:50] for x in deltas.values()]
# for series in all_series:
#     plt.plot(series)
#
#
# # part of reverse
# # a = [[[2, 1], [1, 0], -1, [3, 1]], [[3, 1], [0, -1], -1, [3, 0]], [[2, 0], [-1, 0], -1, [2, 0]], [[2, 0], [-1, 0], -1, [1, 0]]]
# # print(len(a))
# # print(a[::-1])
# # print(a[::-1][0][0])
# # print(a[::-1][1][0])
# # print(a[::-1][2][0])
# # print(a[::-1][3][0])
# # print("The experiment begins")
# # for i, step in enumerate(a[::-1]):
# #     if step[0] not in [x[0] for x in a[::-1][i+1:]]:
# #         print(f"i is {i}")
# #         print(f"step[0] is {step[0]}")
# #         idx = (step[0][0], step[0][1])
# #         print(f"idx is {idx}")
# #         print(f"step[0][0] is {step[0][0]}")
# #     else:
# #         print(f"i is {i}")
# #         print(f"{step[0]} has appeared")
#
# # part of average
# alist = np.array([1, 2, 3, 4, 5, 6])
# print(alist)
# print(np.average(alist))

import numpy as np
# a = np.array([[1,2,3,4], [1,2,3,4],[1,2,3,4],[1,2,3,4]])
# print(a)
# print(sum(a))
# print(sum(sum(a)))

import gym
import numpy as np
actions = gym.spaces.Discrete(4)
# for i in range(10):
#     print(actions.sample())
  # 1 terminal state
# for i in range(10):
#     a = np.random.randint(0, 16)
#     print(f'a is {a} ')
#     print(f'a // 4 is {a // 4}')
#     print(f'a % 4 is {a % 4}')

a = np.array([[1,2,3,4], [5,6,7,8]])
print(len(a))
print(np.argmax(a[0]))
print(type(a[0][1]))


