import numpy as np


def stochastic(obs=None):
    action = np.random.choice([-4, 4, -1, 1])
    return action


values = np.zeros(16)
print(values[15])
# returns = np.zeros(16,)
# a = [1,2,3,4,5,6]
# rewards = [-1, -1, -1, -1, -1, 9]
# # for i, b in enumerate(a[::-1]):
# #     print((i, b))
# #     print(rewards[::-1][i])
#
# print(a[::-1][1:])
for i in range(50):
    a = np.random.randint(0, 16)
    print(a)
