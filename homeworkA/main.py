from grid import Grid
import numpy as np

# state-values predicted by first view Monte Carlo
play = Grid()
iterations = 100000
values = np.zeros((4, 4))
returns = {(i, j): list() for i in range(4) for j in range(4)}
gamma = 0.9

for a in range(iterations):
    episode = play.generate_episode()
    G = 0
    for i, step in enumerate(episode[::-1]):
        # episode contains [s0, action0, reward1,s1, s1, action1],...[sT-1, actionT-1, final_reward, sT]
        G = gamma*G + step[2]
        if step[0] not in [x[0] for x in episode[::-1][i+1:]]:  # justify whether step[0] is the first meet
            idx = (step[0][0], step[0][1])  # idx is (x, y)
            returns[idx].append(G)
            temp_values = np.average(returns[idx])
            values[idx[0], idx[1]] = temp_values
    # show the process every 5000 iterations
    if a % 5000 == 0:
        print(f'This is the No.{a} iterations.')
        print(values)

# the final state value functions
print('The final state values are:')
print(values)
