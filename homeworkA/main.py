from gridworld import GridWorld
from policies import first_visit_monte_carlo_evaluate

# (2)  state-values evaluated by first view Monte Carlo
values = first_visit_monte_carlo_evaluate()
print(f'The final values are:{values.reshape(4, 4)}')

# (3)  evaluate the policy performance R_Avg
values = first_visit_monte_carlo_evaluate(discount=1.0)
print(f'The final policy performance R_Avg is {sum(values)/15}.')

# (4) the optimal tabular policy with Q-learning and its policy performance R_Avg

