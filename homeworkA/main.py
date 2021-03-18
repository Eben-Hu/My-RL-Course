from gridworld import GridWorld
from policies import first_visit_monte_carlo_evaluate, q_learning
import numpy as np

# (2)  state-values evaluated by first view Monte Carlo
# values = first_visit_monte_carlo_evaluate()
# print(f'The final values are:{values.reshape(4, 4)}')

# (3)  evaluate the policy performance R_Avg
# values = first_visit_monte_carlo_evaluate(discount=1.0)
# print(f'The final policy performance R_Avg is {sum(values)/15}.')

# (4) the optimal tabular policy with Q-learning and its policy performance R_Avg
q_tables = np.zeros((16, 4), dtype=np.float) + np.random.normal(0, 0.3, (16, 4))
q_tables[15][:] = 0
final_q_tables, final_policy = q_learning(q_tables)
print(final_q_tables)
print(final_policy)

