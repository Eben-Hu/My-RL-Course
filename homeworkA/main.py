from policies import first_visit_monte_carlo_evaluate, q_learning
import numpy as np


# (2)  state-values evaluated by first view Monte Carlo
def monte_carlo():
    values = first_visit_monte_carlo_evaluate()
    print(f'The final values are:{values.reshape(4, 4)}')


# (3)  evaluate the policy performance R_Avg
def monte_carlo_performance():
    values = first_visit_monte_carlo_evaluate(gamma=1.0)
    print(f'The final policy performance R_Avg is {sum(values) / 15}.')


# (4) the optimal tabular policy with Q-learning and its policy performance R_Avg
def qlearning():
    q_tables = np.zeros((16, 4), dtype=np.float) + np.random.normal(0, 0.3, (16, 4))
    q_tables[15][:] = 0
    final_q_tables, final_policy, final_performance = q_learning(q_tables)
    print(f'The final q tables is :{final_q_tables} with the final optimal policy {final_policy} and the final '
          f'performance value is {final_performance}.')


if __name__ == '__main__':
    qlearning()
