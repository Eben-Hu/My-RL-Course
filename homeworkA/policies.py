
def monte_carlo(env, player):
episodes = []
env.reset()




    def generate_episode(self):
        # this function is for Monte Carlo to generate episodes
        initial_state = random.choice(self.states[:-1])
        episode = []  # episode contains [s0, action0, reward1,s1, s1, action1,...sT-1, actionT-1, final_reward, sT]
        while True:
            if list(initial_state) in self.terminate_state:
                return episode
            else:
                action = self.get_action
                final_state, current_reward = self.step(initial_state, action)
                episode.append([list(initial_state), action, current_reward, list(final_state)])
                initial_state = final_state

def q_learning():

