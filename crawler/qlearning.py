import random


class QLearner:

    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.q_table = {}   # {(state, action): value}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_q(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state, actions):

        if not actions:
            return None

        if random.random() < self.epsilon:
            print("EXPLORING")
            return random.choice(actions)

        print("EXPLOITING")

        q_values = [(self.get_q(state, a), a) for a in actions]
        return max(q_values, key=lambda x: x[0])[1]

    def update(self, state, action, reward, next_state, next_actions):

        current_q = self.get_q(state, action)

        max_next_q = max(
            [self.get_q(next_state, a) for a in next_actions],
            default=0
        )

        new_q = current_q + self.alpha * (
            reward + self.gamma * max_next_q - current_q
        )

        self.q_table[(state, action)] = new_q