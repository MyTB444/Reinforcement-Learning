import numpy as np


class EnvironmentModel:
    def __init__(self, n_states, n_actions, seed=None):
        self.n_states = n_states
        self.n_actions = n_actions
        self.random_state = np.random.RandomState(seed)

    def p(self, next_state, state, action):
        raise NotImplementedError()

    def r(self, next_state, state, action):
        raise NotImplementedError()

    def draw(self, state, action):
        p = [self.p(ns, state, action) for ns in range(self.n_states)]
        next_state = self.random_state.choice(self.n_states, p=p)
        reward = self.r(next_state, state, action)
        return next_state, reward


class Environment(EnvironmentModel):
    def __init__(self, n_states, n_actions, max_steps, dist, seed=None):
        EnvironmentModel.__init__(self, n_states, n_actions, seed)
        self.max_steps = max_steps
        self.dist = dist
        if self.dist is None:
            self.dist = np.full(n_states, 1. / n_states)

    def reset(self):
        self.n_steps = 0
        self.state = self.random_state.choice(self.n_states, p=self.dist)
        return self.state

    def step(self, action):
        if action < 0 or action >= self.n_actions:
            raise Exception('Invalid action.')
        self.n_steps += 1
        done = (self.n_steps >= self.max_steps)
        self.state, reward = self.draw(self.state, action)
        return self.state, reward, done

    def render(self):
        print(f"State: {self.state}, Steps: {self.n_steps}/{self.max_steps}")


class GridWorld(Environment):
    def __init__(self, grid_size=4, max_steps=20, seed=None):
        self.grid_size = grid_size
        n_states = grid_size * grid_size
        super().__init__(n_states, n_actions=4, max_steps=max_steps, dist=None, seed=seed)

    def p(self, next_state, state, action):
        row, col = state // self.grid_size, state % self.grid_size
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
        dr, dc = moves[action]
        new_row, new_col = row + dr, col + dc

        if 0 <= new_row < self.grid_size and 0 <= new_col < self.grid_size:
            expected = new_row * self.grid_size + new_col
            return 1.0 if next_state == expected else 0.0
        return 1.0 if next_state == state else 0.0

    def r(self, next_state, state, action):
        goal = self.n_states - 1
        return 1.0 if next_state == goal else -0.1


if __name__ == "__main__":
    actions = ['w', 's', 'a', 'd']  # Numpad directions
    env = GridWorld(grid_size=4, max_steps=20, seed=0)

    state = env.reset()
    env.render()

    done = False
    while not done:
        c = input('\nMove: ')
        if c not in actions:
            raise Exception('Invalid action')
        state, r, done = env.step(actions.index(c))
        env.render()
