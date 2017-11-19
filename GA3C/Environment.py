import gym


class Environment:
    def __init__(self, name):
        self.name = name
        self.env = gym.make('CartPole-v0')

    def reset(self):
        self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self):
        self.env.render()
