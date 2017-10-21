import mujoco_py as mj
import numpy as np
import random
from math import degrees
from agent import Agent

class Environment:
    def __init__(self):
        self.viewer = None

    def spawn_agent(self, topology, epochs, memory_length, batch_size, learning_rate, gamma, epsilon, epsilon_min, epsilon_decay):
        self.agent = Agent(topology, epochs, memory_length, batch_size, learning_rate, gamma, epsilon, epsilon_min, epsilon_decay)
        self.viewer = None
        self.sim = self.agent.sim
        self.initial_state = self.sim.get_state()

    def is_done(self):
        x_pos = self.sim.get_state().qpos[0]
        angle = degrees(self.sim.get_state().qpos[1])
        if (20. > angle > -20.) and (0.95 > x_pos > -0.95):
            return False
        else:
            return True

    def get_reward(self):
        if not self.is_done():
            return 1
        else:
            return -10

    def step(self, index):
        self.agent.do_action(self.agent.get_possible_actions()[index])
        self.sim.step()
        s = np.array([self.sim.get_state().qpos.tolist() + self.sim.get_state().qvel.tolist()])
        r = self.get_reward()
        return s, r, self.is_done()

    def render(self):
        if not self.viewer:
            self.viewer = mj.MjViewer(self.agent.sim)
        self.viewer.render()

    def reset(self, number_of_random_actions):
        self.sim.set_state(self.initial_state)
        # Perform a few random actions to make the problem stochastic.
        for n in range(number_of_random_actions):
            a = random.randint(0, len(self.agent.get_possible_actions()) - 1)
            self.step(a)
        return np.array([self.sim.get_state().qpos.tolist() + self.sim.get_state().qvel.tolist()])


if __name__ == "__main__":
    env = Environment()
    env.spawn_agent(topology=[[4, 24, 24, 2], ['relu', 'relu', 'linear']], memory_length=2000, batch_size=64, epochs=20,
                    learning_rate=0.01, gamma=0.95, epsilon=1, epsilon_min=0.01, epsilon_decay=0.995)
    epochs = 2000

    score_list = []
    for e in range(epochs):
        state = env.reset(number_of_random_actions=2)
        for time in range(500):

            action = env.agent.act(state)

            new_state, reward, done = env.step(action)

            memory = (state, action, reward, new_state, done)
            env.agent.add_memory(memory)

            state = new_state
            if done or time == 499:
                print("episode: {}/{}, score: {}, e: {}".format(e, epochs, time, round(env.agent.epsilon, 2)))
                score_list.append(time)
                break

        if len(env.agent.replay_memory) > env.agent.batch_size:
            env.agent.replay()

    settings = "topology=[[4, 24, 24, 2], ['relu', 'relu', 'linear']], memory_length=2000, batch_size=64, epochs=20, " \
               "learning_rate=0.01, gamma=0.95, epsilon=1, epsilon_min=0.1, epsilon_decay=0.995"
    file = open("./logs/" + settings + ".txt", 'w')
    file.write("#" + settings + "\n")
    for score in score_list:
        file.write(str(score) + '\n')

