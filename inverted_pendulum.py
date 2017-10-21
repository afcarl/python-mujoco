import mujoco_py as mj
import numpy as np
from math import degrees
from agent import Agent


class Environment:
    def __init__(self):
        self.viewer = None

    def spawn_agent(self, topology, memory_length, learning_rate, gamma, epsilon, epsilon_min, epsilon_decay):
        self.agent = Agent(topology, memory_length, learning_rate, gamma, epsilon, epsilon_min, epsilon_decay)
        self.viewer = None
        self.sim = self.agent.sim
        self.initial_state = self.sim.get_state()

    def is_done(self):
        angle = degrees(self.sim.get_state().qpos[1])
        if 20. > angle > -20.:
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

    def reset(self):
        self.sim.set_state(self.initial_state)
        return np.array([self.initial_state.qpos.tolist() + self.initial_state.qvel.tolist()])


if __name__ == "__main__":
    env = Environment()
    env.spawn_agent(topology=[[4, 24, 24, 2], ['relu', 'relu', 'linear']], memory_length=500,
                    learning_rate=0.001, gamma=0.95, epsilon=1, epsilon_min=0.01, epsilon_decay=0.995)
    epochs = 1000
    batch_size = 32

    time_list = [0.01]
    for e in range(epochs):
        state = env.reset()
        for time in range(500):

            if sum(time_list) / len(time_list) > 150:
                env.render()

            action = env.agent.act(state)

            new_state, reward, done = env.step(action)

            memory = (state, action, reward, new_state, done)
            env.agent.add_memory(memory)

            state = new_state
            time_list.append(time)

            if done or time == 499:
                print("episode: {}/{}, score: {}, average: {}, e: {}".format(e, epochs, time,
                                                                             sum(time_list) / len(time_list),
                                                                             env.agent.epsilon))
                break

        if len(env.agent.replay_memory) > batch_size:
            env.agent.replay(batch_size)
