import mujoco_py as mj
import numpy as np
import random
from math import degrees
from agent import Agent
import msvcrt
import time


class Environment:
    def __init__(self):
        self.agent = None
        self.viewer = None
        self.sim = None
        self.initial_state = None

    def spawn_agent(self, parameters):
        self.agent = Agent(parameters)
        self.sim = self.agent.sim
        self.initial_state = self.sim.get_state()

    def is_done(self):
        x_pos = self.sim.get_state().qpos[0]
        angle1 = degrees(self.sim.get_state().qpos[1])
        angle2 = degrees(self.sim.get_state().qpos[2])
        if (20. > angle1 > -20.) and (20. > angle2 > -20.) and (0.99 > x_pos > -0.99):
            return False
        else:
            return True

    def get_reward(self):
        if not self.is_done():
            return 1
        else:
            return -1

    def step(self, index):
        self.agent.do_action(self.agent.get_possible_actions()[index])
        self.sim.step()
        s = np.array([self.sim.get_state().qpos.tolist() + self.sim.get_state().qvel.tolist()])
        r = self.get_reward()
        return s, r, self.is_done()

    def render(self):
        if not self.viewer:
            self.viewer = mj.MjViewer(self.sim)
        self.viewer.render()

    def reset(self, number_of_random_actions):
        self.sim.set_state(self.initial_state)
        # Perform a few random actions to make the problem stochastic.
        for n in range(number_of_random_actions):
            a = random.randint(0, len(self.agent.get_possible_actions()) - 1)
            self.step(a)
        return np.array([self.sim.get_state().qpos.tolist() + self.sim.get_state().qvel.tolist()])

    def test_agent(self, parameters, model_path):
        self.spawn_agent(parameters)
        self.agent.load_model(model_path)
        self.agent.epsilon = 0.01

        while True:
            state = env.reset(number_of_random_actions=5)
            for time in range(100000):
                self.render()
                action = self.agent.act(state)
                new_state, _, done = self.step(action)
                state = new_state
                if done:
                    break

if __name__ == "__main__":
    env = Environment()
    PARAMETERS = {'model3Dpath': 'xml/inverted_double_pendulum.xml',
                  'topology': [[6, 48, 24, 2], ['relu', 'relu', 'linear']],
                  'memory_length': 10000,
                  'batch_size': 64,
                  'epochs': 6,
                  'learning_rate': 0.001,
                  'gamma': 0.99,
                  'epsilon': 1,
                  'epsilon_min': 0.1,
                  'epsilon_decay': 0.999}
    # env.test_agent(PARAMETERS, './models/inverted_double_pendulum_v0.h5')

    env.spawn_agent(PARAMETERS)

    epochs = 100000
    max_steps = 2000
    score_list = []
    q_values = [0.]
    temp_q = []
    for e in range(epochs):
        if msvcrt.kbhit():
            if ord(msvcrt.getch()) == 59:
                break

        state = env.reset(number_of_random_actions=5)
        for step in range(max_steps):
            action = env.agent.act(state)
            new_state, reward, done = env.step(action)

            q = np.sum(env.agent.q_network.predict(state), axis=1)
            temp_q.append(q[0]/2)

            if done:
                new_state = None

            memory = (state, action, reward, new_state, done)
            env.agent.add_memory(abs(reward), memory)

            state = new_state

            if done or step == max_steps-1:
                print("Episode: {}, Score: {}/{}, Q-Value: {}, epsilon: {}".format(e, step, max_steps-1, int(q_values[-1]),
                                                                                   round(env.agent.epsilon, 2)))
                score_list.append(step)
                break

        if env.agent.memory_length >= env.agent.batch_size:
            env.agent.replay()

        q_values.append(sum(temp_q) / len(temp_q))

        # Decay the epsilon
        if env.agent.epsilon > env.agent.epsilon_min:
            env.agent.epsilon *= env.agent.epsilon_decay

        if e % 5 == 0:
            env.agent.update_target()

        #if int(q_values[-1] >= 100):
         #   break

    env.agent.save_model('./models/inverted_double_pendulum_v0.h5')

    time_string = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    file = open("./logs/inverted_double_pendulum/" + time_string + ".txt", 'w')
    file.write(str(PARAMETERS) + "\n")
    for i in range(len(score_list)):
        file.write(str(score_list[i]) + ', ' + str(q_values[i]) + '\n')
