import mujoco_py as mj
import numpy as np
import random
from math import degrees, radians
import tensorflow as tf
np.set_printoptions(suppress=True)
from neural_network import NeuralNetwork


class Brain:
    def __init__(self, layout, functions, memory_size):
        self.memory_size = memory_size
        self.replay_memory = np.zeros(memory_size)
        self.neural_net = NeuralNetwork(layout, functions)
        self.memories_saved = 0

    # memory = [state, action, reward, new_state]
    def add_memory(self, memory):
        if self.memories_saved < self.memory_size[0]:
            self.replay_memory[self.memories_saved] = memory
            self.memories_saved += 1
        else:
            self.replay_memory = np.roll(self.replay_memory, -1, axis=0)
            self.replay_memory[-1] = memory
            self.memories_saved += 1

    def train(self, train_x, train_y, epochs, learning_rate):
        self.neural_net.train(train_x, train_y, epochs, learning_rate)

    def predict(self, x):
        return self.neural_net.forward(x/1.57)*10.


class InvertedPendulum(Brain):
    def __init__(self, layout, functions, memory_size):
        Brain.__init__(self, layout, functions, memory_size)
        self.model = mj.load_model_from_path('xml/inverted_pendulum.xml')
        self.sim = mj.MjSim(self.model)
        self.initial_state = self.sim.get_state()

    def get_state(self):
        return self.sim.get_state().qpos[1]

    def get_action(self):
        return np.array([[0], [-1], [1]])

    def do_action(self, a):
        self.sim.data.ctrl[0] = a
        return self.get_state()

    def reset(self):
        self.sim.set_state(self.initial_state)

    def is_alive(self):
        if degrees(self.sim.get_state().qpos[1]) > 60 or degrees(self.sim.get_state().qpos[1]) < -60:
            return False
        else:
            return True


class Simulation:
    def __init__(self):
        self.inverted_pendulum = InvertedPendulum(layout=[1, 10, 3], functions=['sigmoid', 'sigmoid'], memory_size=(300, 4))
        self.viewer = mj.MjViewer(self.inverted_pendulum.sim)
        self.learning_rate = 0.01
        self.discount_value = 0.9
        self.epsilon = 1

    def get_reward(self):
        angle = degrees(self.inverted_pendulum.sim.get_state().qpos[1])
        if angle < 5 and angle > -5:
            reward = 10
        elif (angle > 5 and angle < 45) and (angle < -5 and angle > -45):
            reward = -1
        else:
            reward = -10
        return reward
    '''''

    def get_reward(self):
        angle = self.inverted_pendulum.sim.get_state().qpos[1]
        reward = min(50., 1 / angle) / 50
        return reward
    '''''

    def run(self):
        frames = -1

        while True:
            frames += 1
            # Lower the actions per second to about 60

            past_state = self.inverted_pendulum.get_state()
            q_values = self.inverted_pendulum.predict(past_state)

            if random.random() < self.epsilon:
                index = random.randint(0, len(q_values[0]) - 1)
            else:
                index = np.argmax(q_values)
            action = self.inverted_pendulum.get_action()[index]

            self.inverted_pendulum.do_action(action)
            self.inverted_pendulum.sim.step()

            reward = self.get_reward()
            new_state = self.inverted_pendulum.get_state()

            memory = [past_state] + [reward] + [index] + [new_state]
            self.inverted_pendulum.add_memory(memory)

            if frames > self.inverted_pendulum.memory_size[0]:
                sample_size = 100
                sample = self.inverted_pendulum.replay_memory[np.random.randint(self.inverted_pendulum.memory_size[0], size=sample_size)]
                if self.inverted_pendulum.is_alive():
                    train_x = sample[:, 0:1]
                    train_y = self.inverted_pendulum.predict(train_x)
                    max_q_prime = self.inverted_pendulum.predict(sample[:, 3:])
                    for i in range(len(train_y)):
                        train_y[i][int(sample[:, 2][i])] = sample[:, 1][i] + self.discount_value * np.max(max_q_prime[i])

            if frames % 300 == 0 and frames > self.inverted_pendulum.memory_size[0]:

                #
                # Looks like it is reverse learning meaning killing itself as fast as possible
                #

                print('Training')
                self.inverted_pendulum.train(train_x/1.57, train_y/10., 100000, self.learning_rate)
                print(train_x/1.57)
                print(self.inverted_pendulum.predict(train_x))
                print(self.inverted_pendulum.neural_net.error[0]/self.inverted_pendulum.neural_net.error[-1])
                self.epsilon *= 0.9

            else:

                if not self.inverted_pendulum.is_alive():
                    self.inverted_pendulum.reset()

                self.inverted_pendulum.sim.step()
                self.viewer.render()

simulation = Simulation()
simulation.run()
