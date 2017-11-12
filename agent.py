import mujoco_py as mj
import numpy as np
import random
from collections import deque
from sum_tree import SumTree
import os

# Set to run on CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam


class Memory:
    def __init__(self, memory_length):
        self.memory_length = memory_length
        self.tree = SumTree(memory_length)
        self.e = 0.01
        self.a = 0.6

    def _get_priority(self, error):
        return (error + self.e) ** self.a

    def add_memory(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample_memory(self, n):
        batch = []
        segment = self.tree.total() / n

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append( (idx, data) )

        return batch

    def update_memory(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)


class Brain(Memory):
    def __init__(self, topology, epochs, memory_length, batch_size, learning_rate, gamma, epsilon, epsilon_min,
                 epsilon_decay):
        Memory.__init__(self, memory_length)
        self.memory_length = memory_length
        self.batch_size = batch_size
        self.topology = topology
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_network = self._build_model()
        self.q_network_target = self._build_model()

    def load_model(self, path):
        del self.q_network
        del self.q_network_target
        self.q_network = load_model(path)
        self.q_network_target = self.q_network

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(self.topology[0][1], input_dim=self.topology[0][0], activation=self.topology[1][0]))
        for i in range(len(self.topology[0])-2):
            model.add(Dense(self.topology[0][i+2], activation=self.topology[1][i+1]))
        model.compile(loss='mean_squared_error',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def save_model(self, path):
        self.q_network.save(path, overwrite=True)

    def train(self, x, y):
        self.q_network.fit(x, y, epochs=self.epochs, verbose=0)

    def update_target(self):
        self.q_network_target.set_weights(self.q_network.get_weights())


class Agent(Brain):
    def __init__(self, parameters):
        Brain.__init__(self, parameters['topology'], parameters['epochs'], parameters['memory_length'],
                       parameters['batch_size'], parameters['learning_rate'], parameters['gamma'],
                       parameters['epsilon'], parameters['epsilon_min'], parameters['epsilon_decay'])
        self.model3D = mj.load_model_from_path(parameters['model3Dpath'])
        self.sim = mj.MjSim(self.model3D)

    def get_possible_actions(self):
        return np.array([[-1], [1]])

    def do_action(self, a):
        self.sim.data.ctrl[0:] = a

    def act(self, state):
        q_values = self.q_network.predict(state)
        if random.random() < self.epsilon:
            action = random.randint(0, len(q_values[0]) - 1)
        else:
            action = np.argmax(q_values)
        return action

    def replay(self):
        mini_batch = self.sample_memory(self.batch_size)
        states = np.stack([state[1][0][0] for state in mini_batch])
        errors = np.zeros(len(mini_batch))
        target = self.q_network.predict(states)

        for i in range(len(mini_batch)):
            a = mini_batch[i][1][1]
            r = mini_batch[i][1][2]
            s_ = mini_batch[i][1][3]

            old_val = target[i][a]
            if s_ is None:
                target[i][a] = r
            else:
                target[i][a] = r + self.gamma * self.q_network_target.predict(s_)[0][np.argmax(self.q_network.predict(s_))]

            errors[i] = abs(old_val - target[i][a])
            self.update_memory(mini_batch[i][0], errors[i])

        self.train(states, target)
