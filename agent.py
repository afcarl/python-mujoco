import mujoco_py as mj
import numpy as np
import random
from collections import deque
import os

# Set to run on cpu
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam


class Brain:
    def __init__(self, topology, epochs, memory_length, batch_size, learning_rate, gamma, epsilon, epsilon_min,
                 epsilon_decay):
        self.memory_length = memory_length
        self.replay_memory = deque(maxlen=self.memory_length)
        self.batch_size = batch_size
        self.topology = topology
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_network = self._build_model()
        self.q_network_target = self.q_network

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
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def save_model(self, path):
        self.q_network.save(path, overwrite=True)

    # memory = [state, action_number, reward, new_state]
    def add_memory(self, memory):
        self.replay_memory.append(memory)

    def train(self, x, y):
        self.q_network.fit(x, y, epochs=self.epochs, verbose=0)


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
        mini_batch = random.sample(self.replay_memory, self.batch_size)
        states = np.stack([state[0][0] for state in mini_batch])
        target = self.q_network.predict(states)

        for i in range(len(mini_batch)):
            a = mini_batch[i][1]
            r = mini_batch[i][2]
            s_ = mini_batch[i][3]

            if s_ is None:
                target[i][a] = r
            else:
                target[i][a] = (r + self.gamma * np.amax(self.q_network.predict(s_)))

        self.train(states, target)
