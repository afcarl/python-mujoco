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
        self.net = self._build_model()

    def load_model(self, path):
        del self.net
        self.net = load_model(path)

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
        self.net.save(path, overwrite=True)
        del self.net

    # memory = [state, action_number, reward, new_state]
    def add_memory(self, memory):
        self.replay_memory.append(memory)

    def replay(self):
        mini_batch = random.sample(self.replay_memory, self.batch_size)
        for state, action, reward, next_state, done in mini_batch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.net.predict(next_state)[0]))
            target_f = self.net.predict(state)
            target_f[0][action] = target

            # If the variable name does not exist create it and initiate it.
            try:
                training_x
                training_y
            except NameError:
                training_x = state
                training_y = target_f

            training_x = np.vstack((training_x, state))
            training_y = np.vstack((training_y, target_f))

        self.net.fit(training_x, training_y, epochs=self.epochs, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


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
        q_values = self.net.predict(state)
        if random.random() < self.epsilon:
            action = random.randint(0, len(q_values[0]) - 1)
        else:
            action = np.argmax(q_values)
        return action
