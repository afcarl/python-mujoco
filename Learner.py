from Memory import Memory
import numpy as np
import os

# Set to run on CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import RMSprop


class Learner(Memory):
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
                      optimizer=RMSprop(lr=self.learning_rate))
        return model

    def save_model(self, path):
        self.q_network.save(path, overwrite=True)

    def train(self, x, y):
        self.q_network.fit(x, y, epochs=self.epochs, verbose=0)

    def update_target(self):
        self.q_network_target.set_weights(self.q_network.get_weights())

    def get_targets(self, batch):
        states = np.stack([state[1][0][0] for state in batch])
        errors = np.zeros(len(batch))
        target = self.q_network.predict(states)

        for i in range(len(batch)):
            a = batch[i][1][1]
            r = batch[i][1][2]
            s_ = batch[i][1][3]

            old_val = target[i][a]
            if s_ is None:
                target[i][a] = r
            else:
                target[i][a] = r + self.gamma * self.q_network_target.predict(s_)[0][
                    np.argmax(self.q_network.predict(s_))]

            errors[i] = abs(old_val - target[i][a])

        return (states, target, errors)

    def replay(self):
        mini_batch = self.sample_memory(self.batch_size)
        states, target, errors = self.get_targets(mini_batch)

        for i in range(len(mini_batch)):
            self.update_memory(mini_batch[i][0], errors[i])
        self.train(states, target)
