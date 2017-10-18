import mujoco_py as mj
import numpy as np
import random
from math import degrees, radians
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import gym

class Brain:
    def __init__(self, learning_rate, gamma, epsilon, epsilon_min, epsilon_decay):
        self.replay_memory = deque(maxlen=2000)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.net = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=4, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(2, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    # memory = [state, action, reward, new_state]
    def add_memory(self, memory):
        self.replay_memory.append(memory)

    def replay(self, batch_size):
        mini_batch = random.sample(self.replay_memory, batch_size)
        training_x = np.zeros((1, 4))
        training_y = np.zeros((1, 2))
        for state, action, reward, next_state, done in mini_batch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.net.predict(next_state)[0]))
            target_f = self.net.predict(state)
            target_f[0][action] = target

            training_x = np.vstack((training_x, state))
            training_y = np.vstack((training_y, target_f))

        np.delete(training_x, 0, axis=0)
        np.delete(training_y, 0, axis=0)

        self.net.fit(training_x, training_y, epochs=10, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class InvertedPendulum(Brain):
    def __init__(self, learning_rate, gamma, epsilon, epsilon_min, epsilon_decay):
        Brain.__init__(self, learning_rate, gamma, epsilon, epsilon_min, epsilon_decay)
        self.model = mj.load_model_from_path('xml/inverted_pendulum.xml')
        self.sim = mj.MjSim(self.model)

    def get_action(self):
        return np.array([[-1], [1]])

    def do_action(self, a):
        self.sim.data.ctrl[0] = a


class Environment:
    def __init__(self):
        self.inverted_pendulum = InvertedPendulum(learning_rate=0.001, gamma=0.95, epsilon=1, epsilon_min=0.01,
                                                  epsilon_decay=0.995)
        self.viewer = None
        self.sim = self.inverted_pendulum.sim
        self.initial_state = self.sim.get_state()

    def is_done(self):
        angle = degrees(self.sim.get_state().qpos[1])
        if 20. > angle > -20.:
            return False
        else:
            return True

    def get_reward(self):
        return 1


    def step(self, action):
        self.inverted_pendulum.do_action(action)
        self.sim.step()
        state = np.array([self.sim.get_state().qpos.tolist() + self.sim.get_state().qvel.tolist()])
        reward = self.get_reward()
        return state, reward, self.is_done()

    def render(self):
        if not self.viewer:
            self.viewer = mj.MjViewer(self.inverted_pendulum.sim)
        self.viewer.render()

    def reset(self):
        self.sim.set_state(self.initial_state)
        return np.array([self.initial_state.qpos.tolist() + self.initial_state.qvel.tolist()])


class Simulation(Environment):
    def __init__(self):
        self.env = Environment.__init__(self)
        self.epochs = 1000
        self.batch_size = 32
        self.env = gym.make('CartPole-v1')

    def run(self):
        for e in range(self.epochs):
            state = self.reset()
            state = self.env.reset()
            state = np.reshape(state, [1, 4])
            for time in range(500):

                #self.render()

                q_values = self.inverted_pendulum.net.predict(state)
                if random.random() < self.inverted_pendulum.epsilon:
                    index = random.randint(0, len(q_values[0]) - 1)
                else:
                    index = np.argmax(q_values)
                action = self.inverted_pendulum.get_action()[index]

                new_state, reward, done, _ = self.env.step(index)
                new_state = np.reshape(new_state, [1, 4])
                #new_state, reward, done = self.step(action)
                #reward = self.get_reward()
                reward = reward if not done else -10

                memory = (state, action, reward, new_state, done)
                self.inverted_pendulum.add_memory(memory)

                state = new_state
                if done:
                    print("episode: {}/{}, score: {}, e: {}".format(e, self.epochs, time,
                                                                    self.inverted_pendulum.epsilon))
                    break

            if len(self.inverted_pendulum.replay_memory) > self.batch_size:
                self.inverted_pendulum.replay(self.batch_size)
                # if e % 10 == 0:
                #     agent.save("./save/cartpole-dqn.h5")

simulation = Simulation()
simulation.run()
