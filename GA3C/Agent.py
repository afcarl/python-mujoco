from Environment import Environment
import multiprocessing
import numpy as np
from Config import Config


class Agent(multiprocessing.Process):
    def __init__(self, id, prediction_q, training_q, epsilon_max, epsilon_min, decay):
        multiprocessing.Process.__init__(self)
        self.id = id
        self.env = Environment('CartPole')
        self.prediction_q = prediction_q
        self.training_q = training_q
        self.wait_q = multiprocessing.Queue(maxsize=1)

        self.epsilon = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.decay = decay

        self.memory = []
        self.R = 0

    def update_epsilon(self):
        if self.epsilon <= self.epsilon_min:
            return self.epsilon_min
        else:
            return self.epsilon * self.decay

    def predict(self, state):
        self.prediction_q.put((self.id, state))
        a_, v = self.wait_q.get()
        return a_, v

    def select_action(self, state):
        self.epsilon = self.update_epsilon()
        if np.random.random() < self.epsilon:
            return np.random.randint(2)
        else:
            return np.random.choice(Config.NUM_ACTIONS, p=self.predict(state)[0])

    def run_episode(self):
        state = self.env.reset()
        R = 0
        while True:
            action = self.select_action(state)
            #self.env.render()
            new_state, reward, done, _ = self.env.step(action)

            if done:
                new_state = None

            self.train(state, action, reward, new_state)

            state = new_state
            R += reward

            if done:
                break

        print('Total Reward: ', R, self.epsilon)

    def train(self, s, a, r, s_):
        def sample(memory, n):
            s, a, _, _ = memory[0]
            _, _, _, s_ = memory[n-1]
            return s, a, self.R, s_

        self.memory.append((s, a, r, s_))
        self.R = (self.R + r * Config.GAMMA_N) / Config.GAMMA

        if s_ is None:
            while len(self.memory) > 0:
                n = len(self.memory)
                s, a, r, s_ = sample(self.memory, n)

                if s_ is None:
                    self.training_q.put((s, a, r, np.zeros(Config.NUM_STATES), 0.))
                else:
                    self.training_q.put((s, a, r, s_, 1.))

                self.R = (self.R - self.memory[0][2]) / Config.GAMMA
                self.memory.pop(0)

            self.R = 0

        if len(self.memory) >= Config.N_STEP_RETURN:
            s, a, r, s_ = sample(self.memory, Config.N_STEP_RETURN)

            if s_ is None:
                self.training_q.put((s, a, r, np.zeros(Config.NUM_STATES), 0.))
            else:
                self.training_q.put((s, a, r, s_, 1.))

            self.R = self.R - self.memory[0][2]
            self.memory.pop(0)

    def run(self):
        while True:
            self.run_episode()
