from GA3C.Environment import Environment
import multiprocessing
import numpy as np
from Config import Config


class Agent(multiprocessing.Process):
    def __init__(self, id, prediction_q, training_q, epsilon, epsilon_min):
        multiprocessing.Process.__init__(self)
        self.id = id
        self.env = Environment('CartPole')
        self.prediction_q = prediction_q
        self.training_q = training_q
        self.wait_q = multiprocessing.Queue(maxsize=1)
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min

    def predict(self, state):
        self.prediction_q.put((self.id, state))
        a_, v = self.wait_q.get()
        return a_, v

    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(2)
        else:
            action = np.random.choice(Config.NUM_ACTIONS, p=self.predict(state)[0])
            return action

    def run_episode(self):
        state = self.env.reset()[0]
        done = False

        reward_sum = 0

        while not done:
            action = self.select_action(state)
            #self.env.render()
            new_state, reward, done, _ = self.env.step(action)
            reward_sum += reward
            yield new_state, reward, done, action, reward_sum

    def run(self):
        while True:
            for new_state, reward, done, action, reward_sum in self.run_episode():
                if done:
                    print('Agent ' + str(self.id) + ' scored: ' + str(reward_sum))