from GA3C.Environment import Environment
import multiprocessing
import time
import numpy as np


class Agent(multiprocessing.Process):
    def __init__(self, id, task_q, result_q, epsilon, epsilon_min):
        multiprocessing.Process.__init__(self)
        self.id = id
        self.env = Environment('CartPole')
        self.task_q = task_q
        self.result_q = result_q
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min

    def predict(self):
        return np.random.randint(2)

    def select_action(self):
        if np.random.random() < self.epsilon:
            return np.random.randint(2)
        else:
            return self.predict()

    def run_episode(self):
        self.env.reset()
        done = False

        reward_sum = 0

        while not done:
            action = self.select_action()
            #self.env.render()
            new_state, reward, done, _ = self.env.step(action)
            reward_sum += reward
            yield new_state, reward, done, action, reward_sum

    def run(self):
        time.sleep(np.random.rand())
        while True:
            for new_state, reward, done, action, reward_sum in self.run_episode():
                if done:
                    print('Agent ' + str(self.id) + ' scored: ' + str(reward_sum))