from multiprocessing import Queue

from GA3C.Brain import Brain
from GA3C.Predictor import Predictor
from GA3C.Trainer import Trainer
from GA3C.Agent import Agent
from GA3C.Config import Config

class Controller:
    def __init__(self):
        self.training_q = Queue(maxsize=Config.MAX_QUEUE_SIZE)
        self.predictor_q = Queue(maxsize=Config.MAX_QUEUE_SIZE)

        self.agents = []
        self.predictors = []
        self.trainers = []

    def add_agent(self, id):
        self.agents.append(Agent(id, self.predictor_q, self.training_q, 1, 0.1))
        self.agents[-1].start()

    def add_predictor(self):
        return

    def add_trainer(self):
        return

    def remove_agent(self):
        return

    def remove_predictor(self):
        return

    def remove_trainer(self):
        return

    def main(self):
        return

if __name__ == '__main__':
    c = Controller()
    for i in range(1):
        c.add_agent(i)

