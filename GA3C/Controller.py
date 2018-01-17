from multiprocessing import Queue
from Brain import Brain
from Predictor import Predictor
from Trainer import Trainer
from Agent import Agent
from Config import Config


class Controller(Brain):
    def __init__(self):
        Brain.__init__(self)
        self.training_q = Queue(maxsize=Config.MAX_QUEUE_SIZE)
        self.predictor_q = Queue(maxsize=Config.MAX_QUEUE_SIZE)

        self.agents = []
        self.predictors = []
        self.trainers = []

    def add_agent(self, id):
        self.agents.append(Agent(id, self.predictor_q, self.training_q, 0.75, 0.1))
        self.agents[-1].start()

    def add_predictor(self, id):
        self.predictors.append(Predictor(self, id))
        self.predictors[-1].start()
        return

    def add_trainer(self, id):
        self.trainers.append(Trainer(self, id))
        self.trainers[-1].start()
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
        c.add_predictor(i)

    for i in range(1):
        c.add_trainer(i)

    for i in range(4):
        c.add_agent(i)

# FIXED Note currently only adding the final experience to the batch. Should be all the experiences!
# Add a shared memory and sample from that to train.
# Add reward decay


