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
        self.agents.append(Agent(id, self.predictor_q, self.training_q, Config.EPSILON_START, Config.EPSILON_END,
                                 Config.EPSILON_DECAY))
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
    for i in range(Config.N_PREDICTORS):
        c.add_predictor(i)

    for i in range(Config.N_TRAINERS):
        c.add_trainer(i)

    for i in range(Config.N_AGENTS):
        c.add_agent(i)

# FIXED Note currently only adding the final experience to the batch. Should be all the experiences!
# FIXED Add a shared memory and sample from that to train.
# FIXED Add reward decay
# Match results with reference code
# Try removing N-step-return

