from threading import Thread
import numpy as np
from Config import Config


class Predictor(Thread):
    def __init__(self, controller, id):
        Thread.__init__(self)
        self.setDaemon(True)

        self.id = id
        self.controller = controller
        self.exit_flag = False

    def run(self):
        ids = np.zeros(Config.PREDICTION_BATCH_SIZE, dtype=np.uint16)
        states = np.zeros((Config.PREDICTION_BATCH_SIZE, Config.NUM_STATES), dtype=np.float32)

        while not self.exit_flag:
            sample = 1

            ids[0], states[0] = self.controller.predictor_q.get()
            while sample < Config.PREDICTION_BATCH_SIZE and not self.controller.predictor_q.empty():
                ids[sample], states[sample] = self.controller.prediction_q.get()
                sample += 1

            batch = states[:sample]
            s_, v = self.controller.predict(batch)

            for i in range(sample):
                if ids[i] < len(self.controller.agents):
                    self.controller.agents[ids[i]].wait_q.put((s_[i], v[i]))
