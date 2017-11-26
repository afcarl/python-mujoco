from threading import Thread
from Config import Config
import numpy as np


class Trainer(Thread):
    def __init__(self, controller, id):
        Thread.__init__(self)
        self.setDaemon(True)

        self.id = id
        self.controller = controller
        self.exit_flag = False

    def run(self):
        while not self.exit_flag:
            batch_size = 0
            while batch_size <= Config.TRAINING_MIN_BATCH_SIZE:
                s, a, r = self.controller.training_q.get()
                s = np.array([s])
                a = np.array([[a]])
                r = np.array([[r]])

                if batch_size == 0:
                    s_batch = s
                    a_batch = a
                    r_batch = r
                else:
                    s_batch = np.concatenate((s_batch, s))
                    a_batch = np.concatenate((a_batch, a))
                    r_batch = np.concatenate((r_batch, r))
                batch_size += s.shape[0]

            self.controller.train(s_batch, a_batch, r_batch)

