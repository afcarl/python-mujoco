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
            s_batch, a_batch, r_batch, s_mask_batch = None, None, None, None
            while batch_size <= Config.TRAINING_MIN_BATCH_SIZE:
                s_original, a_original, r_original, s__original, s_mask = self.controller.training_q.get()

                s = np.array([s_original])
                a = np.zeros((1, Config.NUM_ACTIONS))
                a[0][a_original] = 1
                r = np.array([[r_original]])
                s_mask = np.array([[s_mask]])

                if batch_size == 0:
                    s_batch = s
                    a_batch = a
                    r_batch = r
                    s_mask_batch = s_mask
                else:
                    s_batch = np.concatenate((s_batch, s))
                    a_batch = np.concatenate((a_batch, a))
                    r_batch = np.concatenate((r_batch, r))
                    s_mask_batch = np.concatenate((s_mask_batch, s_mask))
                batch_size += s.shape[0]

            v = self.controller.predict(s_batch)[1]
            r_batch = r_batch + Config.GAMMA_N * v * s_mask_batch

            self.controller.train(s_batch, a_batch, r_batch)

