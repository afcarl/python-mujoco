from GA3C.Environment import Environment
import multiprocessing
import time


class Agent(multiprocessing.Process):
    def __init__(self, id, task_q, result_q):
        multiprocessing.Process.__init__(self)
        self.id = id
        self.environment = Environment('xml/inverted_pendulum.xml')
        self.task_q = task_q
        self.result_q = result_q

    def predict(self):
        return

    def select_action(self):
        return

    def run_episode(self):
        return

    def run(self):
        while True:
            print('Running agent code + ' + str(self.id))
            time.sleep(0.5)