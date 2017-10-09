import mujoco_py as mj
import numpy as np

class Q_learning:
    def __init__(self):
        # 4D space where the first 3 are the 3 states/joints and the last dimension is the Q values for every action.
        self.Q = np.zeros((21,21,21,7))

    def get_q(self, s, a):
        # State is a 1*3 array with forces on every joint.
        converted_state = (np.round(s, 1) + 1) * 10
        action_values = self.Q[int(converted_state[0]), int(converted_state[1]), int(converted_state[2])]
        value = action_values[a]
        return value

    def set_q(self, s, a, q):
        converted_state = (np.round(s, 1) + 1) * 10
        action_values = self.Q[int(converted_state[0]), int(converted_state[1]), int(converted_state[2])]
        action_values[a] = q

    def update_q(self, s, a, r, s_next, alpha, gamma):
        max_q = 0
        for i in range(7):
            if self.get_q(s, i > max_q):
                max_q = self.get_q(s_next, i)

        new_q = self.get_q(s, a) + alpha * (r + gamma - max_q - self.get_q(s, a))
        self.set_q(s, a, new_q)

class Hopper:
    def __init__(self):
        self.model = mj.load_model_from_path('xml/hopper.xml')
        self.sim = mj.MjSim(self.model)
        self.initial_state = self.sim.get_state()

    def get_state(self):
        return self.sim.data.ctrl

    def do_action(self, a):
        self.sim.data.ctrl = a
        return self.sim.data.ctrl

    def reset(self):
        self.sim.set_state(self.initial_state)


class Simulation:
    def __init__(self):
        self.Q = Q_learning()
        self.hopper = Hopper()
        self.viewer = mj.MjViewer(self.hopper.sim)

    def get_r(self):
        r = 1
        return r

    def run(self):
        t = 0
        while True:
            self.hopper.sim.step()
            t += 0.002

            self.viewer.render()

'''''
TO-DO: - hopper needs a is_alive function to check if it is still standing
       - simulation needs to run with a time cap or stop earlier if is_alive = False
       - new class EGreedy where we calculate the next move 
       - write a valid reward function that promotes walking/hopping and no crawling
'''''

