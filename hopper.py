import mujoco_py as mj
import tensorflow as tf
from math import *
import numpy as np

class Qlearning:
    def __init__(self):
        # 4D space where the first 3 are the 3 states/joints and the last dimension is the Q values for every action.
        self.Q = np.zeros((21,21,21,7))

    def getQ(self, s, a):
        # State is a 1*3 array with forces on every joint.
        converted_state = (np.round(s, 1) + 1) * 10
        action_values = self.Q[int(converted_state[0]), int(converted_state[1]), int(converted_state[2])]
        value = action_values[a]
        return value

    def setQ(self, s, a, q):
        converted_state = (np.round(s, 1) + 1) * 10
        action_values = self.Q[int(converted_state[0]), int(converted_state[1]), int(converted_state[2])]
        action_values[a] = q

    def updateQ(self, s, a, r, s_next, alpha, gamma):
        max_Q = 0
        for i in range(7):
            if self.getQ(s, i > max_Q):
                max_Q = self.getQ(s_next, i)

        new_Q = self.getQ(s, a) + alpha * (r + gamma - max_Q - self.getQ(s, a))
        self.setQ(s, a, new_Q)

Q = Qlearning()
model = mj.load_model_from_path('xml/hopper.xml')
sim = mj.MjSim(model)
viewer = mj.MjViewer(sim)

t = 0
while True:

    pos_before = sim.data.qpos[0]

    sim.step()
    t += 0.002

    pos_after = sim.data.qpos[0]

    viewer.render()