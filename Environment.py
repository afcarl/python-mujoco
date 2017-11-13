import mujoco_py as mj
import numpy as np
from math import degrees
import time


class Environment:
    def __init__(self, sim):
        self.sim = sim
        self.viewer = None
        self.initial_state = self.sim.get_state()

    def is_done(self):
        x_pos = self.sim.get_state().qpos[0]
        angle = degrees(self.sim.get_state().qpos[1])
        if (20. > angle > -20.) and (0.99 > x_pos > -0.99):
            return False
        else:
            return True

    def get_reward(self):
        if not self.is_done():
            return 1
        else:
            return -1

    def render(self):
        if not self.viewer:
            self.viewer = mj.MjViewer(self.sim)
        self.viewer.render()

    def reset(self):
        self.sim.set_state(self.initial_state)
        #TODO Add random noise to the reset
        return np.array([self.sim.get_state().qpos.tolist() + self.sim.get_state().qvel.tolist()])
