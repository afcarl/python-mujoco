from mujoco_py import MjViewer, MjSimState
import numpy as np
from math import degrees
import os

# Set to run on CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class Environment:
    def __init__(self, sim):
        self.sim = sim
        self.viewer = None
        self.initial_state = self.sim.get_state()

    def is_done(self):
        x_pos = self.sim.get_state().qpos[0]
        angle = degrees(self.sim.get_state().qpos[1])
        if (12.5 > angle > -12.5) and (0.99 > x_pos > -0.99):
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
            self.viewer = MjViewer(self.sim)
        self.viewer.render()

    def reset(self):
        random_pos = np.random.uniform(0., 0.05, size=(2))
        random_vel = np.random.uniform(0., 0.05, size=(2))
        self.initial_state = MjSimState(time=0.0, qpos=random_pos, qvel=random_vel, act=None, udd_state={})
        self.sim.set_state(self.initial_state)

        return np.array([self.sim.get_state().qpos.tolist() + self.sim.get_state().qvel.tolist()])
