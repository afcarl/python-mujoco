import numpy as np
import random
from keras.models import load_model
from mujoco_py import load_model_from_path, MjSim
from Environment import Environment
import os

# Set to run on CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class Actor(Environment):
    def __init__(self, model_path, epsilon, epsilon_min, epsilon_decay, max_steps):
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model3D = load_model_from_path(model_path)
        self.sim = MjSim(self.model3D)
        self.q_network = None
        self.max_steps = max_steps
        Environment.__init__(self, self.sim)

    def load_model(self, path):
        self.q_network = load_model(path)

    def get_possible_actions(self):
        return np.array([[-1], [1]])

    def do_action(self, a):
        self.sim.data.ctrl[0:] = a

    def act(self, state):
        q_values = self.q_network.predict(state)
        if random.random() < self.epsilon:
            action = random.randint(0, len(q_values[0]) - 1)
        else:
            action = np.argmax(q_values)
        return action

    def observe(self, state):
        action = self.act(state)
        self.do_action(self.get_possible_actions()[action])
        self.sim.step()
        s = np.array([self.sim.get_state().qpos.tolist() + self.sim.get_state().qvel.tolist()])
        r = self.get_reward()
        return s, action, r, self.is_done()
