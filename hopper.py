import mujoco_py as mj
import numpy as np
from neural_net import NeuralNet


class Hopper:
    def __init__(self):
        self.model = mj.load_model_from_path('xml/hopper.xml')
        self.sim = mj.MjSim(self.model)
        self.initial_state = self.sim.get_state()
        self.brain = NeuralNet()

    def get_state(self):
        return self.sim.get_state().qpos[3:6]

    def do_action(self, a):
        self.sim.data.ctrl[0:] = a
        return self.get_state()

    def reset(self):
        self.sim.set_state(self.initial_state)
        print("Reset")

    def is_alive(self):
        if self.sim.data.get_body_xpos('torso')[2] > 0.3:
            return True
        else:
            return False


class Simulation:
    def __init__(self):
        self.hopper = Hopper()
        self.viewer = mj.MjViewer(self.hopper.sim)
        self.learning_rate = 0.1
        self.discount_value = 0.9
        self.epsilon = 0.5
        self.reward = []

    def get_reward(self):
        x1 = self.hopper.sim.data.get_body_xpos('torso')[0]
        x2 = self.hopper.sim.data.get_body_xpos('thigh')[0]
        x3 = self.hopper.sim.data.get_body_xpos('leg')[0]
        x4 = self.hopper.sim.data.get_body_xpos('foot')[0]
        avg_position = (x1 + x2 + x3 + x4)/4

        if len(self.reward) == 0:
            self.reward.append(avg_position)
            self.reward.append(0)

        self.reward[1] = self.reward[0] - avg_position
        self.reward[0] = avg_position
        return self.reward[-1]

    def run(self):
        t = 0
        while True:

            if not self.hopper.is_alive():
                self.hopper.reset()

            sim_state = self.hopper.sim.get_state()
            #sim_state.qpos[5] = +0.01
            self.hopper.sim.set_state(sim_state)

            self.hopper.sim.step()
            t += 0.002

            #print(self.hopper.sim.model.get_joint_qpos_addr('foot_joint'))

            self.viewer.render()

simulation = Simulation()
simulation.run()
