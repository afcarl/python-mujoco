import mujoco_py as mj
import numpy as np
import random
from neural_network import NeuralNetwork


class Hopper:
    def __init__(self):
        self.model = mj.load_model_from_path('xml/hopper.xml')
        self.sim = mj.MjSim(self.model)
        self.initial_state = self.sim.get_state()
        self.brain = NeuralNetwork(layout=[3, 7, 7], functions=['sigmoid', 'sigmoid'])

    def get_state(self):
        return self.sim.get_state().qpos[3:6]

    def get_action(self):
        return np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])

    def do_action(self, a):
        self.sim.data.ctrl[0:] = a
        return self.get_state()

    def reset(self):
        self.sim.set_state(self.initial_state)

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
        self.epsilon = 0.9
        self.reward = []
        self.replay_memory = np.empty([4], dtype=object)

    def get_reward(self):
        x1 = self.hopper.sim.data.get_body_xpos('torso')[0]
        x2 = self.hopper.sim.data.get_body_xpos('thigh')[0]
        x3 = self.hopper.sim.data.get_body_xpos('leg')[0]
        x4 = self.hopper.sim.data.get_body_xpos('foot')[0]
        avg_position = (x1 + x2 + x3 + x4)/4

        if len(self.reward) == 0:
            self.reward.append(avg_position)
            self.reward.append(0)

        self.reward[1] = avg_position - self.reward[0]
        self.reward[0] = avg_position
        return self.reward[-1]

    def run(self):
        t = 0
        frames = 0
        while True:
            # Lower the actions per second to about 60
            if frames % 8 == 0:
                past_state = self.hopper.get_state()
                q_values = self.hopper.brain.predict(past_state)

                if random.random() < self.epsilon:
                    index = random.randint(0, len(q_values) - 1)
                else:
                    index = np.argmax(q_values)
                action = self.hopper.get_action()[index]

                self.hopper.do_action(action)
                self.hopper.sim.step()
                t += 0.002
                frames += 1

                reward = self.get_reward()

                memory = np.array([past_state, index, reward, self.hopper.get_state()])
                if len(self.replay_memory) < 2000:
                    self.replay_memory = np.vstack((self.replay_memory, memory))
                else:
                    self.replay_memory = np.delete(self.replay_memory, 0, 0)
                    self.replay_memory = np.vstack((self.replay_memory, memory))

            else:
                if not self.hopper.is_alive():
                    self.hopper.reset()

                self.hopper.sim.step()
                t += 0.002
                frames += 1
                self.viewer.render()


simulation = Simulation()
simulation.run()
