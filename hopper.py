import mujoco_py as mj
import numpy as np
import random
from neural_network import NeuralNetwork


class Hopper:
    def __init__(self):
        self.model = mj.load_model_from_path('xml/hopper.xml')
        self.sim = mj.MjSim(self.model)
        self.initial_state = self.sim.get_state()
        self.brain = NeuralNetwork(layout=[3, 35, 20, 7], functions=['sigmoid', 'sigmoid', 'softplus'])

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
        self.learning_rate = 0.001
        self.discount_value = 0.9
        self.epsilon = 0.1
        self.reward = []
        self.memory_size = 10000
        self.replay_memory = np.zeros((self.memory_size, 10))

    def get_reward(self):
        x1 = self.hopper.sim.data.get_body_xpos('torso')[0]
        x2 = self.hopper.sim.data.get_body_xpos('thigh')[0]
        x3 = self.hopper.sim.data.get_body_xpos('leg')[0]
        x4 = self.hopper.sim.data.get_body_xpos('foot')[0]
        avg_position = (x1 + x2 + x3 + x4)/4

        if len(self.reward) == 0:
            self.reward.append(avg_position + 1)
            self.reward.append(0)

        self.reward[1] = avg_position + 1 - self.reward[0]
        self.reward[0] = avg_position + 1
        #return self.reward[-1]
        return avg_position

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

                new_state = self.hopper.get_state()
                new_q = self.hopper.brain.predict(new_state)
                max_new_q = np.max(new_q)

                print(np.argmax(new_q), np.average(new_q))

                if frames < self.memory_size:
                    self.replay_memory[frames][0:3] = past_state.tolist()
                    self.replay_memory[frames][3:11] = q_values.tolist()
                    if self.hopper.is_alive():
                        self.replay_memory[frames][index + 3] = reward + self.discount_value * max_new_q
                    else:
                        self.replay_memory[frames][index + 3] = -10
                else:
                    self.replay_memory = np.roll(self.replay_memory, -1, axis=0)
                    self.replay_memory[-1][0:3] = past_state.tolist()
                    self.replay_memory[-1][3:11] = q_values.tolist()
                    if self.hopper.is_alive():
                        self.replay_memory[-1][index + 3] = reward + self.discount_value * max_new_q
                    else:
                        self.replay_memory[-1][index + 3] = -10

                if frames % 481 == 0 and frames > self.memory_size:
                    print('Training')
                    sample = np.random.randint(self.memory_size, size=2000)
                    self.hopper.brain.train(self.replay_memory[sample, 0:3], self.replay_memory[sample, 3:11], 25, self.learning_rate)

            else:
                if not self.hopper.is_alive():
                    self.hopper.reset()

                self.hopper.sim.step()
                t += 0.002
                frames += 1
                self.viewer.render()

simulation = Simulation()
simulation.run()
