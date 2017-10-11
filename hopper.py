import mujoco_py as mj
import numpy as np
import random

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
            if self.get_q(s_next, i) > max_q:
                max_q = self.get_q(s_next, i)

        new_q = self.get_q(s, a) + alpha * (r + gamma - max_q - self.get_q(s, a))
        self.set_q(s, a, new_q)

    def get_action(self, index):
        if index == 0:
            return np.zeros(3)
        elif index == 1:
            return np.array([0.1, 0, 0])
        elif index == 2:
            return np.array([-0.1, 0, 0])
        elif index == 3:
            return np.array([0, 0.1, 0])
        elif index == 4:
            return np.array([0, -0.1, 0])
        elif index == 5:
            return np.array([0, 0, 0.1])
        elif index == 6:
            return np.array([0, 0, -0.1])

    def get_random_action(self, q_values):
        # a = [nothing, +, -, +, -, +, -]
        index = random.randint(0, len(q_values))
        return self.get_action(index)

    def get_best_action(self, q_values):
        same_q = len(np.where(q_values == max(q_values))[0])
        if same_q >= 2:
            index = random.randint(0, same_q-1)
        else:
            index = np.argmax(q_values)
        return self.get_action(index)

    def get_EGreedy_action(self, q_values, epsilon):
        random_float = random.random()
        if random_float < epsilon:
            return self.get_random_action(q_values)
        else:
            return self.get_best_action(q_values)

class Hopper:
    def __init__(self):
        self.model = mj.load_model_from_path('xml/hopper.xml')
        self.sim = mj.MjSim(self.model)
        self.initial_state = self.sim.get_state()

    def get_state(self):
        return self.sim.data.ctrl

    def do_action(self, a):
        self.sim.data.ctrl[0:] = a
        return self.sim.data.ctrl

    def reset(self):
        self.sim.set_state(self.initial_state)


class Simulation:
    def __init__(self):
        self.Q = Q_learning()
        self.hopper = Hopper()
        self.viewer = mj.MjViewer(self.hopper.sim)
        self.learning_rate = 0.1
        self.discount_value = 0.9
        self.epsilon = 0.5

    def get_r(self):
        r = 1
        return r

    def run(self):
        t = 0
        while True:


            old_state = self.hopper.get_state()
            action = self.Q.get_best_action(self.Q.get_q(self.hopper.get_state(), [range(7)]))
            new_state = self.hopper.do_action(action*10)
            print(new_state)
            #self.Q.update_q(old_state, action, self.get_r(), new_state, self.learning_rate, self.discount_value)

            self.hopper.sim.step()
            t += 0.002

            #print(self.Q.get_q(self.hopper.get_state(), [range(7)]))

            self.viewer.render()

simulation = Simulation()
simulation.run()

'''''
TO-DO: - hopper needs a is_alive function to check if it is still standing
       - simulation needs to run with a time cap or stop earlier if is_alive = False
       - new class EGreedy where we calculate the next move 
       - write a valid reward function that promotes walking/hopping and no crawling
'''''

