from mujoco_py import MjSim, load_model_from_path


class Environment:
    def __init__(self, model_path):
        self.name = 'TestEnvironment'
        self.model = load_model_from_path(model_path)
        self.sim = MjSim(self.model)
        self.initial_state = self.sim.get_state()

    def get_state(self):
        return self.sim.get_state()

    def reset(self):
        self.sim.set_state(self.initial_state)

    def step(self):
        self.sim.step()

