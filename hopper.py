import mujoco_py as mj
import tensorflow as tf
from math import *

model = mj.load_model_from_path('xml/hopper.xml')
sim = mj.MjSim(model)
viewer = mj.MjViewer(sim)

t = 0
while True:
    pos_before = sim.data.qpos[0]

    sim.data.ctrl[2] = cos(t / 0.2) * 10
    sim.data.ctrl[0] = cos(t / 0.05) * 1

    sim.step()
    t += 0.002

    pos_after = sim.data.qpos[0]
    alive_bonus = 1
    reward = (pos_after - pos_before) / 0.002
    reward += alive_bonus

    print(reward)

    viewer.render()