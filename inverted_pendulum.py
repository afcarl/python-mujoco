import mujoco_py as mj
import numpy as np
from Actor import Actor
from Learner import Learner
import msvcrt
import os

# Set to run on CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


if __name__ == "__main__":
    PARAMETERS = {'model3Dpath': 'xml/inverted_pendulum.xml',
                  'topology': [[4, 64, 2], ['relu', 'linear']],
                  'memory_length': 100000,
                  'batch_size': 128,
                  'epochs': 15,
                  'learning_rate': 0.001,
                  'gamma': 0.99,
                  'epsilon': 1,
                  'epsilon_min': 0.1,
                  'epsilon_decay': 0.995}

    learner = Learner(PARAMETERS['topology'], PARAMETERS['epochs'], PARAMETERS['memory_length'],
                      PARAMETERS['batch_size'], PARAMETERS['learning_rate'], PARAMETERS['gamma'],
                      PARAMETERS['epsilon'], PARAMETERS['epsilon_min'], PARAMETERS['epsilon_decay'])

    # Create the actors
    n_actors = 5
    actors = np.zeros(n_actors, dtype=object)
    sims = np.zeros(n_actors, dtype=object)
    for i in range(n_actors):
        actors[i] = Actor(PARAMETERS['model3Dpath'], np.random.uniform(0.5, 1.), PARAMETERS['epsilon_min'],
                          PARAMETERS['epsilon_decay'])
        actors[i].q_network = learner.q_network

    epochs = 1000
    max_steps = 1001
    score_list = []
    q_values = [0.]

    for e in range(epochs):
        if msvcrt.kbhit():
            if ord(msvcrt.getch()) == 59:
                break

        for actor in actors:
            state = actor.reset()
            for step in range(max_steps):

                new_state, action, reward, done = actor.observe(state)

                if done:
                    new_state = None

                _, _, errors = learner.get_targets([(0, (state, action, reward, new_state, done))])
                learner.add_memory(errors[0], (state, action, reward, new_state, done))

                state = new_state

                if done or step == max_steps-1:
                    print("Episode: {}, Score: {}/{}, epsilon: {}".format(e, step, max_steps-1, round(actor.epsilon, 2)))
                    score_list.append(step)
                    break

            # Decay the epsilon
            if actor.epsilon > actor.epsilon_min:
                actor.epsilon *= actor.epsilon_decay

            actor.q_network = learner.q_network

        learner.replay()

        if e % 5 == 0:
            # print("Updated Target Network")
            learner.update_target()

    learner.save_model('./models/inverted_pendulum_v0.2.h5')
