import numpy as np
from Actor import Actor
from Learner import Learner
import msvcrt
import os
import matplotlib.pyplot as plt

# Set to run on CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


if __name__ == "__main__":
    PARAMETERS = {'model3Dpath': 'xml/inverted_pendulum.xml',
                  'topology': [[4, 64, 2], ['relu', 'linear']],
                  'memory_length': 10000,
                  'batch_size': 128,
                  'epochs': 20,
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
                          PARAMETERS['epsilon_decay'], max_steps=200)
        actors[i].q_network = learner.q_network

    plt.ion()

    epochs = 1000
    max_steps = 501
    q_values = [0.]
    scores = []

    for e in range(epochs):
        if msvcrt.kbhit():
            if ord(msvcrt.getch()) == 59:
                break

        score_list = []
        for actor in actors:
            actor.q_network = learner.q_network

            state = actor.reset()
            for step in range(max_steps):

                new_state, action, reward, done = actor.observe(state)

                if done:
                    new_state = None

                _, _, errors = learner.get_targets([(0, (state, action, reward, new_state, done))])
                learner.add_memory(errors[0], (state, action, reward, new_state, done))

                state = new_state

                if done or step == max_steps-1:
                    score_list.append(step)
                    break

            # Decay the epsilon
            if actor.epsilon > actor.epsilon_min:
                actor.epsilon *= actor.epsilon_decay

        print("Episode: {}, Score: {}/{}".format(e, sum(score_list) / len(score_list), max_steps - 1))
        scores.append(sum(score_list) / len(score_list))
        x = range(len(scores))
        y = scores

        learner.replay()
        plt.scatter(x[-1], y[-1])
        plt.pause(0.05)

        if e % 5 == 0:
            learner.update_target()

    learner.save_model('./models/inverted_pendulum_v0.2.h5')
