import matplotlib.pyplot as plt
import numpy as np
import os
import ast

folder_path = "./logs/inverted_pendulum/"
PARAMETERS = {'model3Dpath': 'xml/inverted_pendulum.xml',
                  'topology': [[4, 24, 24, 2], ['relu', 'relu', 'linear']],
                  'memory_length': 2000,
                  'batch_size': 32,
                  'epochs': 100,
                  'learning_rate': 0.001,
                  'gamma': 0.995,
                  'epsilon': 1,
                  'epsilon_min': 0.1,
                  'epsilon_decay': 0.995}

files = os.listdir(folder_path)
for file in files:
    f = open(folder_path + file)
    line = f.readline()
    parameters = ast.literal_eval(line)
    if parameters == PARAMETERS:
        data = np.genfromtxt(folder_path + file, skip_header=1)
        N = 50
        moving_mean = np.convolve(data, np.ones((N,))/N, mode='valid')
        plt.scatter(range(len(data)), data)
        plt.plot(range(int(N/2 - 1), len(data) - int(N/2)), moving_mean, 'r')
        plt.show()
