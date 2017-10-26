import matplotlib.pyplot as plt
import numpy as np
import os
import ast

folder_path = "./logs/inverted_pendulum/"
PARAMETERS = {'model3Dpath': 'xml/inverted_pendulum.xml',
                  'topology': [[4, 24, 24, 2], ['relu', 'relu', 'linear']],
                  'memory_length': 10000,
                  'batch_size': 64,
                  'epochs': 1,
                  'learning_rate': 0.001,
                  'gamma': 0.99,
                  'epsilon': 1,
                  'epsilon_min': 0.01,
                  'epsilon_decay': 0.99}

files = os.listdir(folder_path)
for file in files:
    f = open(folder_path + file)
    line = f.readline()
    parameters = ast.literal_eval(line)
    if parameters == PARAMETERS:
        data = np.genfromtxt(folder_path + file, skip_header=1)
        plt.plot(range(len(data)), data)
        plt.show()
