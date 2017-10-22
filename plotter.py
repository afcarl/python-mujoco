import matplotlib.pyplot as plt
import numpy as np
import os

files = os.listdir("./logs")
data = np.genfromtxt("./logs/" + files[3], comments="#")

for i in files:
    print(i)

N = 50
moving_mean = np.convolve(data, np.ones((N,))/N, mode='valid')
plt.scatter(range(len(data)), data)
plt.plot(range(int(N/2 - 1), len(data) - int(N/2)), moving_mean, 'r')
plt.show()