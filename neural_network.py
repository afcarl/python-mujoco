import numpy as np
import random
from activation_functions import activation

np.seterr(over='ignore')


class NeuralNetwork:
    def __init__(self, layout, functions):
        # Constructors
        self.layout = layout
        self.activation_functions = functions
        self.error = 0

        # Creating a list with all the numpy arrays. These are make by reading the layout list above.
        self.weights = [0] * (len(layout) - 1)
        for i in range(len(layout) - 1):
            self.weights[i] = np.random.uniform(low=-1.0, high=1.0, size=(self.layout[i], self.layout[i + 1]))

        # Store the results of the layers during the feed forward pass.
        self.output_layer = [0] * (len(self.weights)+1)
        self.input_layer = [0] * len(self.weights)

        # Store the gradients and deltas of the network
        self.gradients = [0] * (len(self.weights))
        self.delta = [0] * (len(self.weights))

    def forward(self, X):
        # Takes the given X and forwards it through the network with the randomly generated weights.
        self.output_layer[0] = X
        for w in range(len(self.weights)):
            self.input_layer[w] = np.dot(self.output_layer[w], self.weights[w])
            self.output_layer[w + 1] = activation(self.input_layer[w], self.activation_functions[w], derivative=False)
        return self.output_layer[-1]

    def errorFunction(self, X, Y):
        # Calculate the cost function or how close the network is to match the weights to give the correct output.
        y = self.forward(X)
        return 1 / len(X) * sum(sum((Y - y) ** 2.0))

    def errorFunctionPrime(self, X, Y):
        y = self.forward(X)

        self.delta[0] = np.multiply(-(Y - y), activation(self.input_layer[-1], self.activation_functions[-1], derivative=True))
        self.gradients[0] = np.dot(self.output_layer[-2].T, self.delta[0])

        for w in range(len(self.gradients) - 1):
            self.delta[w + 1] = np.multiply(np.dot(self.delta[w], self.weights[-1 - w].T), activation(self.input_layer[-2 - w],
                                                                    self.activation_functions[-1-w], derivative=True))
            self.gradients[w + 1] = np.dot(self.output_layer[-3 - w].T, self.delta[w + 1])

        self.gradients = self.gradients[::-1]
        self.delta = self.delta[::-1]
        return self.gradients, self.delta

    def train(self, trainX, trainY, epochs, learning_rate):
        self.error = [0] * epochs
        for i in range(epochs):
            for j in range(len(self.weights)):
                self.weights[j] = np.subtract(self.weights[j], learning_rate * self.errorFunctionPrime(trainX, trainY)[0][j])
            self.error[i] = self.errorFunction(trainX, trainY)*100

