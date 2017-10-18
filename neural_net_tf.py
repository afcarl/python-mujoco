import tflearn


class NeuralNet:
    def __init__(self, learning_rate, layout):
        self.learning_rate = learning_rate
        self.layout = layout
        self.net = self.model()

    def model(self):
        net = tflearn.layers.core.input_data(shape=[None, self.layout[0]], name='input')

        net = tflearn.layers.core.fully_connected(net, self.layout[1], activation='sigmoid')

        net = tflearn.layers.core.fully_connected(net, self.layout[2], activation='sigmoid')
        net = tflearn.layers.estimator.regression(net, optimizer='adam', learning_rate=self.learning_rate,
                                                  loss='categorical_crossentropy', name='targets')
        model = tflearn.DNN(net, tensorboard_verbose=0)

        return model

    def train(self, train_x, train_y, epochs):
        self.net.fit({'input': train_x}, {'targets': train_y}, n_epoch=epochs, show_metric=False)
        return self.net

    def predict(self, x):
        return self.net.predict(x)
