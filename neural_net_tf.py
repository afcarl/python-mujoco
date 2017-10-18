import tflearn


class NeuralNet:
    def __init__(self, learning_rate, layout):
        self.learning_rate = learning_rate
        self.layout = layout

    def model(self):
        net = tflearn.layers.core.input_data(shape=[None, self.layout[0]], name='input')

        net = tflearn.layers.core.fully_connected(net, self.layout[1], activation='sigmoid')

        net = tflearn.layers.core.fully_connected(net, self.layout[2], activation='softmax')
        net = tflearn.layers.estimator.regression(net, optimizer='sgd', learning_rate=self.learning_rate,
                                                  loss='categorical_crossentropy', name='targets')
        model = tflearn.DNN(net, tensorboard_verbose=0)

        return model

    def train(self, train_x, train_y, epochs,model=False):
        if not model:
            model = self.model()

        model.fit({'input': train_x}, {'targets': train_y}, n_epoch=epochs, show_metric=True)
        return model

    def predict(self, x, model=False):
        if not model:
            model = self.model()
        return model.predict(x)
