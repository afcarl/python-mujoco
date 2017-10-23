import tensorflow as tf


parameters = {'topology': [[6, 24, 24, 2], ['relu', 'relu', 'linear']]}

class NeuralNet:
    def __init__(self, topology, learning_rate):
        self.topology = topology
        self.learning_rate = learning_rate
        self.x = tf.placeholder(tf.float32, (None, self.topology[0][0]))
        self.y = tf.placeholder(tf.float32, (None, self.topology[0][-1]))
        self.logits = self.model()

        # Initialize the variables
        self.init = tf.global_variables_initializer()
        tf.Session().run(self.init)

    def model(self):
        layer = tf.layers.dense(inputs=self.x, units=self.topology[0][1], activation=self.topology[1][0])
        for i in range(len(self.topology[0]) - 2):
            layer = tf.layers.dense(inputs=layer, units=self.topology[0][i + 2], activation=self.topology[1][i + 1])
        return layer

    def train(self, training_x, training_y, epochs):
        loss_operator = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.model(), labels=self.y))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        train_operator = optimizer.minimize(loss_operator)

        with tf.Session() as sess:
            for epoch in range(epochs):
                _, cost = sess.run([train_operator, loss_operator], feed_dict={self.x: training_x, self.y: training_y})
                sess.run([self.logits])

    def predict(self, x):
        return self.logits




