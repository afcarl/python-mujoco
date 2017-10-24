import tensorflow as tf


class NeuralNet:
    def __init__(self, topology):
        self.layout = topology[0]
        self.activation_functions = topology[1]

        with tf.name_scope('input'):
            self.x = tf.placeholder(tf.float32, (None, self.layout[0]))
            self.y = tf.placeholder(tf.float32, (None, self.layout[-1]))

        # Creating a list with all the weights for every layer. These are made by reading the layout list above.
        self.weights = [0] * (len(self.layout) - 1)
        for i in range(len(self.weights)):
            self.weights[i] = tf.Variable(tf.random_uniform([self.layout[i], self.layout[i + 1]], -1, 1),
                                          name='W' + str(i+1))

        # Creating a list with all the biases for every layer.
        self.bias = [0] * (len(self.layout) - 1)
        for i in range(len(self.bias)):
            self.bias[i] = tf.Variable(tf.zeros([self.layout[i+1]]), name='b' + str(i + 1))

        # Store the results of the layers during the feedforward pass.
        self.output_layer = [0] * (len(self.weights) + 1)
        self.input_layer = [0] * len(self.weights)

        # Store the gradients and deltas of the network
        self.gradients = [0] * (len(self.weights))
        self.delta = [0] * (len(self.weights))

        # Tensorboard scalars
        tf.summary.scalar('error', self.error_function())
        self.merged_summary_op = tf.summary.merge_all()

        # Initialize the variables
        self.sess = tf.Session()
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

        logs_path = './tmp/tensorflow_logs'
        self.summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    def forward(self):
        # Takes the given X and forwards it through the network with the randomly generated weights.
        self.output_layer[0] = self.x
        for w in range(len(self.weights)):
            self.input_layer[w] = tf.matmul(self.output_layer[w], self.weights[w]) + self.bias[w]
            if self.activation_functions[w] == 'relu':
                self.output_layer[w + 1] = tf.nn.relu(self.input_layer[w])
            elif self.activation_functions[w] == 'linear':
                self.output_layer[w + 1] = self.input_layer[w]

        return self.output_layer[-1]

    def error_function(self):
        error = tf.reduce_sum(tf.square(self.y - self.forward()))
        return error

    def train(self, training_x, training_y, learning_rate, epochs):
        trainer = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.error_function())

        for epoch in range(epochs):
            _, e, summery = self.sess.run([trainer, self.error_function(), self.merged_summary_op], feed_dict={self.x: training_x, self.y: training_y})
            self.summary_writer.add_summary(summery, epoch)

    def predict(self, x):
        return self.sess.run(self.forward(), feed_dict={self.x: x})




