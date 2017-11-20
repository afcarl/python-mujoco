from Config import Config
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Input
from keras import backend as K


class Brain:
    def __init__(self):
        self.session = tf.Session()
        K.set_session(self.session)
        K.manual_variable_initialization(True)

        self.model = self._build_model()
        self.graph = self._build_graph(self.model)

        self.session.run(tf.global_variables_initializer())
        self.default_graph = tf.get_default_graph()

        self.default_graph.finalize()

    def _build_model(self):

        input_layer = Input(batch_shape=(None, Config.NUM_STATES))
        dense_layer = Dense(16, activation='relu')(input_layer)

        actions = Dense(Config.NUM_ACTIONS, activation='softmax')(dense_layer)
        values = Dense(1, activation='linear')(dense_layer)

        model = Model(inputs=[input_layer], outputs=[actions, values])
        model._make_predict_function()

        return model

    def _build_graph(self, model):
        s = tf.placeholder(tf.float32, shape=(None, Config.NUM_STATES))
        a = tf.placeholder(tf.float32, shape=(None, Config.NUM_ACTIONS))
        r = tf.placeholder(tf.float32, shape=(None, 1))

        a_, v = model(s)

        log_prob = tf.log(tf.reduce_sum(a_ * a, axis=1, keep_dims=True) + 1e-10)
        advantage = r - v

        loss_policy = log_prob * tf.stop_gradient(advantage)
        loss_value = Config.LOSS_V * tf.square(advantage)
        entropy = Config.LOSS_ENTROPY * tf.reduce_sum(a_ * tf.log(a_ + 1e-10), axis=1, keep_dims=True)

        loss_total = tf.reduce_mean(loss_policy + loss_value + entropy)

        optimizer = tf.train.RMSPropOptimizer(Config.LEARNING_RATE, decay=0.99)
        minimize = optimizer.minimize(loss_total)

        return s, a, r, minimize

    def train(self, s, a, r):
        s_, a_, r_, minimize = self.graph
        self.session.run(minimize, feed_dict={s_: s, a_: a, r_: r})

    def predict(self, s):
        with self.default_graph.as_default():
            a_, v = self.model.predict(s)
            return a_, v
