import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer

PROB_EPS = 1e-3


class PolicyDQN:
    def __init__(self, **kwargs):
        with tf.variable_scope('policy'):
            self.__init_model__(**kwargs)

    def __init_model__(self, input_shape, hidden_units=128):
        self.states = tf.placeholder(tf.float32, [None] + list(input_shape), name='states')
        self.weights = tf.placeholder(tf.float32, [None, 1], name='weights')
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        h1 = tf.layers.dense(inputs=tf.layers.flatten(self.states),
                             units=512,
                             activation=tf.nn.tanh,
                             kernel_initializer=xavier_initializer())
        h2 = tf.layers.dense(inputs=h1,
                             units=hidden_units,
                             activation=tf.nn.tanh,
                             kernel_initializer=xavier_initializer())
        self.probs = tf.layers.dense(inputs=h2,
                                     units=1,
                                     activation=tf.sigmoid,
                                     kernel_initializer=xavier_initializer())
        loss = -tf.reduce_mean(self.probs * self.weights)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = optimizer.minimize(loss)

    def predict(self, sess, state):
        return self.predict_on_batch(sess, [state])[0][0]

    def predict_on_batch(self, sess, states):
        return sess.run(self.probs, feed_dict={self.states: states})

    def train_on_batch(self, sess, states, weights, learning_rate):
        return sess.run(self.train_op, feed_dict={self.states: states,
                                                  self.weights: weights,
                                                  self.learning_rate: learning_rate})
