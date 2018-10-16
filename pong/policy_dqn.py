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
        self.actions = tf.placeholder(tf.float32, [None, 1], name='actions')

        conv1 = tf.layers.conv2d(inputs=self.states,
                                 filters=64,
                                 kernel_size=8,
                                 strides=4,
                                 data_format='channels_first',
                                 activation=tf.nn.tanh,
                                 kernel_initializer=xavier_initializer())
        h1 = tf.layers.dense(inputs=tf.layers.flatten(conv1),
                             units=128,
                             activation=tf.nn.tanh,
                             kernel_initializer=xavier_initializer())
        self.prob_1 = tf.layers.dense(inputs=h1,
                                      units=1,
                                      activation=tf.sigmoid,
                                      kernel_initializer=xavier_initializer())
        self.prob_1 = self.prob_1 * 0.9998 + 0.0001
        self.prob_0 = 1 - self.prob_1

        self.log_prob = tf.log(self.actions * self.prob_1 + (1 - self.actions) * (1 - self.prob_1))
        loss = tf.reduce_mean(self.log_prob * self.weights)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = optimizer.minimize(loss)

    def predict(self, sess, state):
        return self.predict_on_batch(sess, [state])[0][0]

    def predict_on_batch(self, sess, states):
        return sess.run(self.prob_1, feed_dict={self.states: states})

    def train_on_batch(self, sess, states, actions, weights, learning_rate):
        return sess.run(self.train_op, feed_dict={self.states: states,
                                                  self.actions: actions,
                                                  self.weights: weights,
                                                  self.learning_rate: learning_rate})
