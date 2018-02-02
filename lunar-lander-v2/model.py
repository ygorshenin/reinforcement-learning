import tensorflow as tf

from env import *


HIDDEN_UNITS = 64


class Model:
    def __init__(self):
        self.state = tf.placeholder(tf.float32, shape=[None, STATES_DIM], name='state')
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        self.q_curr = tf.placeholder(tf.float32, shape=[None, ACTIONS_DIM], name='q_curr')

        W1 = tf.layers.dense(inputs=self.state,
                             units=HIDDEN_UNITS,
                             activation=tf.tanh,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             name='W1')
        W2 = tf.layers.dense(inputs=W1,
                             units=HIDDEN_UNITS,
                             activation=tf.tanh,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             name='W2')
        self.q_pred = tf.layers.dense(inputs=W2,
                                      units=ACTIONS_DIM,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                      name='q_pred')

        optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        self.loss = tf.losses.mean_squared_error(self.q_curr, self.q_pred)
        self.train_op = optimizer.minimize(self.loss)

    def predict(self, s, session):
        feed_dict = {self.state: s}
        return session.run(self.q_pred, feed_dict=feed_dict)

    def train(self, ss, qs, lr, session):
        feed_dict = {self.state: ss, self.q_curr: qs, self.learning_rate: lr}
        return session.run(self.train_op, feed_dict=feed_dict)
