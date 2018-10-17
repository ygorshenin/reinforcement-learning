import numpy as np
import tensorflow as tf

from tensorflow.contrib.layers import xavier_initializer


class DQN:
    def __init__(self, input_shape):
        self.ss = tf.placeholder(tf.float32, [None] + list(input_shape))
        self.qs_true = tf.placeholder(tf.float32, [None, 2])
        self.lr = tf.placeholder(tf.float32)

        conv1 = tf.layers.conv2d(inputs=self.ss,
                                 filters=64,
                                 kernel_size=8,
                                 strides=4,
                                 activation=tf.nn.relu,
                                 data_format='channels_first',
                                 kernel_initializer=xavier_initializer())
        h1 = tf.layers.dense(inputs=tf.layers.flatten(conv1),
                             units=256,
                             activation=tf.nn.tanh,
                             kernel_initializer=xavier_initializer())
        self.qs_pred = tf.layers.dense(inputs=h1,
                                       units=2,
                                       kernel_initializer=xavier_initializer())

        loss = tf.losses.mean_squared_error(labels=self.qs_true, predictions=self.qs_pred)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_op = optimizer.minimize(loss)

    def predict(self, sess, s):
        return sess.run(self.qs_pred, feed_dict={self.ss: np.array([s])})[0]

    def predict_on_batch(self, sess, ss):
        return sess.run(self.qs_pred, feed_dict={self.ss: ss})

    def train_on_batch(self, sess, ss, qs_true, lr):
        return sess.run(self.train_op, feed_dict={self.ss: ss, self.qs_true: qs_true, self.lr: lr})
