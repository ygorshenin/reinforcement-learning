import numpy as np
import tensorflow as tf

from operator import mul

HIDDEN_UNITS = 100


class DQN:
    def __init__(self, input_shape, output_dim):
        self.ss = tf.placeholder(tf.float32, [None] + list(input_shape))
        self.qs_target = tf.placeholder(tf.float32, [None, output_dim])
        self.lr = tf.placeholder(tf.float32)

        W_conv_1 = tf.get_variable("W_conv_1",
                                   shape=[5, 5, 1, 10],
                                   initializer=tf.contrib.layers.xavier_initializer())
        b_conv_1 = tf.get_variable("b_conv_1",
                                   shape=[10],
                                   initializer=tf.zeros_initializer())

        conv_1 = tf.nn.conv2d(self.ss, W_conv_1, strides=[1, 1, 1, 1], padding="SAME", data_format="NHWC")
        h_conv_1 = tf.nn.tanh(conv_1)

        num_hidden_1 = 1
        for dim in h_conv_1.shape[1:]:
            num_hidden_1 *= int(dim)

        flatten = tf.reshape(h_conv_1, [-1, num_hidden_1])

        num_hidden_2 = 50

        W_2 = tf.get_variable("W_2",
                              shape=[num_hidden_1, num_hidden_2],
                              initializer=tf.contrib.layers.xavier_initializer())
        b_2 = tf.get_variable("b_2",
                              shape=[num_hidden_2],
                              initializer=tf.zeros_initializer())
        h_2 = tf.nn.tanh(tf.matmul(flatten, W_2) + b_2)

        W_3 = tf.get_variable("W_3",
                              shape=[num_hidden_2, output_dim],
                              initializer=tf.contrib.layers.xavier_initializer())
        b_3 = tf.get_variable("b_3",
                              shape=[output_dim],
                              initializer=tf.zeros_initializer())

        self.qs_predict = tf.matmul(h_2, W_3) + b_3

        loss = tf.square(self.qs_target - self.qs_predict)
        loss = tf.reduce_mean(loss)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_op = optimizer.minimize(loss)

    def predict(self, sess, s):
        return sess.run(self.qs_predict, feed_dict={self.ss: np.array([s])})[0]

    def predict_on_batch(self, sess, ss):
        return sess.run(self.qs_predict, feed_dict={self.ss: ss})

    def train_on_batch(self, sess, ss, qs_target, lr):
        return sess.run(self.train_op, feed_dict={self.ss: ss, self.qs_target: qs_target, self.lr: lr})
