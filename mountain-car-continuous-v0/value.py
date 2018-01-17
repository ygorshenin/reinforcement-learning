import tensorflow as tf


class Value:
    def __init__(self, *args):
        with tf.variable_scope('value'):
            self.__init_model(*args)

    def __init_model(self, env, hidden_units):
        self.s = tf.placeholder(tf.float32, shape=[None, env.states_dim], name='s')

        H1 = tf.layers.dense(inputs=self.s,
                             units=hidden_units,
                             activation=tf.tanh,
                             kernel_initializer=tf.contrib.layers.xavier_initializer())
        H2 = tf.layers.dense(inputs=H1,
                             units=hidden_units,
                             activation=tf.tanh,
                             kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.v_predicted = tf.layers.dense(
            inputs=H2,
            units=1,
            kernel_initializer=tf.contrib.layers.xavier_initializer())

        self.v_expected = tf.placeholder(tf.float32, shape=[None, 1], name='v_expected')
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        loss = tf.losses.mean_squared_error(self.v_predicted, self.v_expected)
        loss = tf.reduce_sum(loss)

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.minimize(loss)

    def get_value(self, s, sess):
        return sess.run(self.v_predicted, feed_dict={self.s: s})

    def train(self, s, v, learning_rate, sess):
        feed_dict = {self.s: s, self.v_expected: v, self.learning_rate: learning_rate}
        sess.run(self.train_op, feed_dict=feed_dict)
