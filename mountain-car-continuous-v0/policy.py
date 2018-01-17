import tensorflow as tf


class Policy:
    def __init__(self, *args):
        with tf.variable_scope('policy'):
            self.__init_policy(*args)

    def __init_policy(self, env, hidden_units):
        self.s = tf.placeholder(tf.float32, shape=[None, env.states_dim], name='s')

        H1 = tf.layers.dense(inputs=self.s,
                             units=hidden_units,
                             activation=tf.tanh,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             name='H1')
        H2 = tf.layers.dense(inputs=H1,
                             units=hidden_units,
                             activation=tf.tanh,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             name='H2')

        mu = tf.layers.dense(inputs=H2,
                             units=1,
                             activation=None,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             name='mu')
        sigma = tf.layers.dense(inputs=H2,
                                units=1,
                                activation=None,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name='sigma')
        sigma = tf.exp(sigma) + 1e-5
        distribution = tf.distributions.Normal(mu, sigma)

        self.action_predicted = tf.clip_by_value(distribution.sample(),
                                                 env.min_action,
                                                 env.max_action)

        self.action_expected = tf.placeholder(tf.float32, shape=[None, 1], name='action_expected')
        self.advantage = tf.placeholder(tf.float32, shape=[None, 1], name='advantage')
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        loss = -tf.log(distribution.prob(self.action_expected) + 1e-5) * self.advantage
        loss = tf.reduce_mean(loss)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.minimize(loss)

    def get_action(self, s, sess):
        return sess.run(self.action_predicted, feed_dict={self.s: s})

    def train(self, s, a, advantage, learning_rate, sess):
        feed_dict = {self.s: s,
                     self.action_expected: a,
                     self.advantage: advantage,
                     self.learning_rate: learning_rate}
        sess.run(self.train_op, feed_dict=feed_dict)
