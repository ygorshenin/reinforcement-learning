import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer


class ValueDQN:
    def __init__(self, **kwargs):
        with tf.variable_scope('value'):
            self.__init_model__(**kwargs)

    def __init_model__(self, input_shape, hidden_units=128):
        self.states = tf.placeholder(tf.float32, shape=[None] + list(input_shape), name='states')
        self.values_true = tf.placeholder(tf.float32, shape=[None, 1], name='values_true')
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        h1 = tf.layers.dense(inputs=tf.layers.flatten(self.states),
                             units=512,
                             activation=tf.nn.tanh,
                             kernel_initializer=xavier_initializer())
        h2 = tf.layers.dense(inputs=h1,
                             units=256,
                             activation=tf.nn.tanh,
                             kernel_initializer=xavier_initializer())
        self.values_pred = tf.layers.dense(inputs=h2,
                                           units=1,
                                           kernel_initializer=xavier_initializer())

        self.loss = tf.losses.mean_squared_error(self.values_true, self.values_pred)
        self.loss = tf.reduce_mean(self.loss)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = optimizer.minimize(self.loss)

    def predict(self, sess, state):
        return self.predict_on_batch(sess, [state])[0][0]

    def predict_on_batch(self, sess, states):
        return sess.run(self.values_pred, feed_dict={self.states: states})

    def train_on_batch(self, sess, states, values_true, learning_rate):
        feed_dict = {self.states: states,
                     self.values_true: values_true,
                     self.learning_rate: learning_rate}
        return sess.run([self.train_op, self.loss], feed_dict=feed_dict)
