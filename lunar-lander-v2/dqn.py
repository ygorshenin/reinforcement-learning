import tensorflow as tf

from tensorflow.contrib.layers import xavier_initializer


class DQN:
    """Deep Q-network, same number of hidden units on all hidden
       layers, with tanh non-linearities between layers.
    """
    def __init__(self, name, states_dim, actions_dim, hidden_layers, hidden_units):
        with tf.variable_scope(name):
            self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')

            self.state = tf.placeholder(tf.float32, shape=[None, states_dim], name='state')
            self.weights = tf.placeholder(tf.float32, shape=[None, 1], name='state')

            self.q_current = self.state

            self.Ws = []
            self.bs = []
            hidden_layers += 1
            for i in range(hidden_layers):
                input_dim = states_dim if i == 0 else hidden_units
                output_dim = actions_dim if i + 1 == hidden_layers else hidden_units

                W = tf.get_variable(name='W{}'.format(i),
                                    shape=[input_dim, output_dim],
                                    initializer=xavier_initializer())
                b = tf.get_variable(name='b{}'.format(i),
                                    shape=[output_dim],
                                    initializer=xavier_initializer())
                self.Ws.append(W)
                self.bs.append(b)

                self.q_current = tf.matmul(self.q_current, W) + b
                if i + 1 != hidden_layers:
                    self.q_current = tf.tanh(self.q_current)

            self.q_target = tf.placeholder(tf.float32, shape=[None, actions_dim], name='q_target')

            optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
            self.loss = tf.square(self.q_current - self.q_target)
            self.loss = tf.multiply(self.loss, self.weights)
            self.loss = tf.reduce_sum(self.loss)
            # self.loss = tf.losses.mean_squared_error(self.q_current, self.q_target)
            self.train_op = optimizer.minimize(self.loss)

    def train(self, ss, qs, ws, lr, sess):
        feed_dict = {self.state: ss, self.q_target: qs, self.weights: ws, self.learning_rate: lr}
        return sess.run(self.train_op, feed_dict=feed_dict)

    def predict(self, ss, sess):
        feed_dict = {self.state: ss}
        return sess.run(self.q_current, feed_dict=feed_dict)

    def copy_to(self, rhs, sess):
        ops = []
        for (Wl, Wr) in zip(self.Ws, rhs.Ws):
            ops.append(tf.assign(Wr, Wl))
        for (bl, br) in zip(self.bs, rhs.bs):
            ops.append(tf.assign(br, bl))
        sess.run(ops)
