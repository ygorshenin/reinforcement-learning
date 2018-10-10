from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten

HIDDEN_UNITS = 100


class DQN:
    def __init__(self, input_shape, output_shape):
        model = Sequential()
        model.add(Conv2D(filters=10,
                         kernel_size=(3, 3),
                         input_shape=input_shape,
                         activation='relu'))
        model.add(Flatten())
        model.add(Dense(HIDDEN_UNITS, activation='relu'))
        model.add(Dense(output_shape))
        model.compile(optimizer='adam', loss='mean_squared_error')

        self.model = model

    def predict(self, batch):
        return self.model.predict_on_batch(batch)

    def train(self, batch, q_target):
        return self.model.train_on_batch(batch, q_target)
