from keras.models import Model
from keras.layers import Input, Dense, Conv1D, Flatten, LeakyReLU, MaxPool1D, Dropout, BatchNormalization, Softmax
from keras.initializers import RandomNormal
import numpy as np
from keras import optimizers

def CNN_Model(num_classes, input_size):
    inputs = Input(shape=(input_size,1))
    initializer = RandomNormal(mean=0.0, stddev=np.sqrt(0.05))

    opt = optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-08
    )

    x = Conv1D(filters=16, kernel_size=21, kernel_initializer=initializer)(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPool1D(2, 2)(x)

    x = Conv1D(filters=32, kernel_size=11, kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPool1D(2, 2)(x)

    x = Conv1D(filters=64, kernel_size=5, kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPool1D(2, 2)(x)

    x = Flatten()(x)

    x = Dense(units=2048, activation='tanh', kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = Dropout(rate = 0.5)(x)

    x = Dense(units=num_classes, kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    outputs = Softmax(dtype='float32')(x)

    model = Model(inputs=inputs, outputs=outputs, name="Raman_CNN")

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model
