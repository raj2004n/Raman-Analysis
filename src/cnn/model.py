from keras.models import Model
from keras.layers import Input, Dense, Conv1D, Flatten, LeakyReLU, MaxPool1D, Dropout, BatchNormalization, Softmax
from keras.initializers import RandomNormal
import numpy as np
from keras import optimizers

from src.config import CnnConfig

def CNN_Model(num_classes, input_size, cfg=None):
    """Builds and compiles the convolutional model.

    Args:
        num_classes (int): Number of output classes, so the width of the final dense layer.
        input_size (int): Number of points on the standardised wavenumber axis.
        cfg (CnnConfig, optional): The architecture and optimiser settings. Defaults to
            CnnConfig(), which is the architecture this model was built with before the
            settings became arguments.

    Returns:
        The compiled Keras model.
    """
    if cfg is None:
        cfg = CnnConfig()

    inputs = Input(shape=(input_size,1))
    initializer = RandomNormal(mean=0.0, stddev=np.sqrt(cfg.initializer_variance))

    opt = optimizers.Adam(
    learning_rate=cfg.learning_rate,
    beta_1=cfg.beta_1,
    beta_2=cfg.beta_2,
    epsilon=cfg.epsilon
    )

    # one convolutional block per pair of filter count and kernel size, in order
    x = inputs
    for filters, kernel_size in zip(cfg.conv_filters, cfg.conv_kernel_sizes):
        x = Conv1D(filters=filters, kernel_size=kernel_size, kernel_initializer=initializer)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = MaxPool1D(cfg.pool_size, cfg.pool_size)(x)

    x = Flatten()(x)

    x = Dense(units=cfg.dense_units, activation='tanh', kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = Dropout(rate = cfg.dropout)(x)

    x = Dense(units=num_classes, kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    outputs = Softmax(dtype='float32')(x)

    model = Model(inputs=inputs, outputs=outputs, name="Raman_CNN")

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model
