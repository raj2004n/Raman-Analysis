from keras.models import Model
from keras.layers import Input, Dense, Conv1D, Flatten, LeakyReLU, MaxPool1D, Dropout, BatchNormalization, Softmax, Activation, Reshape
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


def CNN_Model_Liu(num_classes, input_size, cfg=None, negative_slope=0.01):
    """Builds and compiles the model as Liu et al. draw it in Fig. 1 of their 2017 paper.

    Args:
        num_classes (int): Number of output classes, so the width of the final dense layer.
        input_size (int): Number of points on the standardised wavenumber axis.
        cfg (CnnConfig, optional): The architecture and optimiser settings. Defaults to
            CnnConfig(), which both models are built from.
        negative_slope (float): The gradient of the LeakyReLU's negative half. The paper writes
            it as alpha and gives it no value, so this is the 0.01 of Maas et al., the reference
            the paper cites for the function.

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

    x = inputs
    for filters, kernel_size in zip(cfg.conv_filters, cfg.conv_kernel_sizes):
        x = Conv1D(filters=filters, kernel_size=kernel_size, kernel_initializer=initializer)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(negative_slope=negative_slope)(x)
        x = MaxPool1D(cfg.pool_size, cfg.pool_size)(x)

    x = Flatten()(x)

    # each dense layer is normalised before its non-linearity, so the tanh reads a centred
    # input of unit scale rather than whatever spread the kernel left behind
    x = Dense(units=cfg.dense_units, kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = Activation("tanh")(x)

    # to enable Pooling(2,2) (a dense output has no axis for a pooling layer to read)
    # the vector is given a trailing axis of one and pooled as a 1-D map
    x = Reshape((cfg.dense_units, 1))(x)
    x = MaxPool1D(cfg.pool_size, cfg.pool_size)(x)
    x = Flatten()(x)

    x = Dropout(rate=cfg.dropout)(x)

    x = Dense(units=num_classes, kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = Activation("tanh")(x)
    outputs = Softmax(dtype='float32')(x)

    model = Model(inputs=inputs, outputs=outputs, name="Raman_CNN_Liu")

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model
