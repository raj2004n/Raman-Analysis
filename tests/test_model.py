"""The convolutional model is built from a config, and the config's defaults are the old architecture.

The settings used to be literals in the middle of the model builder, so a caller could not
change the architecture and nothing recorded what a trained model had actually been built
with. They come from ``CnnConfig`` now. The layer list and the parameter count below were
recorded from the model as it was before that change, which is what makes the restructure
reviewable: the same maths has to produce the same shapes and the same number of weights.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from src.config import CnnConfig
from src.cnn.model import CNN_Model

NUM_CLASSES = 3
INPUT_SIZE = 2000

# recorded from the model before the settings moved into CnnConfig
EXPECTED_LAYERS = [
    ("input_layer", "InputLayer", (None, 2000, 1)),
    ("conv1d", "Conv1D", (None, 1980, 16)),
    ("batch_normalization", "BatchNormalization", (None, 1980, 16)),
    ("leaky_re_lu", "LeakyReLU", (None, 1980, 16)),
    ("max_pooling1d", "MaxPooling1D", (None, 990, 16)),
    ("conv1d_1", "Conv1D", (None, 980, 32)),
    ("batch_normalization_1", "BatchNormalization", (None, 980, 32)),
    ("leaky_re_lu_1", "LeakyReLU", (None, 980, 32)),
    ("max_pooling1d_1", "MaxPooling1D", (None, 490, 32)),
    ("conv1d_2", "Conv1D", (None, 486, 64)),
    ("batch_normalization_2", "BatchNormalization", (None, 486, 64)),
    ("leaky_re_lu_2", "LeakyReLU", (None, 486, 64)),
    ("max_pooling1d_2", "MaxPooling1D", (None, 243, 64)),
    ("flatten", "Flatten", (None, 15552)),
    ("dense", "Dense", (None, 2048)),
    ("batch_normalization_3", "BatchNormalization", (None, 2048)),
    ("dropout", "Dropout", (None, 2048)),
    ("dense_1", "Dense", (None, 3)),
    ("batch_normalization_4", "BatchNormalization", (None, 3)),
    ("softmax", "Softmax", (None, 3)),
]

EXPECTED_PARAMS = 31883663


def _signature(model):
    return [(layer.name, type(layer).__name__, tuple(layer.output.shape)) for layer in model.layers]


def test_the_default_architecture_is_unchanged():
    """Every layer, in order, with its shape: the same model as before the settings moved."""
    model = CNN_Model(NUM_CLASSES, INPUT_SIZE)

    assert _signature(model) == EXPECTED_LAYERS
    assert model.count_params() == EXPECTED_PARAMS
    assert model.name == "Raman_CNN"


def test_the_default_optimiser_is_unchanged():
    """The Adam settings were literals too, including an epsilon that is easy to fat-finger."""
    model = CNN_Model(NUM_CLASSES, INPUT_SIZE)

    assert model.optimizer.learning_rate.numpy() == pytest.approx(0.001)
    assert model.optimizer.beta_1 == pytest.approx(0.9)
    assert model.optimizer.beta_2 == pytest.approx(0.999)
    assert model.optimizer.epsilon == pytest.approx(1e-08)
    assert model.loss == "categorical_crossentropy"


def test_the_config_drives_the_architecture():
    """A different filter list has to reach the built model, or the config is decoration."""
    wider = replace(CnnConfig(), conv_filters=(32, 64, 128))
    model = CNN_Model(NUM_CLASSES, INPUT_SIZE, cfg=wider)

    shapes = [shape for name, kind, shape in _signature(model) if kind == "Conv1D"]
    assert shapes == [(None, 1980, 32), (None, 980, 64), (None, 486, 128)]
    assert model.count_params() != EXPECTED_PARAMS


def test_the_config_drives_the_optimiser():
    """The Adam settings reach the compiled model rather than the defaults being used."""
    tuned = replace(CnnConfig(), learning_rate=0.01, beta_1=0.8, beta_2=0.99, epsilon=1e-06)
    model = CNN_Model(NUM_CLASSES, INPUT_SIZE, cfg=tuned)

    assert model.optimizer.learning_rate.numpy() == pytest.approx(0.01)
    assert model.optimizer.beta_1 == pytest.approx(0.8)
    assert model.optimizer.beta_2 == pytest.approx(0.99)
    assert model.optimizer.epsilon == pytest.approx(1e-06)


def test_the_class_count_sets_the_head_width():
    """``num_classes`` still decides the output width, which the caller derives from the data."""
    model = CNN_Model(7, INPUT_SIZE)

    assert tuple(model.layers[-1].output.shape) == (None, 7)
    assert tuple(model.layers[-3].output.shape) == (None, 7)
