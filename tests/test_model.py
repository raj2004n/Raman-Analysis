"""The convolutional model is built from a config, and the config's defaults are the old architecture.

The settings used to be literals in the middle of the model builder, so a caller could not
change the architecture and nothing recorded what a trained model had actually been built
with. They come from ``CnnConfig`` now. The layer list and the parameter count below were
recorded from the model as it was before that change, which is what makes the restructure
reviewable: the same maths has to produce the same shapes and the same number of weights.

``src/cnn/model.py`` holds a second builder, ``CNN_Model_Liu``, which follows the paper where
``CNN_Model`` had departed from it. Its signature is recorded here the same way, so the two can
be read side by side: the convolutional half is the same down to the dense head, and the head is
where they part.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from src.config import CnnConfig
from src.cnn.model import CNN_Model, CNN_Model_Liu

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

# recorded from CNN_Model_Liu as it was written: the same convolutional half, and a dense head
# that pools, normalises before its tanh and carries a tanh on the output layer as well. Only the
# layer type and the shape are pinned, not the name: Keras numbers a name from a count that runs
# across every model the process has built, so the names a model gets depend on how many were
# built before it. ``EXPECTED_LAYERS`` above can pin them because it is the first model the file
# builds.
EXPECTED_LAYERS_LIU = [
    ("InputLayer", (None, 2000, 1)),
    ("Conv1D", (None, 1980, 16)),
    ("BatchNormalization", (None, 1980, 16)),
    ("LeakyReLU", (None, 1980, 16)),
    ("MaxPooling1D", (None, 990, 16)),
    ("Conv1D", (None, 980, 32)),
    ("BatchNormalization", (None, 980, 32)),
    ("LeakyReLU", (None, 980, 32)),
    ("MaxPooling1D", (None, 490, 32)),
    ("Conv1D", (None, 486, 64)),
    ("BatchNormalization", (None, 486, 64)),
    ("LeakyReLU", (None, 486, 64)),
    ("MaxPooling1D", (None, 243, 64)),
    ("Flatten", (None, 15552)),
    ("Dense", (None, 2048)),
    ("BatchNormalization", (None, 2048)),
    ("Activation", (None, 2048)),
    ("Reshape", (None, 2048, 1)),
    ("MaxPooling1D", (None, 1024, 1)),
    ("Flatten", (None, 1024)),
    ("Dropout", (None, 1024)),
    ("Dense", (None, 3)),
    ("BatchNormalization", (None, 3)),
    ("Activation", (None, 3)),
    ("Softmax", (None, 3)),
]

# 3072 fewer than EXPECTED_PARAMS: the pooled head halves the output layer's input
EXPECTED_PARAMS_LIU = 31880591


def _signature(model):
    return [(layer.name, type(layer).__name__, tuple(layer.output.shape)) for layer in model.layers]


def _conv_shapes(model):
    """The output shape of each convolutional layer, in order."""
    return [tuple(layer.output.shape) for layer in model.layers if type(layer).__name__ == "Conv1D"]


def _conv_blocks(model):
    """Each convolutional layer as its kernel size, filter count and output shape."""
    return [
        (layer.kernel.shape[0], layer.filters, tuple(layer.output.shape))
        for layer in model.layers
        if type(layer).__name__ == "Conv1D"
    ]


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


def test_the_paper_architecture_is_the_recorded_one():
    """Every layer of ``CNN_Model_Liu``, in order, with its shape and the parameter count."""
    model = CNN_Model_Liu(NUM_CLASSES, INPUT_SIZE)

    assert [(kind, shape) for _, kind, shape in _signature(model)] == EXPECTED_LAYERS_LIU
    assert model.count_params() == EXPECTED_PARAMS_LIU
    assert model.name == "Raman_CNN_Liu"


def test_the_two_models_part_only_in_the_dense_head():
    """The convolutional half is shared, which is what makes the head the thing to review."""
    paper = CNN_Model_Liu(NUM_CLASSES, INPUT_SIZE)
    earlier = CNN_Model(NUM_CLASSES, INPUT_SIZE)

    assert _conv_blocks(paper) == _conv_blocks(earlier)
    assert _conv_blocks(paper) == [
        (21, 16, (None, 1980, 16)),
        (11, 32, (None, 980, 32)),
        (5, 64, (None, 486, 64)),
    ]

    # and from the flatten onwards the two differ in what the head is made of
    assert [type(layer).__name__ for layer in paper.layers[13:]] != [
        type(layer).__name__ for layer in earlier.layers[13:]
    ]


def test_the_paper_architecture_normalises_before_each_tanh():
    """Fig. 1 draws the batch norm and then the tanh; ``CNN_Model`` has the tanh first."""
    model = CNN_Model_Liu(NUM_CLASSES, INPUT_SIZE)

    kinds = [type(layer).__name__ for layer in model.layers]
    tanhs = [index for index, kind in enumerate(kinds) if kind == "Activation"]

    assert len(tanhs) == 2, "both dense layers carry a tanh, where the earlier model has one"
    for index in tanhs:
        assert kinds[index - 1] == "BatchNormalization", f"{model.layers[index].name} follows a normalisation"


def test_the_paper_architecture_pools_the_dense_head():
    """Fig. 1's 2x2 pool halves the 2048-wide vector, and the output layer reads the halved one."""
    model = CNN_Model_Liu(NUM_CLASSES, INPUT_SIZE)

    pooled = [shape for _, kind, shape in _signature(model) if kind == "MaxPooling1D"]
    assert pooled == [(None, 990, 16), (None, 490, 32), (None, 243, 64), (None, 1024, 1)]

    output_layer = model.layers[-4]
    assert tuple(output_layer.kernel.shape) == (1024, NUM_CLASSES)


def test_the_paper_architecture_uses_the_papers_slope():
    """Eq. 1 writes the slope as alpha and gives it no value; 0.01 is what the paper cites for it."""
    model = CNN_Model_Liu(NUM_CLASSES, INPUT_SIZE)

    slopes = [layer.negative_slope for layer in model.layers if type(layer).__name__ == "LeakyReLU"]
    assert slopes == [0.01, 0.01, 0.01]

    steeper = CNN_Model_Liu(NUM_CLASSES, INPUT_SIZE, negative_slope=0.1)
    assert {layer.negative_slope for layer in steeper.layers if type(layer).__name__ == "LeakyReLU"} == {0.1}


def test_the_paper_architecture_takes_the_configs_architecture_and_optimiser():
    """The two builders share ``CnnConfig``, so a setting reaches both of them."""
    wider = replace(CnnConfig(), conv_filters=(32, 64, 128), learning_rate=0.01)
    model = CNN_Model_Liu(NUM_CLASSES, INPUT_SIZE, cfg=wider)

    shapes = _conv_shapes(model)
    assert shapes == [(None, 1980, 32), (None, 980, 64), (None, 486, 128)]
    assert model.optimizer.learning_rate.numpy() == pytest.approx(0.01)
