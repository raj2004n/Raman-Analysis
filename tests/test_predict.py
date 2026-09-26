"""Predicting runs the model a training run actually wrote.

The bug this covers: the predicting script carried its own input length and its own
artifact filenames, pointing at a model trained for 1024 points while the training
script wrote one trained for 2000. Both loaded, so the mismatch was silent. Here the
model is built for the fixture's own axis, which means the prediction can only succeed
if the length and the wavenumber range are read from the run's manifest.
"""

from __future__ import annotations

from dataclasses import replace

import keras
import numpy as np
import pytest
from sklearn.preprocessing import LabelEncoder

from src.config import TrainConfig
from src.cnn import runs
from src.cnn.predict import predict
from tests import synthetic

LABELS = ["Calcite", "Pyrite", "Quartz"]


def _model_for(axis, num_classes=len(LABELS)):
    """A model whose only acceptable input is a spectrum of the fixture's length."""
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(axis.size, 1)),
            keras.layers.Flatten(),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="categorical_crossentropy")
    return model


def _run_for(axis, dataset, tmp_path, name="fixture", **overrides):
    """A saved run trained on the fixture's axis, with any config field overridden."""
    config = replace(
        TrainConfig(
            dataset=str(dataset),
            input_size=axis.size,
            x_min=float(np.min(axis)),
            x_max=float(np.max(axis)),
        ),
        **overrides,
    )

    directory = runs.run_directory(name, root=tmp_path)
    runs.save_run(directory, config=config, model=_model_for(axis), encoder=LabelEncoder().fit(LABELS))
    return directory


@pytest.fixture
def axis():
    return synthetic.spectral_axis()


def test_predict_reads_its_length_from_the_manifest(tmp_path, grid_dir, axis):
    """The settings come from run.json, so a model of the wrong length cannot be used."""
    run_dir = _run_for(axis, grid_dir, tmp_path)

    labels, top5 = predict(run_dir, grid_dir, synthetic.ROWS, synthetic.COLS)

    assert labels.shape == (synthetic.ROWS, synthetic.COLS)
    assert top5.shape[:2] == (synthetic.ROWS, synthetic.COLS)
    assert top5.shape[-1] == 2
    assert set(np.ravel(labels)) <= set(LABELS)

    # Every pixel carries a ranked (mineral, probability) pair per class. The real
    # classifier has far more classes than this stand-in, so the list is longer there.
    assert top5.shape[2] == len(LABELS)
    for rank in range(top5.shape[2]):
        label, probability = top5[0, 0][rank]
        assert label in LABELS
        assert 0.0 <= float(probability) <= 1.0


def test_predict_refuses_a_run_trained_outside_the_measured_range(tmp_path, grid_dir, axis):
    """Padding a spectrum with zeros to reach the training range changes the prediction."""
    run_dir = _run_for(axis, grid_dir, tmp_path, x_max=float(np.max(axis)) + 200.0)

    with pytest.raises(ValueError, match="pad the spectrum with zeros"):
        predict(run_dir, grid_dir, synthetic.ROWS, synthetic.COLS)
