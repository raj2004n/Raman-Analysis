"""A training run and the manifest that describes it.

The defect these cover: the model training wrote and the model prediction loaded were
different files, trained at different input lengths (2000 and 1024), and nothing
recorded either length. Both models loaded and both ranked, so the mismatch produced
predictions rather than an error. The manifest is what makes the lengths comparable,
and these tests check that a disagreement is refused rather than honoured.
"""

from __future__ import annotations

import json

import keras
import numpy as np
import pytest
from sklearn.preprocessing import LabelEncoder

from src.cnn import runs

# In the order LabelEncoder puts them, which is sorted.
LABELS = ["Calcite", "Pyrite", "Quartz"]


def _tiny_model(input_size=64, num_classes=len(LABELS)):
    """A model small enough to save and load in a test, with the shape of the real one."""
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_size, 1)),
            keras.layers.Flatten(),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="categorical_crossentropy")
    return model


def _encoder():
    return LabelEncoder().fit(LABELS)


def _saved_run(directory, name="probe", input_size=64, x_min=313.3, x_max=1028.4):
    return runs.save_run(
        directory,
        name=name,
        model=_tiny_model(input_size=input_size),
        encoder=_encoder(),
        input_size=input_size,
        num_classes=len(LABELS),
        x_min=x_min,
        x_max=x_max,
    )


def test_a_run_round_trips_through_its_directory(tmp_path):
    manifest = _saved_run(runs.run_directory("probe", root=tmp_path))

    loaded = runs.load_run(runs.run_directory("probe", root=tmp_path))

    assert loaded.manifest == manifest
    assert loaded.manifest.input_size == 64
    assert loaded.manifest.num_classes == len(LABELS)
    assert loaded.manifest.x_min == pytest.approx(313.3)
    assert loaded.manifest.x_max == pytest.approx(1028.4)
    assert list(loaded.encoder.classes_) == LABELS
    assert (loaded.directory / runs.MANIFEST_NAME).is_file()


def test_the_manifest_is_readable_json(tmp_path):
    """It is meant to be legible without this code, so it stays a plain JSON object."""
    _saved_run(runs.run_directory("probe", root=tmp_path))

    text = (runs.run_directory("probe", root=tmp_path) / runs.MANIFEST_NAME).read_text()
    written = json.loads(text)

    assert written["input_size"] == 64
    assert written["num_classes"] == 3
    assert written["classes"] == LABELS
    assert set(written) >= {"input_size", "num_classes", "x_min", "x_max", "name", "created"}


def test_a_model_shorter_than_its_manifest_is_refused(tmp_path):
    """The original mismatch: a model trained at one length, loaded as if trained at another.

    This is the failure that used to be silent. The model is saved for 64 points and
    the manifest is then rewritten to claim 2000, which is what pointing a prediction
    at the wrong artifact amounts to. Loading has to refuse it.
    """
    directory = runs.run_directory("probe", root=tmp_path)
    _saved_run(directory)

    manifest_path = directory / runs.MANIFEST_NAME
    written = json.loads(manifest_path.read_text())
    written["input_size"] = 2000
    manifest_path.write_text(json.dumps(written))

    with pytest.raises(ValueError) as raised:
        runs.load_run(directory)

    message = str(raised.value)
    assert "64" in message and "2000" in message


def test_a_directory_that_is_not_a_run_says_which_runs_exist(tmp_path):
    _saved_run(runs.run_directory("probe", root=tmp_path))
    empty = tmp_path / "not-a-run"
    empty.mkdir()

    with pytest.raises(FileNotFoundError) as raised:
        runs.load_run(empty)

    assert "run.json" in str(raised.value)


def test_input_length_must_match_the_model_shape():
    manifest = runs.RunManifest(
        name="probe", created="2026-01-01T00:00:00", input_size=2000,
        num_classes=3, x_min=313.3, x_max=1028.4,
    )

    runs.check_model_input_length(manifest, (None, 2000, 1))  # convolutional
    runs.check_model_input_length(manifest, (None, 2000))  # dense

    with pytest.raises(ValueError, match="2000"):
        runs.check_model_input_length(manifest, (None, 1024, 1))

    with pytest.raises(ValueError, match="input shape"):
        runs.check_model_input_length(manifest, None)


def test_spectra_must_cover_the_training_range():
    """Outside the measured range the resampling pads with zeros, so it is refused."""
    manifest = runs.RunManifest(
        name="probe", created="2026-01-01T00:00:00", input_size=2000,
        num_classes=3, x_min=313.3, x_max=1028.4,
    )

    runs.check_axis_covers_training_range(manifest, np.arange(200.0, 1200.0))
    runs.check_axis_covers_training_range(manifest, np.array([313.3, 700.0, 1028.4]))

    # A sample measured over a narrower window than the model was trained on.
    with pytest.raises(ValueError, match="pad the spectrum with zeros"):
        runs.check_axis_covers_training_range(manifest, np.arange(500.0, 900.0))

    # And one that stops short at the top end.
    with pytest.raises(ValueError, match="pad the spectrum with zeros"):
        runs.check_axis_covers_training_range(manifest, np.arange(200.0, 800.0))


def test_runs_are_listed_newest_first(tmp_path):
    _saved_run(runs.run_directory("first", root=tmp_path), name="first")
    _saved_run(runs.run_directory("second", root=tmp_path), name="second")

    # created is written to the second precision, so make the order explicit rather
    # than depending on how fast the two writes happened.
    later = runs.run_directory("second", root=tmp_path) / runs.MANIFEST_NAME
    written = json.loads(later.read_text())
    written["created"] = "2099-01-01T00:00:00"
    later.write_text(json.dumps(written))

    assert runs.list_runs(root=tmp_path) == ["second", "first"]
    assert runs.latest_run(root=tmp_path) == runs.run_directory("second", root=tmp_path)


def test_no_runs_at_all_is_not_an_error(tmp_path):
    assert runs.list_runs(root=tmp_path) == []
    assert runs.latest_run(root=tmp_path) is None


def test_a_hand_written_run_name_is_used_as_given(tmp_path):
    """Without --name the directory is a timestamp, which is still a valid run name."""
    name = runs.new_run_name()
    assert name.startswith("run_")

    directory = runs.run_directory(name, root=tmp_path)
    _saved_run(directory, name=name)

    assert (directory / runs.MANIFEST_NAME).is_file()
    assert runs.list_runs(root=tmp_path) == [name]
