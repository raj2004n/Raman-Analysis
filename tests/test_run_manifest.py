"""A training run and the manifest that describes it.

The defect these cover: the model training wrote and the model prediction loaded were
different files, trained at different input lengths (2000 and 1024), and nothing
recorded either length. Both models loaded and both ranked, so the mismatch produced
predictions rather than an error. The manifest is what makes the lengths comparable,
and these tests check that a disagreement is refused rather than honoured.

The later half is the record a reported number is defended from: the seed, the dataset
digests, the versions and the score. Those are checked for what they are meant to be
sensitive to (a changed spectrum, a moved directory) rather than for their values, which
depend on where the fixture was written.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import pickle
from dataclasses import replace
from pathlib import Path

import keras
import numpy as np
import pytest
from sklearn.preprocessing import LabelEncoder

from src.config import TrainConfig
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


def _dataset(directory, names=("a.txt", "b.txt")):
    """A directory of files for the digest to be taken over, made if it is not there."""
    directory = directory if isinstance(directory, Path) else Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_text(f"{name}\n")
    return directory


def _config(dataset, **overrides):
    """The config a run is saved with. Its dataset has to exist, since it is digested."""
    return replace(
        TrainConfig(
            dataset=str(dataset), input_size=64, x_min=313.3, x_max=1028.4, name="probe"
        ),
        **overrides,
    )


def _saved_run(directory, dataset, **overrides):
    """Saves a tiny model into ``directory`` and returns the manifest written."""
    return runs.save_run(
        directory,
        config=_config(dataset, **overrides),
        model=_tiny_model(input_size=overrides.get("input_size", 64)),
        encoder=_encoder(),
    )


def test_a_run_round_trips_through_its_directory(tmp_path, dataset):
    manifest = _saved_run(runs.run_directory("probe", root=tmp_path), dataset)

    loaded = runs.load_run(runs.run_directory("probe", root=tmp_path))

    assert loaded.manifest == manifest
    assert loaded.manifest.input_size == 64
    assert loaded.manifest.num_classes == len(LABELS)
    assert loaded.manifest.x_min == pytest.approx(313.3)
    assert loaded.manifest.x_max == pytest.approx(1028.4)
    assert list(loaded.encoder.classes_) == LABELS
    assert (loaded.directory / runs.MANIFEST_NAME).is_file()


def test_the_manifest_is_readable_json(tmp_path, dataset):
    """It is meant to be legible without this code, so it stays a plain JSON object."""
    _saved_run(runs.run_directory("probe", root=tmp_path), dataset)

    text = (runs.run_directory("probe", root=tmp_path) / runs.MANIFEST_NAME).read_text()
    written = json.loads(text)

    assert written["input_size"] == 64
    assert written["num_classes"] == 3
    assert written["classes"] == LABELS
    assert set(written) >= {
        "input_size", "num_classes", "x_min", "x_max", "name", "created",
        "seed", "dataset", "dataset_hash", "dataset_order", "package_versions",
        "metrics", "config",
    }


def test_a_model_shorter_than_its_manifest_is_refused(tmp_path, dataset):
    """The original mismatch: a model trained at one length, loaded as if trained at another.

    This is the failure that used to be silent. The model is saved for 64 points and
    the manifest is then rewritten to claim 2000, which is what pointing a prediction
    at the wrong artifact amounts to. Loading has to refuse it.
    """
    directory = runs.run_directory("probe", root=tmp_path)
    _saved_run(directory, dataset)

    manifest_path = directory / runs.MANIFEST_NAME
    written = json.loads(manifest_path.read_text())
    written["input_size"] = 2000
    manifest_path.write_text(json.dumps(written))

    with pytest.raises(ValueError) as raised:
        runs.load_run(directory)

    message = str(raised.value)
    assert "64" in message and "2000" in message


def test_a_directory_that_is_not_a_run_says_which_runs_exist(tmp_path, dataset):
    _saved_run(runs.run_directory("probe", root=tmp_path), dataset)
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


def test_runs_are_listed_newest_first(tmp_path, dataset):
    _saved_run(runs.run_directory("first", root=tmp_path), dataset)
    _saved_run(runs.run_directory("second", root=tmp_path), dataset)

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


def test_a_hand_written_run_name_is_used_as_given(tmp_path, dataset):
    """Without --name the directory is a timestamp, which is still a valid run name."""
    name = runs.new_run_name()
    assert name.startswith("run_")

    directory = runs.run_directory(name, root=tmp_path)
    manifest = _saved_run(directory, dataset)

    assert manifest.name == name, "the name a run is listed under is the name it reports"
    assert (directory / runs.MANIFEST_NAME).is_file()
    assert runs.list_runs(root=tmp_path) == [name]


# --------------------------------------------------------------------------------------
# The record a reported number is defended from. What follows checks what each fact is
# sensitive to, since a digest of a fixture written to a temporary directory has no
# value worth pinning.
# --------------------------------------------------------------------------------------


@pytest.fixture
def dataset(tmp_path):
    """A couple of files standing in for a dataset directory."""
    return _dataset(tmp_path / "data")


def test_the_manifest_records_the_seed_and_the_dataset(tmp_path, dataset):
    """A run with no seed and no dataset recorded cannot be replayed from its own directory."""
    manifest = _saved_run(runs.run_directory("probe", root=tmp_path), dataset, seed=11)

    assert manifest.seed == 11
    assert manifest.dataset == str(dataset)
    assert manifest.dataset_hash.startswith("sha256:")
    assert manifest.dataset_order.startswith("sha256:")


def test_the_manifest_records_the_config_it_trained_with(tmp_path, dataset):
    """The epoch cap and the augmentation counts travel with the run, not with the terminal."""
    manifest = _saved_run(
        runs.run_directory("probe", root=tmp_path), dataset, epochs=7, patience=2, n_shift=3, n_noise=4
    )

    assert manifest.config["epochs"] == 7
    assert manifest.config["patience"] == 2
    assert (manifest.config["n_shift"], manifest.config["n_noise"]) == (3, 4)
    assert manifest.config["seed"] == manifest.seed


def test_the_metrics_are_recorded_as_given(tmp_path, dataset):
    """The score the report quotes is measured by the caller and written unchanged."""
    manifest = runs.save_run(
        runs.run_directory("probe", root=tmp_path),
        config=_config(dataset),
        model=_tiny_model(),
        encoder=_encoder(),
        metrics={"test_loss": 0.5, "test_accuracy": 0.75},
    )

    written = json.loads((runs.run_directory("probe", root=tmp_path) / runs.MANIFEST_NAME).read_text())
    assert manifest.metrics == {"test_loss": 0.5, "test_accuracy": 0.75}
    assert written["metrics"]["test_accuracy"] == 0.75


def test_a_run_scores_nothing_until_it_is_told(tmp_path, dataset):
    """A manifest written without metrics says so by holding an empty object."""
    manifest = _saved_run(runs.run_directory("probe", root=tmp_path), dataset)

    assert manifest.metrics == {}


def test_the_dataset_digest_ignores_where_the_data_lives(tmp_path):
    """A copy of the same spectra is the same data, wherever it was copied to."""
    first = _dataset(tmp_path / "one" / "data")
    second = _dataset(tmp_path / "two" / "data")

    assert runs.hash_dataset(first) == runs.hash_dataset(second)


def test_the_dataset_digest_ignores_a_new_modification_time(tmp_path, dataset):
    """Re-copying a dataset rewrites mtimes, which says nothing about the spectra."""
    before = runs.hash_dataset(dataset)

    os.utime(dataset / "a.txt", (0, 0))

    assert runs.hash_dataset(dataset) == before


def test_the_dataset_digest_changes_with_the_data(tmp_path, dataset):
    """An edited spectrum, an added file and a removed file are all different data."""
    before = runs.hash_dataset(dataset)

    (dataset / "a.txt").write_text("edited\n")
    edited = runs.hash_dataset(dataset)
    (dataset / "c.txt").write_text("new\n")
    added = runs.hash_dataset(dataset)
    (dataset / "b.txt").unlink()
    removed = runs.hash_dataset(dataset)

    assert len({before, edited, added, removed}) == 4, "each change has to move the digest"


def test_the_dataset_digest_refuses_a_directory_that_is_not_there(tmp_path):
    """Hashing nothing would record the digest of an empty dataset as if it were the data."""
    with pytest.raises(FileNotFoundError, match="no dataset to digest"):
        runs.hash_dataset(tmp_path / "absent")


def test_a_run_cannot_be_saved_for_a_dataset_that_is_not_there(tmp_path):
    """The record is refused rather than written with the digest left out."""
    with pytest.raises(FileNotFoundError):
        _saved_run(runs.run_directory("probe", root=tmp_path), tmp_path / "absent")


def test_the_read_order_digest_follows_the_files_in_the_directory(tmp_path, dataset):
    """It covers the ``.txt`` files, so adding, removing or renaming one moves it.

    This is the digest that records which order the loader saw, so it has to change when
    the set of files the loader would read changes, and it must not depend on the order
    the digests happen to be taken in.
    """
    before = runs.dataset_order_digest(dataset)

    (dataset / "c.txt").write_text("new\n")
    added = runs.dataset_order_digest(dataset)
    (dataset / "notes.md").write_text("not a spectrum\n")
    other = runs.dataset_order_digest(dataset)
    (dataset / "a.txt").unlink()
    removed = runs.dataset_order_digest(dataset)

    assert before != added
    assert other == added, "only the .txt files are read, so another file is not data"
    assert removed not in (before, added)


def test_the_read_order_digest_is_the_order_the_loader_uses(tmp_path, monkeypatch):
    """The digest lists the names ``glob`` returns, which is the call the loader makes."""
    directory = _dataset(tmp_path / "data", names=("a.txt", "b.txt", "c.txt"))
    seen = []
    real_glob = runs.glob.glob

    def recording_glob(pattern):
        names = real_glob(pattern)
        seen.extend(Path(name).name for name in names)
        return names

    monkeypatch.setattr(runs.glob, "glob", recording_glob)
    digest = runs.dataset_order_digest(directory)

    expected = "sha256:" + hashlib.sha256("\n".join(seen).encode("utf-8")).hexdigest()
    assert seen, "the digest has to read the directory rather than assume an order"
    assert digest == expected


def test_the_versions_are_the_ones_installed(tmp_path, dataset):
    """The record names the versions the run used, not the ones the code was written against."""
    versions = runs.package_versions()

    assert set(versions) == set(runs.RECORDED_PACKAGES)
    assert versions["keras"] == importlib.metadata.version("keras")
    assert versions["numpy"] == importlib.metadata.version("numpy")


def test_a_package_that_is_not_installed_is_recorded_as_such():
    """Leaving it out would read as a package nobody asked about."""
    assert runs.package_versions(("a-package-that-is-not-installed",)) == {
        "a-package-that-is-not-installed": "not installed"
    }


def test_a_manifest_written_before_the_record_existed_still_loads(tmp_path):
    """Runs written before this phase are still runs: the missing facts read as absent."""
    older = {
        "name": "probe", "created": "2026-01-01T00:00:00", "input_size": 64,
        "num_classes": 3, "x_min": 313.3, "x_max": 1028.4, "classes": LABELS,
    }
    directory = runs.run_directory("probe", root=tmp_path)
    directory.mkdir(parents=True)
    (directory / runs.MANIFEST_NAME).write_text(json.dumps(older))
    _tiny_model().save(directory / runs.MODEL_NAME)
    with open(directory / runs.ENCODER_NAME, "wb") as handle:
        pickle.dump(_encoder(), handle)

    loaded = runs.load_run(directory)

    assert loaded.manifest.name == "probe"
    assert loaded.manifest.seed is None
    assert loaded.manifest.metrics == {}
    assert loaded.manifest.config == {}
