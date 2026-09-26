"""Importing the training module must not train, and a run must follow its config.

The module used to be a script: four wildcard imports, a hardcoded dataset path and a
hardcoded wavenumber range, all at module level, so importing it started a training run
and nothing about it could be reused or tested. These tests pin both halves of the fix,
using a dataset path and a wavenumber range that are deliberately not the defaults.
"""

from __future__ import annotations

import argparse
import importlib
from dataclasses import replace

import numpy as np
import pytest
import ramanspy as rp

from src.config import TrainConfig
from src.cnn import train as train_module
from src.data import loader

# not the defaults, so a value still being read from a literal is visible
OTHER_DATASET = "~/Code/Data_SH/another_cell"
OTHER_RANGE = (100.0, 900.0)


def test_importing_the_module_does_not_train_anything(monkeypatch):
    """Re-executes the module body with the dataset loader and argparse booby-trapped.

    A reload runs everything at module level, so if any of it reached for the data or
    built a parser, this raises instead of starting a job.
    """
    def explosion(*args, **kwargs):
        raise AssertionError("importing the training module started work at module level")

    monkeypatch.setattr(loader, "load_CNN_data", explosion)
    monkeypatch.setattr(argparse, "ArgumentParser", explosion)

    reloaded = importlib.reload(train_module)

    assert callable(reloaded.train), "the module must expose the training flow as a function"
    assert callable(reloaded.main)


def test_the_command_line_defaults_are_the_config_defaults():
    """The help text and the code cannot disagree about the defaults, since both read them."""
    args = train_module.parse_args([])

    assert args.dataset == TrainConfig.dataset
    assert args.input_size == TrainConfig.input_size
    assert args.x_min == TrainConfig.x_min
    assert args.x_max == TrainConfig.x_max
    assert args.epochs == TrainConfig.epochs
    assert args.patience == TrainConfig.patience
    assert args.n_shift == TrainConfig.n_shift
    assert args.n_noise == TrainConfig.n_noise
    assert args.seed == TrainConfig.seed
    assert args.name is None


def test_the_command_line_can_over_rule_every_default():
    """A second dataset and a different axis are arguments, which is the point of the config."""
    args = train_module.parse_args(
        ["--dataset", OTHER_DATASET, "--input-size", "1200", "--x-min", "100", "--x-max", "900",
         "--epochs", "5", "--patience", "2", "--n-shift", "0", "--n-noise", "0", "--seed", "4",
         "--name", "trial"]
    )

    assert args.dataset == OTHER_DATASET
    assert (args.input_size, args.x_min, args.x_max) == (1200, 100.0, 900.0)
    assert (args.epochs, args.patience) == (5, 2)
    assert (args.n_shift, args.n_noise) == (0, 0)
    assert args.seed == 4
    assert args.name == "trial"


class _FakeModel:
    """Stands in for the compiled model: records what it was fitted with."""

    def __init__(self):
        self.fitted = {}
        self.scored = []

    def summary(self):
        return None

    def fit(self, x, y, **kwargs):
        self.fitted = {"x": x, "y": y, **kwargs}
        return "history"

    def evaluate(self, x, y, **kwargs):
        self.scored.append({"x": x, "y": y, **kwargs})
        return [0.25, 0.75]


class _FakeRunDir:
    """Stands in for the run directory: records that it was created, and joins paths."""

    def __init__(self):
        self.created = []

    def mkdir(self, **kwargs):
        self.created.append(kwargs)

    def __truediv__(self, other):
        return other


@pytest.fixture
def training(monkeypatch):
    """Replaces every heavy step of a run and records what each one was given."""
    recorded = {"model": _FakeModel(), "run_dir": _FakeRunDir()}

    def fake_load(path):
        recorded["load_path"] = path
        spectra = [rp.Spectrum(np.linspace(0, 1, 50) + i, np.linspace(100, 900, 50)) for i in range(4)]
        metadata = [{"##NAMES": name} for name in ("quartz", "calcite", "quartz", "calcite")]
        return spectra, metadata

    monkeypatch.setattr(train_module, "load_CNN_data", fake_load)

    def fake_model(num_classes, input_size, cfg=None):
        recorded["model_args"] = {"num_classes": num_classes, "input_size": input_size}
        return recorded["model"]

    monkeypatch.setattr(train_module, "CNN_Model", fake_model)

    def fake_standardise(spectra, target_length, x_min=0, x_max=1400):
        recorded.setdefault("standardise", []).append(
            {"target_length": target_length, "x_min": x_min, "x_max": x_max, "n": len(spectra)}
        )
        return np.zeros((len(spectra), target_length, 1))

    monkeypatch.setattr(train_module, "standardise_data", fake_standardise)

    real_split = train_module.leave_one_out_split

    def fake_split(y_data, rng):
        recorded["rng"] = rng
        recorded["split"] = real_split(y_data, rng)
        return recorded["split"]

    monkeypatch.setattr(train_module, "leave_one_out_split", fake_split)

    def fake_augment(x, y, n_shift, n_noise, rng):
        recorded["augmentation_rng"] = rng
        recorded.setdefault("augment", []).append({"n_shift": n_shift, "n_noise": n_noise})
        return x, y

    monkeypatch.setattr(train_module, "build_augmented_dataset", fake_augment)
    monkeypatch.setattr(train_module, "set_random_seed", lambda seed: recorded.setdefault("seeds", []).append(seed))
    monkeypatch.setattr(train_module, "plot_model", lambda *a, **k: None)
    monkeypatch.setattr(train_module, "show_results", lambda *a, **k: None)

    def fake_save_run(directory, **kwargs):
        recorded["save_run"] = {"directory": directory, **kwargs}
        return type("Manifest", (), {"name": kwargs["config"].name or "run"})()

    monkeypatch.setattr(train_module.runs, "save_run", fake_save_run)
    monkeypatch.setattr(
        train_module.runs, "run_directory", lambda name, root=None: recorded["run_dir"]
    )
    return recorded


def test_a_run_takes_its_dataset_and_axis_from_the_config(training):
    """The path, the point count and the wavenumber range all come from the config."""
    cfg = replace(TrainConfig(), dataset=OTHER_DATASET, input_size=1200,
                  x_min=OTHER_RANGE[0], x_max=OTHER_RANGE[1], epochs=5, patience=3,
                  n_shift=2, n_noise=4, name="trial")

    train_module.train(cfg)

    assert training["load_path"] == OTHER_DATASET
    assert [call["target_length"] for call in training["standardise"]] == [1200, 1200]
    assert {(call["x_min"], call["x_max"]) for call in training["standardise"]} == {OTHER_RANGE}
    assert training["augment"] == [{"n_shift": 2, "n_noise": 4}]
    assert training["model_args"]["input_size"] == 1200, "the model is built for the configured axis"
    assert training["model"].fitted["epochs"] == 5
    assert training["model"].fitted["callbacks"][0].patience == 3


def test_a_run_records_the_axis_and_length_it_trained_on(training):
    """The manifest has to carry the length and the range, since a prediction must reproduce both.

    Everything the record is built from is read out of the config the run was given, so
    these assertions are about the same object rather than about a list of arguments that
    could drift from it.
    """
    cfg = replace(TrainConfig(), dataset=OTHER_DATASET, input_size=1200,
                  x_min=OTHER_RANGE[0], x_max=OTHER_RANGE[1], name="trial")

    manifest = train_module.train(cfg)

    saved = training["save_run"]
    assert saved["config"] is cfg
    assert saved["config"].name == "trial"
    assert saved["config"].input_size == 1200
    assert (saved["config"].x_min, saved["config"].x_max) == OTHER_RANGE
    assert saved["metrics"] == {"test_loss": 0.25, "test_accuracy": 0.75}, (
        "the score has to be measured on the held-out spectra and travel with the run"
    )
    assert manifest is not None, "the manifest of the run is what train returns"


def test_the_run_directory_is_there_before_the_run_writes_into_it(training):
    """The model plot is written into the run directory before the manifest is, so it has to exist first.

    A run under a name used for the first time is the ordinary case, and creating the
    directory only when the manifest was saved left that run failing on the plot with no
    such file or directory.
    """
    train_module.train(replace(TrainConfig(), dataset=OTHER_DATASET, name="trial"))

    assert training["run_dir"].created == [{"parents": True, "exist_ok": True}]


def test_the_score_is_measured_on_the_held_out_spectra(training):
    """The accuracy a report quotes is the model's on the spectra it did not train on."""
    train_module.train(replace(TrainConfig(), dataset=OTHER_DATASET, name="trial"))

    scored = training["model"].scored
    assert len(scored) == 1, "one evaluation per run, on the test split"
    assert scored[0]["x"].shape[0] == 2, "the two spectra held out of the four-fixture"
    assert scored[0]["y"].shape[0] == 2
    assert scored[0]["verbose"] == 0, "the score is read, not printed a second time"


def test_one_seeded_generator_carries_the_whole_data_side(training):
    """The split and the augmentation draw from the same generator, built from the config.

    Two generators seeded the same way would give the same numbers here, but the run would
    then be splitting and augmenting from separate streams while the record named one seed.
    """
    train_module.train(replace(TrainConfig(), dataset=OTHER_DATASET, name="trial", seed=99))

    rng = training["rng"]
    assert isinstance(rng, np.random.Generator)
    assert training["augmentation_rng"] is rng, "one stream, not two that happen to agree"


def test_the_held_out_spectra_follow_the_config_seed(training):
    """The same config holds out the same spectra; a different seed holds out others.

    This is the draw that decides which spectra the reported accuracy was measured on, so
    it is the one a replay has to reproduce.
    """
    cfg = replace(TrainConfig(), dataset=OTHER_DATASET, name="trial", seed=99)

    train_module.train(cfg)
    first = tuple(training["split"][1])
    train_module.train(cfg)
    again = tuple(training["split"][1])

    splits = set()
    for seed in range(1, 7):
        train_module.train(replace(cfg, seed=seed))
        splits.add(tuple(training["split"][1]))

    assert first == again
    assert len(splits) > 1, "the fixture has more than one possible split, so seeds must differ"


def test_the_framework_seed_is_the_config_seed(training):
    """Keras draws the initial weights from its own global state, so it is seeded too.

    Without this, two runs of the same config would share a dataset and a split and still
    produce different models, which is the case the identity check exists to rule out.
    """
    train_module.train(replace(TrainConfig(), dataset=OTHER_DATASET, name="trial", seed=4))

    assert training["seeds"] == [4]
