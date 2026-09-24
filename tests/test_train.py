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
    assert args.name is None


def test_the_command_line_can_over_rule_every_default():
    """A second dataset and a different axis are arguments, which is the point of the config."""
    args = train_module.parse_args(
        ["--dataset", OTHER_DATASET, "--input-size", "1200", "--x-min", "100", "--x-max", "900",
         "--epochs", "5", "--patience", "2", "--n-shift", "0", "--n-noise", "0", "--name", "trial"]
    )

    assert args.dataset == OTHER_DATASET
    assert (args.input_size, args.x_min, args.x_max) == (1200, 100.0, 900.0)
    assert (args.epochs, args.patience) == (5, 2)
    assert (args.n_shift, args.n_noise) == (0, 0)
    assert args.name == "trial"


class _FakeModel:
    """Stands in for the compiled model: records what it was fitted with."""

    def __init__(self):
        self.fitted = {}

    def summary(self):
        return None

    def fit(self, x, y, **kwargs):
        self.fitted = {"x": x, "y": y, **kwargs}
        return "history"


@pytest.fixture
def training(monkeypatch):
    """Replaces every heavy step of a run and records what each one was given."""
    recorded = {"model": _FakeModel()}

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
    monkeypatch.setattr(train_module, "build_augmented_dataset", lambda x, y, n_shift, n_noise: (
        recorded.setdefault("augment", []).append({"n_shift": n_shift, "n_noise": n_noise}) or (x, y)
    ))
    monkeypatch.setattr(train_module, "plot_model", lambda *a, **k: None)
    monkeypatch.setattr(train_module, "show_results", lambda *a, **k: None)

    def fake_save_run(directory, **kwargs):
        recorded["save_run"] = {"directory": directory, **kwargs}
        return type("Manifest", (), {"name": kwargs.get("name", "run")})()

    monkeypatch.setattr(train_module.runs, "save_run", fake_save_run)
    monkeypatch.setattr(train_module.runs, "run_directory", lambda name, root=None: recorded.setdefault(
        "run_dir", type("Dir", (), {"__truediv__": lambda self, other: other})()
    ))
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
    """The manifest has to carry the length and the range, since a prediction must reproduce both."""
    cfg = replace(TrainConfig(), dataset=OTHER_DATASET, input_size=1200,
                  x_min=OTHER_RANGE[0], x_max=OTHER_RANGE[1], name="trial")

    manifest = train_module.train(cfg)

    saved = training["save_run"]
    assert saved["name"] == "trial"
    assert saved["input_size"] == 1200
    assert (saved["x_min"], saved["x_max"]) == OTHER_RANGE
    assert saved["num_classes"] == 2, "two named minerals in the fixture"
    assert manifest is not None, "the manifest of the run is what train returns"
