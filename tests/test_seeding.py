"""What a training run's seed decides, and that nothing else decides it.

A run draws in five places: which spectrum each class holds out for the test set, the
shift, the noise and the mixing coefficient of every augmentation, and the network's
initial weights. The first four are drawn from a generator built from the run's seed, so
they are repeatable and can be tested by giving the same seed twice; the weights are
drawn by Keras from its own global state, so ``train`` seeds that too.

The last test is the one that keeps this true: it reads the package's syntax tree and
fails if any module reaches for the global generator again.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from src.cnn import cnn_preprocessing as pre
from src.config import TrainConfig

REPO_ROOT = Path(__file__).resolve().parent.parent

#: The only thing a module may reach for through ``np.random``: the constructor of a
#: generator that belongs to one run. Everything else on that namespace reads or writes
#: the process-wide state, which is what makes a result unrepeatable.
ALLOWED_GLOBAL_RNG_ATTRIBUTES = {"default_rng", "Generator"}


def _spectrum(points=16, offset=0.0):
    """One standardised spectrum, shaped the way the augmentation functions expect."""
    return (np.linspace(0.0, 1.0, points) + offset).reshape(-1, 1)


def _split(seed):
    """The held-out and kept indices for one seed, over the same six labelled spectra."""
    labels = np.array(["quartz"] * 3 + ["calcite"] * 3)
    return pre.leave_one_out_split(labels, np.random.default_rng(seed))


def _augmented(seed, n_shift=2, n_noise=2):
    """One augmentation pass over the same spectra, from a seed.

    The spectra are the same for every seed, so anything that differs between two calls
    of this came from the augmentation rather than from which spectra were augmented.
    """
    labels = np.array(["quartz"] * 3 + ["calcite"] * 3)
    x = np.array([_spectrum(offset=i * 0.1) for i in range(len(labels))])
    return pre.build_augmented_dataset(x, labels, n_shift, n_noise, np.random.default_rng(seed))


def test_the_split_is_the_same_for_the_same_seed():
    """Which spectra the accuracy is measured on has to be a property of the config."""
    train_index, test_index = _split(7)
    again_train, again_test = _split(7)

    assert np.array_equal(train_index, again_train)
    assert np.array_equal(test_index, again_test)


def test_the_split_moves_with_the_seed():
    """A seed that changed nothing would not be a seed. Five draws from three spectra."""
    splits = {tuple(_split(seed)[1]) for seed in range(5)}

    assert len(splits) > 1, "the held-out spectrum is drawn, so different seeds have to differ"


def test_one_spectrum_is_held_out_per_class():
    """The split is leave-one-out: a class of one stays in, a class of more loses one."""
    labels = np.array(["quartz"] * 3 + ["calcite"] * 3 + ["pyrite"])
    train_index, test_index = pre.leave_one_out_split(labels, np.random.default_rng(0))

    assert sorted(labels[test_index]) == ["calcite", "quartz"], "one per class that has more than one"
    assert "pyrite" not in labels[test_index], "a class of one has nothing to spare"


def test_the_augmentation_is_the_same_for_the_same_seed():
    """The augmented set is part of what the network is fitted on, so it has to repeat."""
    first = _augmented(11)
    second = _augmented(11)

    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])


def test_the_augmentation_moves_with_the_seed():
    """Different seeds, different shifts, noise and mixing coefficients.

    The spectra handed in are the same either way, so the difference has to come from
    the draws and not from which spectra were drawn together.
    """
    first = _augmented(11)
    second = _augmented(12)

    assert np.array_equal(first[1], second[1]), "the same spectra are augmented either way"
    assert not np.array_equal(first[0], second[0]), "the augmented intensities have to differ"


def test_the_shifted_spectrum_is_never_the_unshifted_one():
    """A zero shift would add a duplicate of the original to the training set."""
    spectrum = np.arange(16, dtype=float).reshape(-1, 1)

    for _ in range(20):
        assert not np.array_equal(pre.augment_shift(spectrum, np.random.default_rng(_)), spectrum)


def test_the_augmented_set_is_the_originals_plus_the_copies():
    """Two shifts and one noise per original, plus one combination per pair in a class."""
    labels = np.array(["quartz"] * 3)
    x = np.array([_spectrum(offset=i * 0.1) for i in range(3)])

    x_all, y_all = pre.build_augmented_dataset(x, labels, 2, 1, np.random.default_rng(3))

    combinations = 3  # 3 choose 2
    assert len(x_all) == 3 + (2 + 1) * 3 + combinations
    assert len(y_all) == len(x_all)
    assert np.array_equal(x_all[:3], x), "the originals come first, unaugmented"


def test_a_class_with_one_spectrum_gets_no_combinations():
    """There is no pair to combine, and a combination of a spectrum with itself is not augmentation."""
    x_all, _ = pre.build_augmented_dataset(
        np.array([_spectrum()]), np.array(["quartz"]), 1, 0, np.random.default_rng(3)
    )

    assert len(x_all) == 2, "the original and its one shifted copy"


def _global_rng_uses():
    """Every attribute access on ``np.random`` under ``src/``, as file:line:text."""
    uses = []
    for file in sorted((REPO_ROOT / "src").rglob("*.py")):
        tree = ast.parse(file.read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Attribute)
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id == "np"
                and node.value.attr == "random"
            ):
                uses.append(f"{file.relative_to(REPO_ROOT)}:{node.lineno}:np.random.{node.attr}")
    return uses


def test_no_module_draws_from_the_global_generator():
    """The global generator is process-wide, so a draw from it is a draw from no seed.

    Checked against the parsed tree rather than the text, so the docstrings that explain
    this rule are not counted as breaking it.
    """
    offenders = [
        use for use in _global_rng_uses()
        if use.rsplit(".", 1)[-1] not in ALLOWED_GLOBAL_RNG_ATTRIBUTES
    ]

    assert offenders == [], "these draw from the process-wide generator: " + ", ".join(offenders)


def test_the_check_above_would_notice_a_draw(monkeypatch, tmp_path):
    """The guard is only worth having if it fails on the thing it forbids."""
    stray = tmp_path / "src"
    (stray / "pkg").mkdir(parents=True)
    (stray / "pkg" / "drawn.py").write_text("import numpy as np\n\nnp.random.seed(0)\n")
    monkeypatch.setattr("tests.test_seeding.REPO_ROOT", tmp_path)

    offenders = [
        use for use in _global_rng_uses()
        if use.rsplit(".", 1)[-1] not in ALLOWED_GLOBAL_RNG_ATTRIBUTES
    ]

    assert offenders == ["src/pkg/drawn.py:3:np.random.seed"]


def test_the_config_default_is_a_seed():
    """A default of None would leave the shipped configuration as unrepeatable as before."""
    assert isinstance(TrainConfig.seed, int)
