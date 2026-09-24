"""One directory per training run, and the manifest that describes it.

Training writes the model, the label encoder, the weights and a ``run.json`` into a
single directory, and prediction loads all of them from that directory. The settings a
model was trained with therefore travel with the model instead of living as constants
in whichever script happens to load it.

The input length is the setting that matters most here. It decides how long a spectrum
is made before it reaches the network: the measured intensities are interpolated onto
an axis of that many points running from ``x_min`` to ``x_max``. Nothing recorded it
before, so the predicting script carried its own value and a model trained for one
length could be loaded by code that built spectra of another. Keras accepts either, so
the mistake was invisible.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import keras
import numpy as np

from src import paths

#: File names inside a run directory. ``model.keras`` already contains the weights,
#: so there is no separate weights file to keep in step with it.
MANIFEST_NAME = "run.json"
MODEL_NAME = "model.keras"
ENCODER_NAME = "label_encoder.pkl"
TRAINING_CURVES_NAME = "training_curves.png"
MODEL_PLOT_NAME = "model_plot.png"

RUNS_DIRNAME = "runs"


@dataclass(frozen=True)
class RunManifest:
    """What a model was trained with, as written to and read back from ``run.json``."""

    name: str
    created: str
    input_size: int
    num_classes: int
    x_min: float
    x_max: float
    classes: list = field(default_factory=list)

    def to_json(self):
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, text):
        """Rebuilds a manifest from the text of a ``run.json``."""
        return cls(**json.loads(text))


@dataclass(frozen=True)
class Run:
    """A run directory, its manifest, and the artifacts loaded out of it."""

    directory: Path
    manifest: RunManifest
    model: object
    encoder: object


def new_run_name(prefix="run"):
    """A run name that sorts by time, for when no name is given on the command line."""
    return f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}"


def runs_root(root=None):
    """The directory that holds every run.

    Args:
        root (str | Path | None): An artifacts directory to use instead of the
            project's own. Tests pass a temporary directory here.

    Returns:
        Path: ``<artifacts>/runs``.
    """
    base = paths.artifacts_dir() if root is None else Path(root)
    return base / RUNS_DIRNAME


def run_directory(name, root=None):
    """The directory for one named run.

    The directory is not created by this call.

    Args:
        name (str): The run name, as passed to training.

    Returns:
        Path: ``<artifacts>/runs/<name>``.
    """
    return runs_root(root) / name


def list_runs(root=None):
    """The names of the runs on disk, most recently created first.

    Args:
        root (str | Path | None): See ``runs_root``.

    Returns:
        list[str]: Names of directories that hold a manifest. Empty if there are none.
    """
    directory = runs_root(root)
    if not directory.is_dir():
        return []

    created = {}
    for child in directory.iterdir():
        manifest_path = child / MANIFEST_NAME
        if child.is_dir() and manifest_path.is_file():
            try:
                created[child.name] = RunManifest.from_json(manifest_path.read_text()).created
            except (json.JSONDecodeError, TypeError):
                # A directory someone has edited by hand is still a run; it just sorts
                # by name rather than by the time it was written.
                created[child.name] = ""

    return sorted(created, key=lambda name: (created[name], name), reverse=True)


def latest_run(root=None):
    """The most recently created run directory, or None if there are no runs."""
    names = list_runs(root)
    return run_directory(names[0], root) if names else None


def check_model_input_length(manifest, model_input_shape):
    """Confirms a manifest and a model agree on how long a spectrum is.

    Args:
        manifest (RunManifest): The manifest that came with the model.
        model_input_shape (tuple | None): The shape the model declares, as
            ``model.input_shape``, for example ``(None, 2000, 1)``.

    Raises:
        ValueError: The model expects a different number of points per spectrum than
            the manifest claims. This is worth refusing rather than warning about:
            the network will accept whatever length it is handed, so a mismatch
            produces predictions from a model applied outside its training geometry.
    """
    if model_input_shape is None:
        raise ValueError("the model does not declare an input shape, so the manifest cannot be checked")

    # A convolutional model declares (batch, points, channels); a dense one declares
    # (batch, points).
    actual = model_input_shape[1] if len(model_input_shape) == 3 else model_input_shape[-1]
    if int(actual) != int(manifest.input_size):
        raise ValueError(
            f"the model expects {int(actual)} points per spectrum but {MANIFEST_NAME} says "
            f"{int(manifest.input_size)}, so the model and the manifest do not belong together"
        )


def check_axis_covers_training_range(manifest, spectral_axis):
    """Confirms the measured range covers what the model was trained on.

    Args:
        manifest (RunManifest): The manifest that came with the model.
        spectral_axis (np.ndarray): Raman shifts of the spectra about to be predicted.

    Raises:
        ValueError: The spectra do not span the training range. Interpolation onto
            the training axis fills anything outside the measured range with zeros,
            so the model would be shown padding that never appears in its training
            data, and its answer would depend on how narrow the measurement happened
            to be.
    """
    low = float(np.min(spectral_axis))
    high = float(np.max(spectral_axis))
    if low > manifest.x_min or high < manifest.x_max:
        raise ValueError(
            f"the spectra cover {low:.1f} to {high:.1f} cm^-1 but the model was trained on "
            f"{manifest.x_min:.1f} to {manifest.x_max:.1f} cm^-1; resampling outside the "
            f"measured range would pad the spectrum with zeros"
        )


def save_run(directory, name, model, encoder, input_size, num_classes, x_min, x_max):
    """Writes a trained model and everything needed to use it into one directory.

    Args:
        directory (str | Path): Destination, usually from ``run_directory``.
        name (str): The run name, recorded in the manifest.
        model: The trained Keras model.
        encoder: The fitted ``LabelEncoder`` that maps predictions back to minerals.
        input_size (int): Points per spectrum the model was trained on.
        num_classes (int): Number of mineral classes.
        x_min (float): Low end of the training wavenumber range.
        x_max (float): High end of the training wavenumber range.

    Returns:
        RunManifest: What was written, for the caller to report.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    manifest = RunManifest(
        name=name,
        created=datetime.now().isoformat(timespec="seconds"),
        input_size=int(input_size),
        num_classes=int(num_classes),
        x_min=float(x_min),
        x_max=float(x_max),
        classes=[str(label) for label in getattr(encoder, "classes_", [])],
    )

    model.save(directory / MODEL_NAME)
    with open(directory / ENCODER_NAME, "wb") as handle:
        pickle.dump(encoder, handle)
    (directory / MANIFEST_NAME).write_text(manifest.to_json())

    return manifest


def load_run(directory):
    """Loads a run directory and checks that its parts agree with each other.

    Args:
        directory (str | Path): A directory written by ``save_run``.

    Returns:
        Run: The directory, its manifest, its model and its label encoder.

    Raises:
        FileNotFoundError: The directory holds no manifest, so it is not a run.
        ValueError: The model and the manifest disagree about the input length.
    """
    directory = Path(directory)
    manifest_path = directory / MANIFEST_NAME
    if not manifest_path.is_file():
        available = list_runs()
        hint = f" Runs available: {', '.join(available)}." if available else ""
        raise FileNotFoundError(
            f"{directory} is not a run directory: it holds no {MANIFEST_NAME}.{hint}"
        )

    manifest = RunManifest.from_json(manifest_path.read_text())
    model = keras.models.load_model(directory / MODEL_NAME)
    check_model_input_length(manifest, model.input_shape)

    with open(directory / ENCODER_NAME, "rb") as handle:
        encoder = pickle.load(handle)

    return Run(directory=directory, manifest=manifest, model=model, encoder=encoder)
