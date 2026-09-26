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

A manifest also carries what a *replay* needs: the seed, the dataset it was trained on
and digests of that dataset's contents and of the order its files were read in, and the
versions of the packages that ran it. A reported accuracy is only defensible from the
run that produced it, and these are the facts that describe that run.
"""

from __future__ import annotations

import glob
import hashlib
import json
import pickle
from dataclasses import asdict, dataclass, field
from datetime import datetime
from importlib import metadata as importlib_metadata
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

#: The distributions a run depends on directly, under the names they are installed as.
#: Recorded one by one rather than as a full environment freeze: these are what a result
#: depends on, and a few hundred transitive packages would bury them. ``scikit-learn``
#: rather than ``sklearn`` because that is the distribution name, whatever the import is
#: called. The names are the ones ``pyproject.toml`` declares, and a test asserts the two
#: lists agree, so a dependency added to one cannot go unrecorded by the other.
RECORDED_PACKAGES = (
    "keras",
    "tensorflow",
    "numpy",
    "scipy",
    "scikit-learn",
    "ramanspy",
    "kneed",
    "pysptools",
    "matplotlib",
    "pandas",
    "pydot",
)

#: Read size when digesting a dataset file. Large enough that the 81 MB dataset is a
#: second of reading rather than thousands of small calls.
HASH_CHUNK_BYTES = 1 << 20


def package_versions(names=RECORDED_PACKAGES):
    """The installed version of each package a run depends on.

    Args:
        names: Distribution names to look up. Defaults to ``RECORDED_PACKAGES``.

    Returns:
        dict[str, str]: One entry per name. A package that is not installed is recorded
            as ``"not installed"`` rather than left out, since a missing entry reads as a
            name nobody asked for.
    """
    versions = {}
    for name in names:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def hash_dataset(path, chunk_bytes=HASH_CHUNK_BYTES):
    """A digest of a directory's contents, so the data behind a number can be identified.

    Each file is digested over its bytes, the per-file digests are listed in sorted path
    order, and that list is digested again. Contents only: copying the directory
    elsewhere, moving it, or touching a file's modification time leaves the digest alone,
    while editing, adding or removing a spectrum changes it.

    Args:
        path (str | Path): The dataset directory.
        chunk_bytes (int): Read size per file.

    Returns:
        str: ``"sha256:<hex>"``.

    Raises:
        FileNotFoundError: The directory does not exist, which would otherwise be
            recorded as the digest of an empty dataset.
    """
    directory = Path(path).expanduser()
    if not directory.is_dir():
        raise FileNotFoundError(f"{directory} is not a directory, so there is no dataset to digest")

    digests = []
    for file in sorted(entry for entry in directory.rglob("*") if entry.is_file()):
        digest = hashlib.sha256()
        with open(file, "rb") as handle:
            for chunk in iter(lambda: handle.read(chunk_bytes), b""):
                digest.update(chunk)
        digests.append(f"{digest.hexdigest()}  {file.relative_to(directory).as_posix()}")

    return "sha256:" + hashlib.sha256("\n".join(digests).encode("utf-8")).hexdigest()


def dataset_order_digest(path):
    """A digest of the dataset file names in the order the loader reads them.

    ``load_CNN_data`` hands the spectra to training in the order ``ramanspy`` walks the
    directory, and the train and test index arrays are positions in that order. That
    order comes from ``glob``, so it is stable for a directory nobody touches and can
    differ for a fresh copy of the same data. This digest records which order a run saw,
    so a replay that would split the spectra a different way is detectable rather than
    silent.

    The glob here is the one the loader makes, with the same pattern and the same
    directory, so the two see the names in the same order.

    Args:
        path (str | Path): The dataset directory.

    Returns:
        str: ``"sha256:<hex>"`` over the file names, in order.
    """
    directory = Path(path).expanduser()
    names = [Path(name).name for name in glob.glob(f"{directory}/*.txt")]
    return "sha256:" + hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RunManifest:
    """What a model was trained with, as written to and read back from ``run.json``.

    Every field below ``classes`` has a default, so a manifest written before they
    existed still loads: a replay that is missing facts reports them as absent rather
    than refusing to open the run.

    Attributes:
        seed: The seed the run was trained under, or None for a manifest written before
            one was recorded.
        dataset: The dataset path as it was given, unexpanded, so it reads the way the
            command line that produced it did.
        dataset_hash: ``hash_dataset`` of that directory.
        dataset_order: ``dataset_order_digest`` of it, which is what pins the order the
            train and test split was taken in.
        package_versions: The versions the run was trained with, from
            ``package_versions``.
        metrics: What the fitted model scored, from ``model.evaluate`` on the held-out
            spectra.
        config: The training config as it was passed, so the epoch cap and the
            augmentation counts travel with the run.
    """

    name: str
    created: str
    input_size: int
    num_classes: int
    x_min: float
    x_max: float
    classes: list = field(default_factory=list)
    seed: int | None = None
    dataset: str | None = None
    dataset_hash: str | None = None
    dataset_order: str | None = None
    package_versions: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)

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


def save_run(directory, config, model, encoder, metrics=None):
    """Writes a trained model and everything needed to use and to re-run it into one directory.

    Args:
        directory (str | Path): Destination, usually from ``run_directory``. Its last
            path component is the run name recorded in the manifest, so the name a run
            is listed under and the name it reports cannot disagree.
        config (TrainConfig): The config the run was trained with. Every manifest field
            that comes from a setting is read out of it, so the record cannot disagree
            with what was trained: there is one object rather than two lists that have to
            be kept in step.
        model: The trained Keras model.
        encoder: The fitted ``LabelEncoder`` that maps predictions back to minerals.
        metrics (dict, optional): What the fitted model scored, usually the loss and
            accuracy from ``model.evaluate`` on the held-out spectra. Recorded as given.

    Returns:
        RunManifest: What was written, for the caller to report.

    Raises:
        FileNotFoundError: The dataset the config names is not there. The digest of it is
            part of the record, and a run whose data has gone cannot be replayed, so this
            is refused rather than recorded as a manifest with the hash left out.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    manifest = RunManifest(
        name=directory.name,
        created=datetime.now().isoformat(timespec="seconds"),
        input_size=int(config.input_size),
        num_classes=len(getattr(encoder, "classes_", [])),
        x_min=float(config.x_min),
        x_max=float(config.x_max),
        classes=[str(label) for label in getattr(encoder, "classes_", [])],
        seed=config.seed,
        dataset=config.dataset,
        dataset_hash=hash_dataset(config.dataset),
        dataset_order=dataset_order_digest(config.dataset),
        package_versions=package_versions(),
        metrics=dict(metrics or {}),
        config=asdict(config),
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
