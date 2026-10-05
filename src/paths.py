"""Where the project reads and writes files on disk.

The directories the project writes to are built from this file's own location rather
than from the current working directory, so a script writes to the same place no matter
which directory it was started from.

The one exception is the map root, which is where the user's own maps are read from. It
is named by the environment when the interface runs in a container, and falls back to a
directory beside the repository root otherwise, so it is not tied to this file's location
either way.
"""

import os
from pathlib import Path

#: The repository root, which is the parent of the package this module lives in.
REPO_ROOT = Path(__file__).resolve().parents[1]

#: Directory names below the repository root.
OUTPUTS_DIRNAME = "outputs"
ARTIFACTS_DIRNAME = "artifacts"
DATA_DIRNAME = "data"

#: The environment variable naming the directory the maps are opened from. A container
#: sets it to where its bind mount appears; a checkout leaves it unset and gets ``data``
#: below the repository root, which is the directory that mount points at.
DATA_DIR_ENV = "RAMAN_DATA_DIR"


def repo_root():
    """Returns:
        Path: The repository root directory.
    """
    return REPO_ROOT


def outputs_dir():
    """The directory generated figures are written to.

    Returns:
        Path: ``<repo root>/outputs``, created if it does not already exist.
    """
    directory = REPO_ROOT / OUTPUTS_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def artifacts_dir():
    """The directory trained models and run records are written to.

    Returns:
        Path: ``<repo root>/artifacts``, created if it does not already exist.
    """
    directory = REPO_ROOT / ARTIFACTS_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def output_path(name):
    """A path for a generated figure.

    Args:
        name (str): File name, for example ``"prediction_map.png"``.

    Returns:
        Path: ``<repo root>/outputs/<name>``, with the directory created.
    """
    return outputs_dir() / name


def data_root() -> Path:
    """The directory the maps are opened from.

    The root is where the interface looks for maps, and every saved map's path is stored
    relative to it. ``DATA_DIR_ENV`` wins when it is set, so a container names its mount
    point; otherwise the root is ``<repo root>/data``.

    The directory is not created. It holds the user's own maps, and when it is missing
    under a bind mount Docker would create it owned by root, which is a directory the
    user could then not write to. The interface reports a missing root instead.

    Returns:
        Path: The map root, whether or not it exists.
    """
    override = os.environ.get(DATA_DIR_ENV, "").strip()
    root = Path(override).expanduser() if override else REPO_ROOT / DATA_DIRNAME
    return root.resolve()
