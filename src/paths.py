"""Where the project reads and writes files on disk.

Every path here is built from this file's own location rather than from the current
working directory, so a script writes to the same place no matter which directory it
was started from.
"""

from pathlib import Path

#: The repository root, which is the parent of the package this module lives in.
REPO_ROOT = Path(__file__).resolve().parents[1]

#: Directory names below the repository root.
OUTPUTS_DIRNAME = "outputs"
ARTIFACTS_DIRNAME = "artifacts"


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
