"""The one map the interface holds, as files on disk.

The screens work on one map at a time, and the map is read from disk once. That
read writes the loader's four arrays as ``.npy`` files in a session directory, and
the server memory-maps them from there, so a band and a spectrum cost no disk
access and the data behind them is the same object for every screen.

Making the session a directory rather than an object in memory is what makes "one
map at a time" a property of the filesystem: loading another map replaces the
directory, and there is nowhere for a second map to live. A half-written session
is never the current one, because the arrays are written into a job's own
directory and moved into place only once the whole read has finished.

Nothing here imports the web framework. The job worker runs in a separate process
and imports this module to write a session, and that process has no server in it.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np

from src.data.loader import Cube
from src.paths import artifacts_dir

logger = logging.getLogger(__name__)

#: The directory holding the current session, below ``artifacts``.
SESSIONS_DIRNAME = "sessions"

#: The name of the session directory that is in use. There is only ever one.
CURRENT_NAME = "current"

#: The name of the file recording what was read and with what.
META_NAME = "session.json"

#: The array files a session directory holds, and the ``Cube`` attribute each one is.
ARRAY_NAMES = ("cube", "spectra", "axis", "pixel_map")


def sessions_dir() -> Path:
    """Returns:
        Path: ``<artifacts>/sessions``, created if it does not already exist.
    """
    directory = artifacts_dir() / SESSIONS_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def current_path() -> Path:
    """Returns:
        Path: The session directory in use, whether or not it holds anything yet.
    """
    return sessions_dir() / CURRENT_NAME


def metadata() -> dict | None:
    """What the current session holds, as it was written.

    Returns:
        dict or None: The recorded facts about the loaded map, or None when no map is
        loaded. A directory without its record is treated as no session rather than as
        one with unknown contents, since there is nothing to draw from it.
    """
    path = current_path() / META_NAME
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_cube() -> Cube | None:
    """The current session's arrays, memory-mapped.

    Returns:
        Cube or None: The cube the loader produced, with every array read lazily from
        its file, or None when no map is loaded.
    """
    directory = current_path()
    if not (directory / META_NAME).is_file():
        return None

    arrays = {}
    for name in ARRAY_NAMES:
        path = directory / f"{name}.npy"
        if not path.is_file():
            logger.warning("Session %s is missing %s.npy, so it is not usable.", directory, name)
            return None
        arrays[name] = np.load(path, mmap_mode="r")
    return Cube(**arrays)


def write(directory: Path, cube: Cube, facts: dict) -> dict:
    """Writes a cube and its record into a directory, to be moved into place later.

    Called from the job process, which is why it takes the directory to write into
    rather than writing to the current session directly: the parent decides when a
    finished read becomes the session, and a read that fails part way leaves the map
    that was already loaded alone.

    Args:
        directory (Path): The directory to write into, created if absent.
        cube (Cube): The arrays to save.
        facts (dict): What the session record should say about the map.

    Returns:
        dict: The record as written, with the load time added.
    """
    directory.mkdir(parents=True, exist_ok=True)
    for name in ARRAY_NAMES:
        np.save(directory / f"{name}.npy", getattr(cube, name))

    record = {
        **facts,
        # Identifies this read and nothing else, so anything caching work against the
        # loaded map can tell one map from the next. A load time cannot do that job: its
        # resolution is seconds, and loading two maps inside one second is a thing a
        # person does.
        "session_id": uuid4().hex,
        "loaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (directory / META_NAME).write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def replace_with(directory: Path) -> dict:
    """Makes a finished read the current session, releasing whatever was loaded.

    The directory that was current is moved aside rather than deleted first, so the
    session is out of the way before the new one takes its name and there is no
    moment where the name points at a directory that is being written into.

    Args:
        directory (Path): A directory written by :func:`write`.

    Returns:
        dict: The record the new session was written with.

    Raises:
        FileNotFoundError: The directory holds no session record, so it is not a
            finished read and must not become the one in use.
    """
    record = json.loads((directory / META_NAME).read_text(encoding="utf-8"))
    current = current_path()
    released = sessions_dir() / f"{CURRENT_NAME}.released"

    if released.exists():
        shutil.rmtree(released)

    if current.exists():
        current.rename(released)
    directory.rename(current)

    if released.exists():
        shutil.rmtree(released)

    logger.info("Session replaced with %s.", record.get("directory", directory))
    return record


def clear() -> None:
    """Releases the current map, leaving no session behind."""
    current = current_path()
    if current.exists():
        shutil.rmtree(current)
        logger.info("Session released.")
