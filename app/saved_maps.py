"""The maps the interface has opened before, as a list of names and paths.

An entry is a label, a path relative to the map root, and the grid the map was read with.
The list is not a copy of anything: it points at a directory that already exists, so a map
is in one place on disk. Renaming an entry's path moves that directory and deleting an
entry removes it, which is why every path is resolved against the root before it is used
and one that climbs out is refused.

The grid is kept because a map cannot be read without one and the shape of a directory is
not something the app can work out for itself: the files are numbered, and the same count
of them can be laid out several ways. Keeping the size the map was read with is what lets
the list open a map in one press.

The list is a single file under ``artifacts``, beside the session and the runs. It
survives a container rebuild and stays out of the user's maps, at the cost that a new
machine starts with an empty list even though the maps are still on disk.

Nothing here imports the web framework. The list is changed from the server's watcher
thread as a read finishes, and that thread has no request to answer.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from src.paths import artifacts_dir, data_root

logger = logging.getLogger(__name__)

#: The file the list is written to, below ``artifacts``.
SAVED_MAPS_NAME = "saved_maps.json"

#: The key the entries are held under. The file is an object rather than a bare list so
#: that a fact about the list itself can be added later without changing the shape every
#: reader has to understand.
ENTRIES_KEY = "entries"


def saved_maps_path() -> Path:
    """Returns:
        Path: ``<artifacts>/saved_maps.json``, whether or not it exists.
    """
    return artifacts_dir() / SAVED_MAPS_NAME


def _is_entry(value) -> bool:
    """Whether a value read from the file is an entry this module can use."""
    return (
        isinstance(value, dict)
        and isinstance(value.get("label"), str)
        and isinstance(value.get("path"), str)
    )


def _grid_size(entry: dict) -> dict:
    """The rows and columns an entry holds, if it holds a usable pair.

    A file written before the size was kept, or one edited by hand, may have neither, one,
    or something that is not a count. None of those can lay out a map, so they are left out
    of the entry rather than passed on as a size the screen would try to read with. A
    boolean is refused as well: it is an int as far as Python is concerned, and ``true`` is
    not a grid.
    """
    size = {}
    for key in ("rows", "cols"):
        value = entry.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            size[key] = value
    return size if len(size) == 2 else {}


def entries() -> list[dict]:
    """The saved maps, most recently opened first.

    A file that is missing is an empty list rather than an error, since a fresh install
    has nothing saved. One that cannot be read is reported and treated the same way: a
    file that cannot be parsed has no entries to offer, and refusing every screen over it
    would trade one broken file for a broken interface.

    Returns:
        list of dict: Each entry's ``label`` and ``path``, and its ``rows`` and ``cols``
        when it holds a pair, in the order they are listed.
    """
    path = saved_maps_path()
    if not path.is_file():
        return []

    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.warning("%s cannot be read, so no saved maps are listed.", path)
        return []

    if not isinstance(document, dict):
        logger.warning("%s does not hold a list of maps, so none are listed.", path)
        return []

    saved = []
    for entry in document.get(ENTRIES_KEY, []):
        if _is_entry(entry):
            saved.append({"label": entry["label"], "path": entry["path"], **_grid_size(entry)})
    return saved


def _write(saved: list[dict]) -> None:
    """Writes the whole list, which is the only way it is ever changed."""
    saved_maps_path().write_text(
        json.dumps({ENTRIES_KEY: saved}, indent=2), encoding="utf-8"
    )


def relative(directory) -> str | None:
    """A map directory's path as the list stores it.

    Args:
        directory: The map directory, as a path or a string to expand.

    Returns:
        str or None: The path relative to the map root, or None when the directory is
        empty, is the root itself, or lies outside it. A map outside the root cannot be
        named by the list, so it is opened without being saved rather than saved under a
        path that a later rename or delete would refuse to touch.
    """
    if not str(directory).strip():
        return None
    root = data_root()
    candidate = Path(directory).expanduser().resolve()
    if candidate == root or root not in candidate.parents:
        return None
    return candidate.relative_to(root).as_posix()


def _under_root(stored: str) -> Path:
    """The directory a stored path names, checked to be inside the map root.

    What is stored is relative, so an entry cannot point anywhere on its own; what it can
    still do is climb, since ``..`` is an ordinary path component and the file is a text
    file the user can edit. Every path is resolved and then required to be below the root
    before it is used, which is what keeps a rename or a delete inside the user's own
    maps.

    Raises:
        ValueError: The path is the root itself, or it climbs out of it.
    """
    root = data_root()
    candidate = (root / stored).resolve()
    if candidate == root or root not in candidate.parents:
        raise ValueError(f"{stored} is not a map inside {root}.")
    return candidate


def _find(saved: list[dict], stored: str) -> dict:
    """The entry a stored path names.

    Raises:
        FileNotFoundError: No entry in the list has that path.
    """
    for entry in saved:
        if entry["path"] == stored:
            return entry
    raise FileNotFoundError(f"No saved map is at {stored!r}.")


def _check_name(name: str) -> None:
    """Refuses a name that is not a plain folder name.

    A rename reaches the filesystem, so the name has to be one component and nothing
    that would move the directory to another place: a separator would make the move
    somewhere other than inside the root, and a leading dot would hide the folder the
    picker is meant to offer.

    Raises:
        ValueError: The name is empty, names another path, or is hidden.
    """
    if not name or name != Path(name).name or name.startswith("."):
        raise ValueError(f"{name!r} is not a name a map folder can be given.")


def remember(label: str, directory, rows=None, cols=None) -> dict | None:
    """Records a map as opened, most recently first.

    Called when a read becomes the session. A map that is already saved is moved to the
    front rather than added again, so the list follows what is being worked on and never
    holds the same path twice.

    Args:
        label (str): What to call the map. The directory's own name is used when this is
            empty, which is the name the session already reports.
        directory: The map directory that was read.
        rows, cols (int, optional): The grid the map was read with. A read always has one,
            so these are given whenever the entry comes from a read. A grid already on the
            entry is kept when they are not, so a caller that only knows the label cannot
            take away the size that makes the map loadable.

    Returns:
        dict or None: The entry as written, or None when the directory is outside the map
        root and so cannot be named by the list.
    """
    stored = relative(directory)
    if stored is None:
        logger.info("%s is outside the map root, so it is not saved.", directory)
        return None

    kept = next((one for one in entries() if one["path"] == stored), {})
    entry = {
        "label": label or Path(stored).name,
        "path": stored,
        "rows": rows if rows is not None else kept.get("rows"),
        "cols": cols if cols is not None else kept.get("cols"),
    }
    entry = {"label": entry["label"], "path": entry["path"], **_grid_size(entry)}
    _write([entry, *[kept for kept in entries() if kept["path"] != stored]])
    return entry


def rename_label(stored: str, label: str) -> list[dict]:
    """Changes what a saved map is called. The directory is not touched.

    Args:
        stored (str): The entry's path, as the list holds it.
        label (str): The name to show for it.

    Returns:
        list of dict: The list as it now stands.

    Raises:
        FileNotFoundError: No entry has that path.
        ValueError: The label is empty.
    """
    label = (label or "").strip()
    if not label:
        raise ValueError("A saved map needs a name to be called by.")

    saved = entries()
    _find(saved, stored)["label"] = label
    _write(saved)
    return saved


def rename_path(stored: str, name: str) -> list[dict]:
    """Renames the directory a saved map names, and the path the entry stores.

    The label is left alone. It is the user's own name for the map and the folder's name
    is the computer's, so moving the folder does not restate what the map is called.

    Args:
        stored (str): The entry's path, as the list holds it.
        name (str): The folder name to move it to, inside the same directory.

    Returns:
        list of dict: The list as it now stands.

    Raises:
        FileNotFoundError: No entry has that path, or the directory is already gone.
        ValueError: The name is not a folder name, the stored path climbs out of the
            root, or a directory of that name is already there.
    """
    _check_name(name)
    saved = entries()
    entry = _find(saved, stored)

    directory = _under_root(stored)
    if not directory.is_dir():
        raise FileNotFoundError(f"{directory} is not there to rename.")

    target = directory.parent / name
    if target.exists():
        raise ValueError(f"{name} is already there.")

    directory.rename(target)
    moved = relative(target)
    entry["path"] = moved
    _write(saved)
    logger.info("Saved map moved from %s to %s.", stored, moved)
    return saved


def delete(stored: str) -> list[dict]:
    """Removes a saved map: the entry, and the directory it names.

    The entry is looked up before anything is removed, so this can only ever delete a
    directory the list already names. The screen confirms before it calls this, because
    it reaches the disk and there is no undo.

    Args:
        stored (str): The entry's path, as the list holds it.

    Returns:
        list of dict: The list as it now stands.

    Raises:
        FileNotFoundError: No entry has that path, or the directory is already gone.
        ValueError: The stored path climbs out of the map root.
    """
    saved = entries()
    _find(saved, stored)

    directory = _under_root(stored)
    if not directory.is_dir():
        raise FileNotFoundError(f"{directory} is not there to delete.")

    shutil.rmtree(directory)
    remaining = [entry for entry in saved if entry["path"] != stored]
    _write(remaining)
    logger.info("Saved map %s and its directory were deleted.", stored)
    return remaining
