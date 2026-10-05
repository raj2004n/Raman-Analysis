"""The list of maps the interface has opened before.

An entry is a label, a path relative to the map root and the grid the map was read with,
so the list points at a map rather than holding a copy of one. What that costs is that
renaming a path and deleting an entry reach the disk, and most of these tests are about
the two guards that keep them inside the user's own maps: a stored path is resolved under
the root before it is used, and a folder name has to be one path component.

The root is named through the environment here, as a container names it, so the tests do
not depend on where the checkout keeps its own ``data``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app import saved_maps
from src import paths

pytestmark = pytest.mark.usefixtures("isolated_artifacts")


@pytest.fixture
def map_root(monkeypatch, tmp_path) -> Path:
    """A map root holding two maps, named through the environment."""
    root = tmp_path / "data"
    (root / "SB008").mkdir(parents=True)
    (root / "SB009").mkdir()
    monkeypatch.setenv(paths.DATA_DIR_ENV, str(root))
    return root


def _write_raw(document) -> None:
    """Puts a document in the list file, as hand editing or an older version would."""
    saved_maps.saved_maps_path().write_text(json.dumps(document), encoding="utf-8")


# --------------------------------------------------------------- the list itself


def test_nothing_is_saved_before_a_map_is_opened():
    """A fresh install has no list file, which is an empty list and not an error."""
    assert saved_maps.entries() == []


def test_remembering_a_map_writes_its_label_and_a_path_relative_to_the_root(map_root):
    """The list stores a pointer, so nothing is duplicated on disk."""
    entry = saved_maps.remember("First sample", map_root / "SB008")

    assert entry == {"label": "First sample", "path": "SB008"}
    assert saved_maps.entries() == [entry]


def test_a_map_remembered_without_a_label_is_named_by_its_folder(map_root):
    """The session already reports the folder's name, so it is the default."""
    entry = saved_maps.remember("", map_root / "SB009")

    assert entry["label"] == "SB009"


def test_remembering_a_map_again_moves_it_to_the_front(map_root):
    """The list follows what is being worked on, and holds no path twice."""
    saved_maps.remember("", map_root / "SB008")
    saved_maps.remember("", map_root / "SB009")
    saved_maps.remember("", map_root / "SB008")

    assert [entry["path"] for entry in saved_maps.entries()] == ["SB008", "SB009"]


def test_a_map_outside_the_root_is_not_saved(map_root, tmp_path):
    """It has no path relative to the root, and a path that cannot be named is not stored."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    assert saved_maps.remember("", elsewhere) is None
    assert saved_maps.entries() == []


def test_a_list_file_that_cannot_be_read_is_an_empty_list(map_root):
    """One broken file is not worth refusing every screen over."""
    saved_maps.saved_maps_path().write_text("{ not json", encoding="utf-8")

    assert saved_maps.entries() == []


def test_a_list_file_holding_something_else_is_an_empty_list(map_root):
    """The file is the app's own, so a document of the wrong shape is treated as empty."""
    _write_raw(["SB008"])

    assert saved_maps.entries() == []


def test_entries_that_are_not_entries_are_skipped(map_root):
    """A hand-edited file can hold a row that names nothing usable."""
    _write_raw({"entries": [{"label": "ok", "path": "SB008"}, {"label": 7}, "SB009"]})

    assert saved_maps.entries() == [{"label": "ok", "path": "SB008"}]


# ------------------------------------------------- the grid a map was read with


def test_a_map_read_with_a_grid_keeps_it(map_root):
    """The files in a map are numbered, so their count does not say how they were laid out."""
    entry = saved_maps.remember("", map_root / "SB008", rows=4, cols=4)

    assert entry == {"label": "SB008", "path": "SB008", "rows": 4, "cols": 4}
    assert saved_maps.entries() == [entry]


def test_an_entry_written_before_the_grid_was_kept_is_still_listed(map_root):
    """An older file is a usable list, just one whose maps have to have their grid set."""
    _write_raw({"entries": [{"label": "before", "path": "SB008"}]})

    assert saved_maps.entries() == [{"label": "before", "path": "SB008"}]


@pytest.mark.parametrize("grid", [
    {"rows": 4},
    {"cols": 4},
    {"rows": 0, "cols": 4},
    {"rows": -4, "cols": 4},
    {"rows": "4", "cols": 4},
    {"rows": 4.0, "cols": 4},
    {"rows": True, "cols": 4},
])
def test_a_grid_that_is_not_a_pair_of_counts_is_left_out(map_root, grid):
    """Half a grid, or something that is not a count, would lay the map out wrongly.

    It is left out rather than passed on, so the screen asks for the size instead of
    reading with one it cannot use.
    """
    _write_raw({"entries": [{"label": "odd", "path": "SB008", **grid}]})

    assert saved_maps.entries() == [{"label": "odd", "path": "SB008"}]


def test_remembering_a_map_again_keeps_the_grid_it_already_had(map_root):
    """A caller that knows only the label must not take the grid away.

    Without it the map cannot be opened in one press, and a read whose record does not
    carry a size is not a reason to lose the size the entry was last read with.
    """
    saved_maps.remember("", map_root / "SB008", rows=4, cols=4)

    entry = saved_maps.remember("Renamed", map_root / "SB008")

    assert entry == {"label": "Renamed", "path": "SB008", "rows": 4, "cols": 4}


def test_a_map_read_with_a_new_grid_replaces_the_old_one(map_root):
    """A map can be read at another size, and the list follows what was read."""
    saved_maps.remember("", map_root / "SB008", rows=4, cols=4)

    entry = saved_maps.remember("", map_root / "SB008", rows=3, cols=5)

    assert entry == {"label": "SB008", "path": "SB008", "rows": 3, "cols": 5}


# ------------------------------------------------------------ changing an entry


def test_a_label_can_be_changed_without_moving_the_folder(map_root):
    """The label is the user's name for a map, and the folder is the computer's."""
    saved_maps.remember("", map_root / "SB008")

    saved = saved_maps.rename_label("SB008", "Before annealing")

    assert saved == [{"label": "Before annealing", "path": "SB008"}]
    assert (map_root / "SB008").is_dir()


def test_a_label_that_is_empty_is_refused(map_root):
    """An entry with no label would be a row with nothing to click."""
    saved_maps.remember("", map_root / "SB008")

    with pytest.raises(ValueError):
        saved_maps.rename_label("SB008", "   ")
    assert saved_maps.entries()[0]["label"] == "SB008"


def test_renaming_a_path_moves_the_folder_and_keeps_the_label(map_root):
    """Moving the folder does not restate what the map is called."""
    saved_maps.remember("Before annealing", map_root / "SB008")

    saved = saved_maps.rename_path("SB008", "SB008-run2")

    assert saved == [{"label": "Before annealing", "path": "SB008-run2"}]
    assert (map_root / "SB008-run2").is_dir()
    assert not (map_root / "SB008").exists()


def test_a_grid_survives_a_label_and_a_folder_rename(map_root):
    """Neither rename is a read, so neither has anything to say about the grid."""
    saved_maps.remember("", map_root / "SB008", rows=4, cols=4)

    saved_maps.rename_label("SB008", "Before annealing")
    saved = saved_maps.rename_path("SB008", "SB008-run2")

    assert saved == [
        {"label": "Before annealing", "path": "SB008-run2", "rows": 4, "cols": 4}
    ]


def test_a_folder_name_that_names_a_path_is_refused(map_root):
    """A separator would move the map somewhere other than inside the root."""
    saved_maps.remember("", map_root / "SB008")

    for name in ("../escape", "one/two", ".hidden", ""):
        with pytest.raises(ValueError):
            saved_maps.rename_path("SB008", name)

    assert (map_root / "SB008").is_dir()
    assert saved_maps.entries()[0]["path"] == "SB008"


def test_renaming_onto_a_name_already_taken_is_refused(map_root):
    """Two maps cannot share a folder, so the move is refused rather than merged."""
    saved_maps.remember("", map_root / "SB008")
    saved_maps.remember("", map_root / "SB009")

    with pytest.raises(ValueError):
        saved_maps.rename_path("SB008", "SB009")

    assert (map_root / "SB008").is_dir()
    assert [entry["path"] for entry in saved_maps.entries()] == ["SB009", "SB008"]


def test_a_path_that_climbs_out_of_the_root_is_refused(map_root, tmp_path):
    """The file is text the user can edit, so a stored path is resolved rather than trusted."""
    escape = tmp_path / "escape"
    escape.mkdir()
    _write_raw({"entries": [{"label": "escape", "path": "../escape"}]})

    with pytest.raises(ValueError):
        saved_maps.delete("../escape")
    with pytest.raises(ValueError):
        saved_maps.rename_path("../escape", "renamed")

    assert escape.is_dir()
    assert saved_maps.entries() == [{"label": "escape", "path": "../escape"}]


def test_an_entry_that_is_not_in_the_list_is_refused(map_root):
    """Nothing is changed by naming a map the list does not hold."""
    with pytest.raises(FileNotFoundError):
        saved_maps.rename_label("SB008", "anything")
    with pytest.raises(FileNotFoundError):
        saved_maps.delete("SB008")

    assert (map_root / "SB008").is_dir()


# -------------------------------------------------------------------- deleting


def test_deleting_removes_the_entry_and_the_folder(map_root):
    """Delete is the one action that reaches the disk, and it takes both with it."""
    saved_maps.remember("", map_root / "SB008")
    saved_maps.remember("", map_root / "SB009")

    saved = saved_maps.delete("SB008")

    assert saved == [{"label": "SB009", "path": "SB009"}]
    assert not (map_root / "SB008").exists()
    assert (map_root / "SB009").is_dir()


def test_deleting_a_map_whose_folder_is_already_gone_is_refused(map_root):
    """The folder can be removed outside the app, and the entry is then a dead one.

    It is refused rather than quietly dropped, so that deleting only ever reports having
    dealt with something it actually found.
    """
    saved_maps.remember("", map_root / "SB008")
    (map_root / "SB008").rmdir()

    with pytest.raises(FileNotFoundError):
        saved_maps.delete("SB008")
    assert [entry["path"] for entry in saved_maps.entries()] == ["SB008"]
