"""The saved-map routes the Data screen calls.

Every change answers with the whole list rather than with an acknowledgement, so the
screen redraws from one response and what it shows is what the file holds. That is what
these tests check: the answer is the list, and the disk agrees with it.

One test goes the whole way and loads a map through the server, because the entry is
written by the watcher thread that publishes a finished read and nothing else covers
that the two halves meet.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from app import jobs, saved_maps
from src import paths
from tests import synthetic

pytestmark = pytest.mark.usefixtures("isolated_artifacts")


@pytest.fixture
def map_root(monkeypatch, tmp_path) -> Path:
    """A map root holding two maps, named through the environment."""
    root = tmp_path / "data"
    (root / "SB008").mkdir(parents=True)
    (root / "SB009").mkdir()
    monkeypatch.setenv(paths.DATA_DIR_ENV, str(root))
    return root


def test_the_list_is_empty_before_a_map_is_opened(client, map_root):
    """The screen asks for the list on arrival, when there is nothing to show."""
    answer = client.get("/api/maps").json()

    assert answer == {"root": str(map_root.resolve()), "entries": []}


def test_the_list_names_the_root_its_paths_are_relative_to(client, map_root):
    """A relative path means nothing on its own, so the root travels with the list."""
    saved_maps.remember("First sample", map_root / "SB008")

    answer = client.get("/api/maps").json()

    assert answer["root"] == str(map_root.resolve())
    assert answer["entries"] == [{"label": "First sample", "path": "SB008"}]


def test_the_list_carries_the_grid_each_map_was_read_with(client, map_root):
    """A row opens its map in one press, which it can only do if the list says how the map
    was laid out: the files are numbered, and their count does not give the layout."""
    saved_maps.remember("First sample", map_root / "SB008", rows=4, cols=4)

    answer = client.get("/api/maps").json()

    assert answer["entries"] == [
        {"label": "First sample", "path": "SB008", "rows": 4, "cols": 4}
    ]


def test_renaming_a_label_answers_the_list_and_leaves_the_folder(client, map_root):
    """The label is the app's own name for a map, so nothing on disk moves."""
    saved_maps.remember("", map_root / "SB008")

    answer = client.post("/api/maps/label", json={"path": "SB008", "label": "Before annealing"}).json()

    assert answer["entries"] == [{"label": "Before annealing", "path": "SB008"}]
    assert (map_root / "SB008").is_dir()
    assert saved_maps.entries() == answer["entries"]


def test_renaming_a_path_moves_the_folder_and_answers_the_list(client, map_root):
    """The path is what the app stores, and the folder is what it names."""
    saved_maps.remember("Before annealing", map_root / "SB008")

    answer = client.post("/api/maps/path", json={"path": "SB008", "name": "SB008-run2"}).json()

    assert answer["entries"] == [{"label": "Before annealing", "path": "SB008-run2"}]
    assert (map_root / "SB008-run2").is_dir()


def test_deleting_answers_the_list_and_removes_the_folder(client, map_root):
    """Delete is the one action that reaches the disk, so it is checked against it."""
    saved_maps.remember("", map_root / "SB008")
    saved_maps.remember("", map_root / "SB009")

    answer = client.post("/api/maps/delete", json={"path": "SB008"}).json()

    assert answer["entries"] == [{"label": "SB009", "path": "SB009"}]
    assert not (map_root / "SB008").exists()


def test_a_change_naming_no_saved_map_is_refused(client, map_root):
    """An empty path is a request that names nothing, and it says so."""
    for route in ("/api/maps/label", "/api/maps/path", "/api/maps/delete"):
        response = client.post(route, json={"path": "", "label": "x", "name": "x"})

        assert response.status_code == 400, route
        assert "No saved map was named" in response.json()["detail"]


def test_a_change_naming_a_map_the_list_does_not_hold_is_refused(client, map_root):
    """The folder is there, but no entry names it, so nothing is done to it."""
    response = client.post("/api/maps/delete", json={"path": "SB008"})

    assert response.status_code == 404
    assert (map_root / "SB008").is_dir()


def test_a_folder_name_that_is_not_a_folder_name_is_refused(client, map_root):
    """The name reaches the filesystem, so a path is refused before the move."""
    saved_maps.remember("", map_root / "SB008")

    response = client.post("/api/maps/path", json={"path": "SB008", "name": "../escape"})

    assert response.status_code == 400
    assert (map_root / "SB008").is_dir()


def test_a_label_that_is_empty_is_refused(client, map_root):
    """A row with no name is a row with nothing to click."""
    saved_maps.remember("", map_root / "SB008")

    response = client.post("/api/maps/label", json={"path": "SB008", "label": "  "})

    assert response.status_code == 400


def test_a_path_climbing_out_of_the_root_is_refused(client, map_root, tmp_path):
    """The list file is text the user can edit, so a stored path is not trusted."""
    escape = tmp_path / "escape"
    escape.mkdir()
    saved_maps.saved_maps_path().write_text(
        '{"entries": [{"label": "escape", "path": "../escape"}]}', encoding="utf-8"
    )

    response = client.post("/api/maps/delete", json={"path": "../escape"})

    assert response.status_code == 400
    assert escape.is_dir()


def test_loading_a_map_through_the_server_saves_it(client, map_root):
    """The whole path: a read is started over HTTP, and the list gains the map it read.

    The entry is written by the watcher thread that publishes a finished read, so this is
    the one test that covers the load and the list meeting without either being stubbed.
    It carries the grid the read was started with, which is what the row needs to open the
    map again without the size being set by hand.
    """
    synthetic.write_grid(map_root / "SB008")

    started = client.post("/api/map", json={
        "path": str(map_root / "SB008"), "rows": synthetic.ROWS, "cols": synthetic.COLS,
    })
    assert started.status_code == 200, started.text

    job = started.json()["job"]
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        current = client.get(f"/api/job/{job['id']}").json()
        if current["state"] != "running":
            break
        time.sleep(0.1)
    assert current["state"] == "done", current

    assert client.get("/api/maps").json()["entries"] == [
        {
            "label": "SB008",
            "path": "SB008",
            "rows": synthetic.ROWS,
            "cols": synthetic.COLS,
        },
    ]
    assert jobs.current() is not None
