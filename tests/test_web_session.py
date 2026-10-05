"""The session the interface holds, and the job that reads it.

The read is exercised twice over: once by running the worker in this process, where
the state it writes can be read straight back, and once through the server, where a
child process is started and polled. The second is the path a person takes and is
the slower of the two, so it appears once rather than in every case here.

Every test points the artifacts directory at a temporary directory, so a test run
leaves the checkout's own ``artifacts/`` alone.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pytest

from app import jobs, session, worker
from src import paths
from tests import synthetic

pytestmark = pytest.mark.usefixtures("isolated_artifacts")


def _job_request(directory, **fields):
    """A job directory holding one request, as the runner would write it."""
    directory.mkdir(parents=True, exist_ok=True)
    request = {"kind": "load", **fields}
    (directory / worker.REQUEST_NAME).write_text(json.dumps(request), encoding="utf-8")
    return directory


def _state(directory):
    return json.loads((directory / worker.STATE_NAME).read_text(encoding="utf-8"))


# --------------------------------------------------------------- the grid check


def test_no_map_is_loaded_before_one_is_read(client):
    """The interface starts with nothing held, and says so rather than drawing blanks."""
    assert client.get("/api/session").json() == {"loaded": False}


def test_the_grid_check_counts_the_files_without_reading_them(client, grid_dir):
    """The live check is file discovery, so it answers before anything is opened."""
    response = client.get("/api/grid", params={
        "path": str(grid_dir), "rows": synthetic.ROWS, "cols": synthetic.COLS,
    })

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "expected": synthetic.ROWS * synthetic.COLS,
        "count": synthetic.ROWS * synthetic.COLS,
        "message": None,
    }


def test_the_grid_check_quotes_the_grid_when_a_pixel_is_missing(client, tmp_path):
    """The message is the grid's own, so it names the numbers rather than the count.

    A second copy of this wording in the serve layer is a second place to change it,
    and the screen is meant to show what the loader would have said.
    """
    directory = synthetic.write_grid(tmp_path / "gappy")
    for file in directory.iterdir():
        if file.name.startswith("FullGrid[3]"):
            file.unlink()

    report = client.get("/api/grid", params={
        "path": str(directory), "rows": synthetic.ROWS, "cols": synthetic.COLS,
    }).json()

    assert report["ok"] is False
    assert report["count"] is None
    assert "does not hold the 16 numbered .txt files" in report["message"]
    assert "Pixel numbers with no file: [3]." in report["message"]


def test_the_grid_check_names_a_directory_that_holds_no_numbered_file(client, tmp_path):
    """An empty directory is the first thing someone points the field at."""
    empty = tmp_path / "empty"
    empty.mkdir()

    report = client.get("/api/grid", params={
        "path": str(empty), "rows": 2, "cols": 2,
    }).json()

    assert report["ok"] is False
    assert "No numbered .txt files found" in report["message"]


def test_the_picker_walks_directories_and_offers_only_those(client, tmp_path, monkeypatch):
    """It lists directories, because a map is a directory and a file is not one.

    The walk begins at the map root and may not leave it, so the root is the one
    directory with nothing above it to climb to.
    """
    root = tmp_path / "maps"
    (root / "SB008").mkdir(parents=True)
    (root / "SB009").mkdir()
    (root / "notes.txt").write_text("not a map")
    (root / ".hidden").mkdir()
    monkeypatch.setenv(paths.DATA_DIR_ENV, str(root))

    listing = client.get("/api/directories").json()

    assert listing["directories"] == ["SB008", "SB009"]
    assert listing["path"] == str(root.resolve())
    assert listing["parent"] is None


def test_the_picker_walks_down_and_back_up_to_the_root(client, tmp_path, monkeypatch):
    """Below the root the way up is offered, and it leads back into the root."""
    root = tmp_path / "maps"
    (root / "SB008").mkdir(parents=True)
    monkeypatch.setenv(paths.DATA_DIR_ENV, str(root))

    inside = client.get("/api/directories", params={"path": "SB008"}).json()

    assert inside["path"] == str((root / "SB008").resolve())
    assert inside["parent"] == str(root.resolve())


def test_the_picker_refuses_a_path_outside_the_map_root(client, tmp_path, monkeypatch):
    """A walk that left the root would offer maps the saved list cannot name.

    Every entry stores a path relative to the root, so a map from outside it could not
    be saved, renamed or deleted. The picker is the door that is kept shut.
    """
    root = tmp_path / "maps"
    root.mkdir()
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    monkeypatch.setenv(paths.DATA_DIR_ENV, str(root))

    response = client.get("/api/directories", params={"path": str(outside)})

    assert response.status_code == 403
    assert str(outside) in response.json()["detail"]


def test_the_picker_names_the_root_that_is_not_there(client, tmp_path, monkeypatch):
    """The map directory is the user's own, so a missing one is reported rather than made."""
    missing = tmp_path / "not-built-yet"
    monkeypatch.setenv(paths.DATA_DIR_ENV, str(missing))

    response = client.get("/api/directories")

    assert response.status_code == 404
    assert str(missing) in response.json()["detail"]
    assert not missing.exists()


def test_the_picker_refuses_a_path_that_is_not_a_directory(client, tmp_path, monkeypatch):
    """A file is not a place to walk from, and the message says which one it was."""
    root = tmp_path / "maps"
    root.mkdir()
    file = root / "notes.txt"
    file.write_text("not a directory")
    monkeypatch.setenv(paths.DATA_DIR_ENV, str(root))

    response = client.get("/api/directories", params={"path": str(file)})

    assert response.status_code == 404
    assert str(file) in response.json()["detail"]


# ------------------------------------------------------------------- the loader


def test_a_read_writes_a_session_that_reloads_as_the_same_cube(grid_dir, tmp_path):
    """The four arrays are the load, and the session is those arrays memory-mapped."""
    directory = _job_request(tmp_path / "job", path=str(grid_dir),
                             rows=synthetic.ROWS, cols=synthetic.COLS)

    worker.run(str(directory))

    assert _state(directory)["state"] == "done"
    record = session.replace_with(directory / worker.ARRAYS_DIRNAME)

    assert record["label"] == grid_dir.name
    assert (record["rows"], record["cols"]) == (synthetic.ROWS, synthetic.COLS)
    assert record["files"] == synthetic.ROWS * synthetic.COLS
    assert record["bands"] == synthetic.spectral_axis().size

    cube = session.load_cube()
    assert cube.cube.shape == (synthetic.ROWS, synthetic.COLS, record["bands"])
    # The pixel numbering the loader writes is the one the screens turn a click into,
    # so it is compared against this suite's own reading of the walk rather than
    # against itself.
    np.testing.assert_array_equal(cube.pixel_map, synthetic.expected_pixel_map())
    assert cube.spectra.shape[0] == synthetic.ROWS * synthetic.COLS + 1
    assert cube.spectra[0].sum() == 0  # row 0 is unused, since numbering starts at 1


def test_a_read_that_cannot_finish_records_why(tmp_path):
    """A job that stops for a reason writes the reason, where the screen can show it.

    Without this the server would report a job that ended and no cause, which is the
    one thing a person cannot act on.
    """
    directory = _job_request(tmp_path / "job", path=str(tmp_path / "nowhere"), rows=2, cols=2)

    worker.run(str(directory))

    state = _state(directory)
    assert state["state"] == "error"
    assert "No numbered .txt files found" in state["error"]
    assert not (directory / worker.ARRAYS_DIRNAME).exists()


def test_a_job_naming_no_known_kind_is_refused(tmp_path):
    """A request this server cannot run is reported rather than attempted."""
    directory = tmp_path / "job"
    directory.mkdir()
    (directory / worker.REQUEST_NAME).write_text(json.dumps({"kind": "teleport"}))

    worker.run(str(directory))

    assert "no job called 'teleport'" in _state(directory)["error"]


# ----------------------------------------------------------------- the runner


def test_loading_a_map_over_the_server_makes_it_the_session(client, grid_dir):
    """The whole path: asked for over HTTP, read in a child, then reported back."""
    response = client.post("/api/map", json={
        "path": str(grid_dir), "rows": synthetic.ROWS, "cols": synthetic.COLS,
    })
    assert response.status_code == 200, response.text

    job = response.json()["job"]
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        current = client.get(f"/api/job/{job['id']}").json()
        if current["state"] != "running":
            break
        time.sleep(0.1)

    assert current["state"] == "done", current
    assert current["step"] == "finished"

    loaded = client.get("/api/session").json()
    assert loaded["loaded"] is True
    assert loaded["files"] == synthetic.ROWS * synthetic.COLS
    assert loaded["label"] == grid_dir.name

    cube = session.load_cube()
    np.testing.assert_array_equal(cube.pixel_map, synthetic.expected_pixel_map())


def test_a_directory_that_is_not_a_grid_starts_no_job(client, tmp_path):
    """The check runs before the read, so a bad directory costs no process."""
    empty = tmp_path / "empty"
    empty.mkdir()

    response = client.post("/api/map", json={"path": str(empty), "rows": 2, "cols": 2})

    assert response.status_code == 400
    assert "No numbered .txt files found" in response.json()["detail"]
    assert jobs.current() is None


def test_a_second_job_is_refused_while_one_is_running(client, monkeypatch, grid_dir, tmp_path):
    """One job at a time, and the refusal names the one in the way.

    The directory is a real grid so the refusal being tested is the runner's rather
    than the grid check's, which runs first and would answer 400 for anything else.
    """
    monkeypatch.setattr(jobs, "_CURRENT", jobs.Job(
        id="running", kind="load", request={}, directory=tmp_path, state="running",
    ))

    response = client.post("/api/map", json={
        "path": str(grid_dir), "rows": synthetic.ROWS, "cols": synthetic.COLS,
    })

    assert response.status_code == 409
    assert "load job is already running" in response.json()["detail"]


def test_a_session_that_was_never_recorded_is_not_loaded(client):
    """A directory without its record has nothing to draw, so it is not a session."""
    session.current_path().mkdir(parents=True, exist_ok=True)

    assert client.get("/api/session").json() == {"loaded": False}


def test_loading_another_map_leaves_only_one_session(client, grid_dir, tmp_path):
    """One map at a time is a property of the directory, not a rule to remember."""
    second = synthetic.write_grid(tmp_path / "second")

    for directory in (grid_dir, second):
        assert client.post("/api/map", json={
            "path": str(directory), "rows": synthetic.ROWS, "cols": synthetic.COLS,
        }).status_code == 200
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            job = jobs.current()
            if job is not None and job.state != "running":
                break
            time.sleep(0.1)
        assert job.state == "done", job

    assert client.get("/api/session").json()["label"] == "second"
    assert [entry.name for entry in session.sessions_dir().iterdir()] == [session.CURRENT_NAME]
