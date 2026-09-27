"""The band and the spectra the heatmap screen draws from.

The payload is checked against the matplotlib viewer's own function rather than against
a second implementation of the same arithmetic. ``query_rolling_auc`` is what the viewer
has always used to turn a width in cm⁻¹ into a number of bands, so computing the expected
band with it is the check that the browser and the viewer agree, which is the reason the
serving layer calls it instead of reimplementing it.

Most of these tests load a map by calling the worker in this process. That is the same
code the child process runs, without also spending a process and a poll on reading a four
by four grid. The paths that are about the runner rather than about the frame go through
the server in :mod:`tests.test_web_session`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from app import frames, jobs, session, worker
from src.visualisation.view_heatmap import query_rolling_auc
from tests import synthetic

pytestmark = pytest.mark.usefixtures("isolated_artifacts")

#: The width the design settles on, and the one the screen's own rules are written about.
WIDTH_CM = 10.0


@pytest.fixture(autouse=True)
def fresh_window_cache():
    """Starts every test with an empty window cache.

    ``frames._CACHE`` is process state kept between requests so that moving the position
    slider is a slice of work already done rather than a rebuild. A test that loads its
    own map should not have to reason about what an earlier test left in it.
    """
    frames._CACHE.update(key=None, auc=None, starts=None, ends=None, step=None)


def _read(grid, directory, **fields) -> dict:
    """Runs the load worker in this process and makes its arrays the session.

    Returns:
        dict: The session record, as :func:`app.session.write` wrote it.
    """
    directory.mkdir(parents=True, exist_ok=True)
    request = {
        "kind": "load", "path": str(grid),
        "rows": synthetic.ROWS, "cols": synthetic.COLS, **fields,
    }
    (directory / worker.REQUEST_NAME).write_text(json.dumps(request), encoding="utf-8")
    worker.run(str(directory))

    state = json.loads((directory / worker.STATE_NAME).read_text(encoding="utf-8"))
    assert state["state"] == "done", state
    return session.replace_with(directory / worker.ARRAYS_DIRNAME)


def _brighter(directory, factor: float = 3.0):
    """A grid identical to the fixture's but for the height of every spectrum.

    A second map that differs by a known multiple is what makes it possible to say which
    map a band came from: the areas scale with the intensities, so one map's band is the
    other's times the factor.
    """
    grid = synthetic.write_grid(directory)
    for file in grid.iterdir():
        axis, intensity = np.loadtxt(file, unpack=True)
        np.savetxt(file, np.column_stack([axis, intensity * factor]), delimiter="\t", fmt="%.6f")
    return grid


def _agree_on_load_time(directory, record: dict) -> None:
    """Rewrites a session's record so it carries another session's load time.

    A load time has one-second resolution, so two maps read inside the same second carry
    the same one. Doing that deliberately here is what makes the collision certain to a
    test rather than a matter of how fast the machine is.
    """
    path = directory / session.META_NAME
    written = json.loads(path.read_text(encoding="utf-8"))
    written["loaded_at"] = record["loaded_at"]
    path.write_text(json.dumps(written), encoding="utf-8")


def _wait_for(client, job: dict, seconds: float = 180) -> dict:
    """Polls a job until it stops, and returns what it last reported."""
    deadline = time.monotonic() + seconds
    current = job
    while time.monotonic() < deadline:
        current = client.get(f"/api/job/{job['id']}").json()
        if current["state"] != "running":
            return current
        time.sleep(0.1)
    raise AssertionError(f"job {job['id']} did not finish in {seconds} seconds: {current}")


@pytest.fixture
def loaded(grid_dir, tmp_path):
    """The fixture grid read into the session, with the cube it produced."""
    _read(grid_dir, tmp_path / "job")
    return session.load_cube()


# ------------------------------------------------------------------ the payload


def test_the_band_is_the_one_the_viewer_would_draw(client, loaded):
    """The browser and the matplotlib viewer window the cube the same way, band for band."""
    data = client.get("/api/frame", params={
        "center": 3, "width_cm": WIDTH_CM, "pixel": 5,
    }).json()
    areas, starts, ends, step = query_rolling_auc(loaded.cube, loaded.axis, WIDTH_CM)

    assert (data["rows"], data["cols"]) == (synthetic.ROWS, synthetic.COLS)
    assert data["index"] == 3
    assert data["bands"] == areas.shape[-1]
    assert data["window_bands"] == step
    assert data["width_cm"] == WIDTH_CM
    np.testing.assert_allclose(data["band"], areas[:, :, 3])
    assert data["window_start"] == pytest.approx(float(starts[3]))
    assert data["window_end"] == pytest.approx(float(ends[3]))


def test_the_spectrum_is_the_pixels_own_and_the_numbering_is_the_grids(client, loaded):
    """The spectrum is read by pixel number, and the map says which pixel is where."""
    data = client.get("/api/frame", params={"pixel": 7}).json()

    assert data["pixel"] == 7
    np.testing.assert_allclose(data["spectrum"], np.asarray(loaded.spectra[7]))
    np.testing.assert_allclose(data["axis"], np.asarray(loaded.axis))
    np.testing.assert_array_equal(data["pixel_map"], synthetic.expected_pixel_map())


def test_the_limits_are_the_two_scales_the_intensity_slider_works_in(client, loaded):
    """The image is drawn in plain units, so the log range is reported beside them.

    Areas span orders of magnitude and some of them are negative, so the log range is
    taken over the areas shifted above zero, and the shift is reported rather than folded
    into either range: the screen needs it to turn a log end back into an area.
    """
    data = client.get("/api/frame", params={"width_cm": WIDTH_CM}).json()
    areas, _starts, _ends, _step = query_rolling_auc(loaded.cube, loaded.axis, WIDTH_CM)
    limits = data["limits"]

    assert limits["i_min"] == pytest.approx(float(np.min(areas)))
    assert limits["i_max"] == pytest.approx(float(np.max(areas)))
    assert limits["correction"] == pytest.approx(abs(min(limits["i_min"], 0.0)))
    assert limits["log_min"] == pytest.approx(
        float(np.log(limits["i_min"] + limits["correction"] + 1e-10))
    )
    assert limits["log_max"] == pytest.approx(
        float(np.log(limits["i_max"] + limits["correction"] + 1e-10))
    )


def test_the_window_keeps_its_own_rules(client, loaded):
    """Never narrower than two bands, never wider than half the range."""
    data = client.get("/api/frame").json()

    assert data["mean_step"] == pytest.approx(float(np.mean(np.diff(loaded.axis))))
    assert data["range"] == pytest.approx(float(loaded.axis[-1] - loaded.axis[0]))
    assert data["window_min"] == pytest.approx(2 * data["mean_step"])
    assert data["window_max"] == pytest.approx(data["range"] * 0.5)


def test_a_position_past_the_end_of_the_window_is_clamped(client, loaded):
    """A narrower window leaves fewer bands than a wider one, so the slider can point past it.

    Clamping here rather than refusing means dragging the width slider never leaves the
    screen holding a position that no longer exists.
    """
    data = client.get("/api/frame", params={"center": 10 ** 6, "pixel": 10 ** 6}).json()

    assert data["index"] == data["bands"] - 1
    assert data["pixel"] == synthetic.ROWS * synthetic.COLS


def test_a_window_wider_than_the_range_is_refused_in_the_codes_own_words(client, loaded):
    """The refusal is the viewer's, and the screen shows it as written.

    The way out is to ask for a smaller window, and the message gives both widths, so
    there is nothing for the route to add.
    """
    too_wide = 2 * (loaded.axis[-1] - loaded.axis[0])
    response = client.get("/api/frame", params={"width_cm": too_wide})

    with pytest.raises(ValueError) as raised:
        query_rolling_auc(loaded.cube, loaded.axis, too_wide)

    assert response.status_code == 400
    assert response.json()["detail"] == str(raised.value)


def test_the_frame_needs_a_map(client):
    """Nothing is drawn before something is loaded, and the message says what to do."""
    response = client.get("/api/frame")

    assert response.status_code == 409
    assert "No map is loaded" in response.json()["detail"]


# ------------------------------------------------------------------ the window cache


def test_a_second_map_is_not_drawn_with_the_first_maps_window(client, tmp_path):
    """The window belongs to the map, not to the clock.

    Two maps read inside the same second carry the same load time, and a cache keyed on
    that would hand the second map the first one's bands. The records here are made to
    agree on their load time, so what tells the two apart is the session's own identifier.
    """
    first_record = _read(synthetic.write_grid(tmp_path / "first"), tmp_path / "job-a")
    before = np.asarray(client.get("/api/frame", params={"width_cm": WIDTH_CM}).json()["band"])

    second_record = _read(_brighter(tmp_path / "second"), tmp_path / "job-b")
    _agree_on_load_time(session.current_path(), first_record)

    assert second_record["session_id"] != first_record["session_id"]
    after = np.asarray(client.get("/api/frame", params={"width_cm": WIDTH_CM}).json()["band"])

    np.testing.assert_allclose(after, 3.0 * before, rtol=1e-9)


def test_moving_the_position_reuses_the_window_and_moving_the_width_does_not(client, loaded):
    """The cache is what makes the position slider a slice rather than a rebuild."""
    first = client.get("/api/frame", params={"center": 0, "width_cm": WIDTH_CM}).json()
    cached = frames._CACHE["key"]

    second = client.get("/api/frame", params={"center": 9, "width_cm": WIDTH_CM}).json()
    assert frames._CACHE["key"] == cached
    np.testing.assert_allclose(second["band"], frames._CACHE["auc"][:, :, 9])

    wider = client.get("/api/frame", params={"center": 9, "width_cm": WIDTH_CM * 4}).json()
    assert frames._CACHE["key"] != cached
    assert wider["window_bands"] > first["window_bands"]


# ------------------------------------------------------------------------ derive


def test_deriving_re_reads_the_map_with_a_pipeline_and_a_range(client, loaded):
    """The one control that costs a read, and it replaces the session when it finishes."""
    response = client.post("/api/derive", json={"pipeline_id": 1, "start": 300, "end": 900})
    assert response.status_code == 200, response.text

    finished = _wait_for(client, response.json()["job"])
    assert finished["state"] == "done", finished

    record = client.get("/api/session").json()
    assert record["pipeline_id"] == 1
    assert record["crop_start"] == 300
    assert record["crop_end"] == 900
    assert 300 <= record["axis_start"] and record["axis_end"] <= 900
    assert record["bands"] < synthetic.spectral_axis().size

    # The cropped map is what the heatmap now draws, so its axis is the cropped one.
    data = client.get("/api/frame").json()
    assert data["axis"][0] >= 300
    assert data["axis"][-1] <= 900


def test_a_bound_left_blank_keeps_that_end_of_the_range_whole(client, loaded):
    """A blank bound is not a default number: it is the end of the map's own range."""
    response = client.post("/api/derive", json={"pipeline_id": 0, "start": 400, "end": None})
    assert response.status_code == 200, response.text
    assert _wait_for(client, response.json()["job"])["state"] == "done"

    record = client.get("/api/session").json()
    assert record["crop_start"] == 400
    assert record["crop_end"] is None
    assert record["axis_end"] == pytest.approx(synthetic.spectral_axis()[-1])


@pytest.mark.parametrize(
    "body, fragment",
    [
        ({"pipeline_id": 4}, "Unknown pipeline id: 4."),
        ({"pipeline_id": "four"}, "The request is unusable"),
        ({"start": "green"}, "is not a wavenumber"),
        ({"start": 900, "end": 300}, "The range must start below where it ends"),
    ],
)
def test_a_derive_that_makes_no_sense_starts_no_job(client, loaded, body, fragment):
    """Every refusal here costs nothing, which is why they are decided before the read."""
    response = client.post("/api/derive", json=body)

    assert response.status_code == 400
    assert fragment in response.json()["detail"]
    assert jobs.current() is None


def test_deriving_needs_a_map(client):
    """There is nothing to re-read before something has been read."""
    response = client.post("/api/derive", json={"pipeline_id": 1})

    assert response.status_code == 409
    assert "No map is loaded" in response.json()["detail"]


# -------------------------------------------------------------------------- save


def test_exporting_the_heatmap_writes_the_state_on_the_screen(client, loaded, isolated_artifacts):
    """The file is built from the same payload the screen draws from, so the two agree."""
    response = client.post("/api/save", json={
        "kind": "heatmap", "center": 2, "width_cm": WIDTH_CM, "pixel": 4,
    })
    assert response.status_code == 200, response.text

    written = response.json()["path"]
    assert Path(written).is_file()
    assert Path(written).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    # Written under the artifacts directory this test was given, not the checkout's.
    assert Path(written).is_relative_to(isolated_artifacts)


def test_a_window_the_frame_refuses_is_refused_on_export_too(client, loaded):
    """A file that could not be drawn would otherwise be a figure of nothing."""
    response = client.post("/api/save", json={"kind": "heatmap", "width_cm": 10 ** 4})

    assert response.status_code == 400
    assert "exceeds the spectral range" in response.json()["detail"]


def test_the_export_is_drawn_in_the_range_the_screen_hands_it(client, loaded, isolated_artifacts):
    """The intensity range is part of what the screen is showing, so it reaches the file.

    A file drawn to the whole range the data spans would be a different picture from the
    one on screen, and the two would disagree about which pixel is the brightest.
    """
    whole = Path(client.post("/api/save", json={"kind": "heatmap"}).json()["path"])
    narrow = Path(client.post("/api/save", json={
        "kind": "heatmap", "i_min": 0.5, "i_max": 2.0, "name": "narrow.png",
    }).json()["path"])

    assert narrow.is_file()
    assert narrow.read_bytes() != whole.read_bytes()

    # Handing it the data's own range draws exactly what leaving the range out draws.
    limits = client.get("/api/frame").json()["limits"]
    same = Path(client.post("/api/save", json={
        "kind": "heatmap", "i_min": limits["i_min"], "i_max": limits["i_max"], "name": "same.png",
    }).json()["path"])

    assert same.read_bytes() == whole.read_bytes()


def test_an_intensity_range_that_is_not_a_pair_of_numbers_is_refused(client, loaded):
    """Reachable from the screen only if the screen is wrong, and it says so either way."""
    response = client.post("/api/save", json={"kind": "heatmap", "i_min": "bright"})

    assert response.status_code == 400
    assert "not a pair of numbers" in response.json()["detail"]


def test_exporting_something_that_has_no_export_is_refused(client, loaded):
    """The unmixing and prediction figures belong to the jobs that produce them."""
    response = client.post("/api/save", json={"kind": "unmix"})

    assert response.status_code == 400
    assert "There is nothing to save for 'unmix'." in response.json()["detail"]
