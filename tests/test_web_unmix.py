"""The unmixing the browser asks for.

The decomposition is checked against the command line's own path rather than against a
second reading of the same arithmetic. ``show_spectra_ext_viewer`` is what the command
line has always run, so decomposing the fixture cube twice, once through it and once
through the browser's code, is what says the two unmix the same way. It is also why the
browser's code calls the framework's NMF with ``NmfConfig`` rather than building one.

Most of these run the unmix job in this process, which is the same code the child process
runs without spending a process and a poll on a four by four grid. The paths that are
about the route, the runner and the figures going back to the page go through the server.
"""

from __future__ import annotations

import json
import time
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from app import figures, jobs, session, unmix, worker
from src.analysis.phase_number_est import MethodEstimate, PhaseEstimate, estimate_phase_number
from src.config import EstimatorConfig, NmfConfig
from src.data.loader import get_cumulative_hsi_cube, get_raw_hsi_cube
from src.visualisation import view_unmixing
from tests import synthetic

pytestmark = pytest.mark.usefixtures("isolated_artifacts")

#: How many phases the fixture grid holds. Used where the test is not about the estimator,
#: so that a decomposition runs rather than waiting on a count.
PHASES = synthetic.N_PHASES


def _unmix(grid, directory, **fields) -> dict:
    """Runs the unmix job in this process and returns the state it finished with.

    Returns:
        dict: The job's state file, with the ``result`` it wrote when it finished.
    """
    directory.mkdir(parents=True, exist_ok=True)
    request = {
        "kind": "unmix", "path": str(grid),
        "rows": synthetic.ROWS, "cols": synthetic.COLS,
        "count": PHASES, **fields,
    }
    (directory / worker.REQUEST_NAME).write_text(json.dumps(request), encoding="utf-8")
    worker.run(str(directory))
    return json.loads((directory / worker.STATE_NAME).read_text(encoding="utf-8"))


def _wait_for(client, job: dict, seconds: float = 300) -> dict:
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
    """The fixture grid as the session the routes work on, written in this process.

    The unmix job reads the map files itself rather than the session, so what a route
    needs is a session record naming a directory that is really a grid.
    """
    cube = get_cumulative_hsi_cube(str(grid_dir), synthetic.ROWS, synthetic.COLS, 0)
    directory = tmp_path / "session"
    session.write(directory, cube, {
        "directory": str(grid_dir), "label": grid_dir.name,
        "rows": synthetic.ROWS, "cols": synthetic.COLS,
        "files": synthetic.ROWS * synthetic.COLS,
        "bands": int(cube.axis.shape[0]),
        "axis_start": float(cube.axis[0]), "axis_end": float(cube.axis[-1]),
        "pipeline_id": 0, "crop_start": None, "crop_end": None,
    })
    session.replace_with(directory)
    return session.metadata()


# ------------------------------------------------------------------- the settings


def test_the_settings_the_screen_shows_are_the_ones_the_config_holds(client):
    """The card is the config read at the moment it is asked for, not a copy of it.

    Nothing is loaded here on purpose: what a decomposition runs with is a property of
    the framework, not of the map, so the screen can show it before a map is opened.
    """
    assert client.get("/api/session").json()["loaded"] is False

    body = client.get("/api/unmix/settings").json()

    config = NmfConfig()
    assert body["settings"] == {field.name: getattr(config, field.name) for field in fields(config)}
    assert body["min_votes"] == EstimatorConfig().min_votes
    assert body["count_min"] == unmix.COUNT_MIN


def test_the_pipeline_the_screen_names_is_the_one_the_job_builds():
    """The card says which steps run, so the names and the steps have to be the same list."""
    steps = [type(step).__name__ for step in unmix.pipeline().pipeline]

    assert tuple(steps) == unmix.PIPELINE_STEPS
    assert steps == ["WhitakerHayes", "ASLS", "MinMax"], "in the order they are applied"


# -------------------------------------------------------------- the decomposition


def test_the_decomposition_is_the_viewers_own(grid_dir, raw_cube, monkeypatch):
    """The browser unmixes the cube the way the command line's viewer does.

    The viewer is run with its plotting replaced and with the maps and the spectra it
    hands to those plots recorded, which is the only way to see what it decomposed to
    without a window. The decomposition is deterministic, so the two agree exactly rather
    than nearly: the initialisation is computed from the data, not drawn.
    """
    seen = {}

    def record_spectra(spectra, axis, **kwargs):
        seen["phases"] = [np.asarray(phase) for phase in spectra]
        seen["axis"] = np.asarray(axis)
        return None

    def record_image(image, **kwargs):
        seen.setdefault("maps", []).append(np.asarray(image))
        return kwargs.get("ax")

    monkeypatch.setattr(view_unmixing.rp.plot, "spectra", record_spectra)
    monkeypatch.setattr(view_unmixing.rp.plot, "image", record_image)
    monkeypatch.setattr(view_unmixing.plt, "show", lambda: None)

    view_unmixing.show_spectra_ext_viewer(raw_cube, PHASES)
    maps, phases = unmix.decomposition(get_raw_hsi_cube(str(grid_dir), synthetic.ROWS, synthetic.COLS), PHASES)

    assert len(seen["maps"]) == PHASES
    for theirs, ours in zip(seen["maps"], maps):
        np.testing.assert_allclose(ours, theirs)
    for theirs, ours in zip(seen["phases"], phases):
        np.testing.assert_allclose(ours, theirs)


def test_a_fixed_count_is_used_as_given(grid_dir, tmp_path, monkeypatch):
    """A count the screen fixes is not second-guessed, which is what the card promises."""
    def explosion(cube):
        raise AssertionError("the estimator was called for a cube whose count was given")

    monkeypatch.setattr(unmix, "estimate_phase_number", explosion)

    result = _unmix(grid_dir, tmp_path / "job", count=3)["result"]

    assert result["count"] == 3
    assert result["estimate"] is None, "a fixed count has no estimate to report"
    assert [entry["component"] for entry in result["maps"]] == [1, 2, 3]


def test_estimating_the_count_is_the_estimators_own_findings(grid_dir, tmp_path):
    """What the report card draws is what the estimator returned, method for method."""
    result = _unmix(grid_dir, tmp_path / "job", count=None)["result"]
    expected = estimate_phase_number(get_raw_hsi_cube(str(grid_dir), synthetic.ROWS, synthetic.COLS))

    assert result["unmixed"] is True
    assert result["count"] == expected.n
    assert result["estimate"]["confidence"] == expected.confidence
    assert [vote["name"] for vote in result["estimate"]["votes"]] == [vote.name for vote in expected.votes]
    assert result["estimate"]["methods"] == len(expected.votes)


def test_an_undetermined_estimate_unmixes_nothing(grid_dir, tmp_path, monkeypatch):
    """No count means no decomposition, and the report is the whole result.

    This is a finished job rather than a failed one: the estimator's findings are the
    answer. Decomposing anyway would need a count, and the only counts available are the
    ones the rule already threw away.
    """
    monkeypatch.setattr(unmix, "estimate_phase_number", lambda cube: PhaseEstimate(
        None, "undetermined",
        (MethodEstimate("PCA 80%", None, "the method produced no estimate"),
         MethodEstimate("PCA elbow", 17, None)),
    ))

    state = _unmix(grid_dir, tmp_path / "job", count=None)

    assert state["state"] == "done", state
    assert state["result"]["unmixed"] is False
    assert state["result"]["count"] is None
    assert state["result"]["estimate"]["n"] is None
    assert not list((tmp_path / "job").glob("*.png")), "there is nothing to draw without a count"


def test_the_report_lists_every_method_and_why_it_was_rejected():
    """A method thrown out keeps its reason, since that is the row worth reading.

    A reader shown only the number that survived cannot tell a method that agreed from one
    that was discarded for disagreeing, and a rejection written over with a number is how a
    discarded guess ends up indistinguishable from a count.
    """
    report = unmix.report(PhaseEstimate(None, "undetermined", (
        MethodEstimate("PCA 80%", None, "the estimate of 12 is above the limit of 10"),
        MethodEstimate("PCA elbow", 2),
        MethodEstimate("NW-HFC", None, "there was no PCA estimate to compare against"),
    )))

    assert [vote["name"] for vote in report["votes"]] == ["PCA 80%", "PCA elbow", "NW-HFC"]
    assert report["votes"][0]["value"] is None
    assert report["votes"][0]["rejected"] == "the estimate of 12 is above the limit of 10"
    assert report["votes"][0]["counted"] is False
    assert report["votes"][1] == {"name": "PCA elbow", "value": 2, "rejected": None, "counted": True}
    assert report["counted"] == 1
    assert report["methods"] == 3
    assert report["min_votes"] == EstimatorConfig().min_votes


# --------------------------------------------------------------------- the figures


def test_the_job_writes_a_spectra_figure_and_one_map_per_component(grid_dir, tmp_path):
    """One file per panel, so the page lays the maps out in its own grid."""
    result = _unmix(grid_dir, tmp_path / "job", count=3)["result"]

    assert result["spectra"] == figures.ENDMEMBER_NAME
    assert [entry["figure"] for entry in result["maps"]] == [
        figures.ABUNDANCE_NAME.format(component=component) for component in (1, 2, 3)
    ]

    for name in [result["spectra"], *[entry["figure"] for entry in result["maps"]]]:
        written = tmp_path / "job" / name
        assert written.is_file(), f"{name} was not written"
        assert written.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_the_range_crops_before_decomposing(grid_dir, tmp_path):
    """The screen's range narrows this decomposition and leaves the loaded map alone."""
    result = _unmix(grid_dir, tmp_path / "job", count=2, start=400, end=900)["result"]

    assert result["axis_start"] >= 400
    assert result["axis_end"] <= 900
    assert result["bands"] < synthetic.spectral_axis().size


def test_a_bound_left_blank_keeps_that_end_of_the_range_whole(grid_dir, tmp_path):
    """A blank bound is the end of the map's own range, not a default number."""
    result = _unmix(grid_dir, tmp_path / "job", count=2, start=500, end=None)["result"]

    assert result["axis_start"] >= 500
    assert result["axis_end"] == pytest.approx(synthetic.spectral_axis()[-1])


# ----------------------------------------------------------------------- the route


def test_an_unmixing_needs_a_map(client):
    """There is nothing to unmix before something is loaded."""
    response = client.post("/api/unmix", json={"estimate": True})

    assert response.status_code == 409
    assert "No map is loaded" in response.json()["detail"]


@pytest.mark.parametrize(
    "body, fragment",
    [
        ({"estimate": False, "count": "four"}, "has to be a whole number"),
        ({"estimate": False, "count": 0}, "at least 1 component"),
        ({"estimate": False, "count": None}, "has to be a whole number"),
        ({"start": "green"}, "is not a wavenumber"),
        ({"start": 900, "end": 300}, "The range must start below where it ends"),
    ],
)
def test_an_unmixing_that_makes_no_sense_starts_no_job(client, loaded, body, fragment):
    """Every refusal here costs nothing, which is why they are decided before the run."""
    response = client.post("/api/unmix", json=body)

    assert response.status_code == 400
    assert fragment in response.json()["detail"]
    assert jobs.current() is None


def test_the_route_starts_an_unmixing_whose_figures_the_page_can_fetch(client, loaded):
    """The whole path in one go: a job, its result, and the panels it drew.

    The job runs in a child process here, so this is also the check that the request the
    route writes is one the worker can run.
    """
    response = client.post("/api/unmix", json={"estimate": False, "count": 2})
    assert response.status_code == 200, response.text

    started = response.json()["job"]
    assert started["kind"] == "unmix"

    finished = _wait_for(client, started)
    assert finished["state"] == "done", finished
    result = finished["result"]
    assert result["count"] == 2

    response = client.get(f"/api/job/{started['id']}/figure/{result['spectra']}")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content[:8] == b"\x89PNG\r\n\x1a\n"

    for entry in result["maps"]:
        assert client.get(f"/api/job/{started['id']}/figure/{entry['figure']}").status_code == 200


@pytest.mark.parametrize(
    "name",
    ["state.json", "traceback.txt", ".hidden.png", "abundance-99.png", "job.json"],
)
def test_a_name_that_is_not_a_figure_this_job_wrote_is_not_served(client, loaded, name, tmp_path):
    """The name arrives from the browser, so what it may be is fixed before it is read.

    A job's directory also holds its request and its state, so serving whatever was asked
    for would hand those to anything that could guess a name.
    """
    jobs._CURRENT = jobs.Job(
        id="finished", kind="unmix", request={}, directory=tmp_path, state="done",
    )
    (tmp_path / "state.json").write_text("{}", encoding="utf-8")

    response = client.get(f"/api/job/finished/figure/{name}")

    assert response.status_code == 404
    assert "No figure is called" in response.json()["detail"]


def test_a_figure_belongs_to_the_job_that_drew_it(client):
    """An identifier from a run the server no longer holds answers nothing."""
    response = client.get("/api/job/nolongerthejob/figure/endmembers.png")

    assert response.status_code == 404
    assert "no longer the current one" in response.json()["detail"]


def test_the_panels_of_a_job_are_written_where_the_figure_route_looks(grid_dir, tmp_path):
    """The names a result carries and the directory the route reads are the same pair.

    The job's own directory is where its figures go, which is what lets the route refuse a
    name that was never written there rather than reading one from somewhere else.
    """
    directory = tmp_path / "job"
    result = _unmix(grid_dir, directory, count=2)["result"]

    names = [result["spectra"], *[entry["figure"] for entry in result["maps"]]]
    assert sorted(Path(directory).glob("*.png")) == sorted(directory / name for name in names)
