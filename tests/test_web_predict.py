"""The prediction the browser asks for.

The prediction itself is the framework's own call, so what is checked here is everything
around it: that the picker lists what the framework lists, that a run which cannot answer
is refused as a result rather than as a failure, and that the threshold slider redraws
without running the model again.

The colouring and the ordering of the minerals are checked against the matplotlib viewer
rather than against a second reading of the same rule. ``show_prediction_map`` is what the
command line has always drawn, so running it with its prediction replaced and reading back
the image it made is what says the browser draws the same picture. Most of these run the
predict job in this process, which is the code the child process runs without spending a
process, a model load and a poll on it; the paths that are about the runner go through the
server.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import keras
import numpy as np
import pytest
from sklearn.preprocessing import LabelEncoder

from app import jobs, predictions, session, worker
from src.config import TrainConfig
from src.cnn import runs
from src.data.loader import get_cumulative_hsi_cube
from src.visualisation import view_predict
from tests import synthetic

pytestmark = pytest.mark.usefixtures("isolated_artifacts")

#: The minerals the stand-in model knows, which is fewer than a real run's list and is
#: what makes the legend shorter than the class list worth asserting.
LABELS = ["Calcite", "Pyrite", "Quartz"]

#: The threshold the command line starts at, and so the one the screen starts at.
THRESHOLD = 0.80


def _model_for(axis, num_classes=len(LABELS)):
    """A model whose only acceptable input is a spectrum of the fixture's length."""
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(axis.size, 1)),
            keras.layers.Flatten(),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="categorical_crossentropy")
    return model


def _trained_run(axis, dataset, name="fixture", **overrides):
    """A run written where the catalogue looks for runs, trained on the fixture's axis.

    Written at the framework's own location rather than into a directory of this test's
    choosing, because the picker is the thing under test: a run it would not find is not
    a run the screen could offer.
    """
    config = replace(
        TrainConfig(
            dataset=str(dataset),
            input_size=axis.size,
            x_min=float(np.min(axis)),
            x_max=float(np.max(axis)),
        ),
        **overrides,
    )
    directory = runs.run_directory(name)
    runs.save_run(directory, config=config, model=_model_for(axis), encoder=LabelEncoder().fit(LABELS))
    return directory


def _predict_job(grid, run_path, directory, **fields) -> dict:
    """Runs the predict job in this process and returns the state it finished with."""
    directory.mkdir(parents=True, exist_ok=True)
    request = {
        "kind": "predict", "run_path": str(run_path), "path": str(grid),
        "rows": synthetic.ROWS, "cols": synthetic.COLS, **fields,
    }
    (directory / worker.REQUEST_NAME).write_text(json.dumps(request), encoding="utf-8")
    worker.run(str(directory))
    return json.loads((directory / worker.STATE_NAME).read_text(encoding="utf-8"))


def _as_current(directory, result: dict) -> jobs.Job:
    """Makes a job the server is holding, as it is once a prediction has finished."""
    job = jobs.Job(id="predicted", kind="predict", request={}, directory=Path(directory),
                   state="done", result=result)
    jobs._CURRENT = job
    return job


def _wait_for(client, job: dict, seconds: float = 600) -> dict:
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
def axis():
    return synthetic.spectral_axis()


@pytest.fixture
def trained_run(grid_dir, axis):
    """The fixture grid's own run, the one the happy path predicts with."""
    return _trained_run(axis, grid_dir)


@pytest.fixture
def loaded(grid_dir, tmp_path):
    """The fixture grid as the session the predict route works on."""
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


@pytest.fixture
def predicted(grid_dir, trained_run, tmp_path):
    """A finished prediction, made the way the job makes one, and held by the runner."""
    state = _predict_job(grid_dir, trained_run, tmp_path / "job")
    assert state["state"] == "done", state
    _as_current(tmp_path / "job", state["result"])
    return state["result"]


@pytest.fixture
def refused(grid_dir, axis, tmp_path):
    """A finished prediction that refused its run, held by the runner."""
    wide = _trained_run(axis, grid_dir, name="wide", x_max=float(np.max(axis)) + 200.0)
    state = _predict_job(grid_dir, wide, tmp_path / "refused")
    assert state["result"]["refused"] is True, state
    _as_current(tmp_path / "refused", state["result"])
    return state["result"]


# ------------------------------------------------------------------ the picker


def test_the_picker_lists_the_runs_the_framework_lists(client, grid_dir, trained_run, axis):
    """The run card is what the manifest recorded, read without loading the model."""
    body = client.get("/api/runs").json()

    assert body["directory"] == str(runs.runs_root())
    assert [entry["name"] for entry in body["runs"]] == [trained_run.name]

    entry = body["runs"][0]
    assert entry["classes"] == len(LABELS)
    assert entry["directory"] == str(trained_run)
    assert [row["label"] for row in entry["facts"]] == [
        label for label, _ in predictions.FACT_READERS
    ]

    recorded = {row["label"]: row["value"] for row in entry["facts"]}
    assert recorded["input"] == f"{axis.size} points per spectrum"
    assert recorded["classes"] == f"{len(LABELS)} minerals"
    assert recorded["seed"] == str(TrainConfig().seed)
    assert recorded["range"] == f"{np.min(axis):.1f} – {np.max(axis):.1f} cm⁻¹"
    # The dataset reads as its name with enough of the digest to tell two of that name
    # apart, rather than as the path the command line happened to be run from.
    assert recorded["dataset"].startswith(f"{grid_dir.name} · ")
    assert recorded["dataset"].endswith("…")


def test_an_empty_picker_names_the_directory_it_looked_in(client):
    """A page with no runs on it says where it decided that, rather than only that."""
    body = client.get("/api/runs").json()

    assert body["runs"] == []
    assert body["directory"] == str(runs.runs_root())
    assert not Path(body["directory"]).exists()


def test_every_fact_row_is_present_even_when_the_run_recorded_nothing(trained_run):
    """A manifest written before a field existed still loads, and the row says so.

    A row left out would be indistinguishable from a row that was there and empty, and
    the difference matters: one run never recorded a seed, the other was trained without
    one.
    """
    absent = predictions.facts(None)

    assert [row["label"] for row in absent] == [label for label, _ in predictions.FACT_READERS]
    assert all(row["value"] is None for row in absent)
    assert [row["label"] for row in predictions.facts(runs.load_run(trained_run).manifest)] == [
        row["label"] for row in absent
    ]


def test_a_metrics_row_is_absent_when_the_run_scored_nothing(trained_run):
    """The stand-in model is never evaluated on held-out spectra, so there is no row to show."""
    facts = {row["label"]: row["value"] for row in predictions.facts(runs.load_run(trained_run).manifest)}

    assert facts["metrics"] is None


# ------------------------------------------------------------------- the refusal


@pytest.mark.parametrize("name", ["nowhere", "/tmp/not-a-run"])
def test_something_that_is_not_a_run_directory_is_refused(client, loaded, name):
    """A directory with no manifest in it is not a run, and is turned away as one."""
    response = client.post("/api/predict", json={"run": name})

    assert response.status_code == 404
    assert "holds no run.json, so it is not a run directory" in response.json()["detail"]
    assert jobs.current() is None


def test_a_prediction_needs_a_map(client, trained_run):
    """There is nothing to predict over before a map is loaded."""
    response = client.post("/api/predict", json={"run": trained_run.name})

    assert response.status_code == 409
    assert "No map is loaded" in response.json()["detail"]


def test_a_refusal_is_the_jobs_own_result(grid_dir, axis, tmp_path):
    """A run whose training range the map does not cover is a finished job, not a failure.

    ``predict`` refuses rather than warns because the model would answer anyway, and the
    answer is the reason it is not shown. Reporting it as an error would put a traceback
    on the screen where the guard's own sentence belongs.
    """
    wide = _trained_run(axis, grid_dir, name="wide", x_max=float(np.max(axis)) + 200.0)

    state = _predict_job(grid_dir, wide, tmp_path / "job")

    assert state["state"] == "done"
    assert state["result"]["refused"] is True
    assert "pad the spectrum with zeros" in state["result"]["reason"]
    assert state["result"]["run"] == "wide"
    assert not list((tmp_path / "job").glob("*.png"))
    assert not (tmp_path / "job" / predictions.PREDICTIONS_NAME).exists()


def test_a_refused_prediction_cannot_be_redrawn(client, refused):
    """A refusal leaves no pixels behind, and the slider says so rather than drawing one."""
    response = client.post("/api/predict/render", json={"threshold": THRESHOLD})

    assert response.status_code == 409
    assert "refused its run" in response.json()["detail"]


def test_a_refused_prediction_has_no_table(client, refused):
    """The page asks for the per-pixel table and is told there are none."""
    response = client.get(f"/api/job/{jobs.current().id}/predictions")

    assert response.status_code == 404
    assert "refused its run" in response.json()["detail"]


# ------------------------------------------------------------------ the job


def test_the_job_stores_the_predictions_and_draws_the_map(grid_dir, trained_run, tmp_path):
    """What the page reads back is the job's own file, written beside its figure."""
    directory = tmp_path / "job"
    result = _predict_job(grid_dir, trained_run, directory)["result"]

    assert result["refused"] is False
    assert (result["rows"], result["cols"]) == (synthetic.ROWS, synthetic.COLS)
    assert result["figure"] == predictions.MAP_NAME
    assert result["threshold"] == predictions.DEFAULT_THRESHOLD
    assert (directory / predictions.MAP_NAME).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    stored = predictions.load_predictions(directory)
    assert stored["run"] == trained_run.name
    assert np.array(stored["labels"]).shape == (synthetic.ROWS, synthetic.COLS)
    assert set(np.ravel(stored["labels"])) <= set(LABELS)
    assert np.array(stored["top5"], dtype=object).shape == (synthetic.ROWS, synthetic.COLS, len(LABELS), 2)


def test_the_route_predicts_the_loaded_map_that_the_page_can_fetch(client, loaded, trained_run):
    """The whole path in one go: a job, its result, and the map the page asks for.

    The job runs in a child process here, so this is also the check that the request the
    route writes is one the worker can run, with the run named as a path the child can
    find on its own.
    """
    response = client.post("/api/predict", json={"run": trained_run.name})
    assert response.status_code == 200, response.text

    started = response.json()
    assert started["run"] == trained_run.name
    assert started["job"]["kind"] == "predict"

    finished = _wait_for(client, started["job"])
    assert finished["state"] == "done", finished
    result = finished["result"]
    assert result["refused"] is False

    figure = client.get(f"/api/job/{started['job']['id']}/figure/{result['figure']}")
    assert figure.status_code == 200
    assert figure.content[:8] == b"\x89PNG\r\n\x1a\n"

    table = client.get(f"/api/job/{started['job']['id']}/predictions")
    assert table.status_code == 200
    assert np.array(table.json()["top5"], dtype=object).shape[:2] == (synthetic.ROWS, synthetic.COLS)


def test_a_prediction_can_be_cancelled(client, loaded, trained_run):
    """A prediction is cancellable for the same reason an unmixing is: it is a child process.

    The job is stopped from here rather than waited on, so what this checks is that a
    running prediction is one the runner can terminate, which is what the button does.
    """
    started = client.post("/api/predict", json={"run": trained_run.name}).json()["job"]

    response = client.post(f"/api/job/{started['id']}/cancel")

    assert response.status_code == 200
    assert response.json()["state"] in {"cancelled", "done"}


def test_the_table_belongs_to_the_prediction_that_wrote_it(client, predicted):
    """A job's table is read from that job's directory, by the identifier it was given."""
    body = client.get(f"/api/job/{jobs.current().id}/predictions").json()

    assert body["run"] == predicted["run"]
    assert len(body["labels"]) == synthetic.ROWS
    assert all(len(row) == synthetic.COLS for row in body["labels"])
    for rank in range(len(LABELS)):
        label, probability = body["top5"][0][0][rank]
        assert label in LABELS
        assert 0.0 <= probability <= 1.0


def test_a_job_that_is_not_a_prediction_has_no_table(client, predicted, tmp_path):
    """A table is the prediction's own file, so another kind of job has nothing to serve."""
    jobs._CURRENT = jobs.Job(id="unmixed", kind="unmix", request={}, directory=tmp_path, state="done")

    response = client.get("/api/job/unmixed/predictions")

    assert response.status_code == 404
    assert "has no predictions" in response.json()["detail"]


def test_a_table_from_a_job_the_server_no_longer_holds_is_not_served(client, predicted):
    """The identifier of a job that has been replaced answers nothing."""
    response = client.get("/api/job/nolongerthejob/predictions")

    assert response.status_code == 404
    assert "no longer the current one" in response.json()["detail"]


# ------------------------------------------------------------------ the threshold


def test_the_threshold_masks_the_pixels_it_is_not_confident_about():
    """Below the threshold a pixel reads Unknown, and that is the whole effect of the slider."""
    labels = np.array([["Quartz", "Calcite"], ["Pyrite", "Quartz"]], dtype=object)
    top5 = np.array([
        [[("Quartz", 0.90), ("Calcite", 0.05)], [("Calcite", 0.40), ("Quartz", 0.35)]],
        [[("Pyrite", 0.70), ("Quartz", 0.20)], [("Quartz", 0.99), ("Calcite", 0.01)]],
    ], dtype=object)

    drawn = predictions.apply_threshold(labels, top5, THRESHOLD)

    assert list(drawn["labels"][0]) == ["Quartz", predictions.UNKNOWN]
    assert list(drawn["labels"][1]) == [predictions.UNKNOWN, "Quartz"]
    assert drawn["minerals"] == ["Quartz", predictions.UNKNOWN], "Unknown is last, whatever it outranks"
    assert drawn["index"][0, 0] == drawn["index"][1, 1] == 0


def test_the_drawing_is_the_viewers_own_rule(monkeypatch):
    """The browser colours and orders the minerals the way the matplotlib viewer does.

    The viewer is run with its prediction replaced, which is what makes it drawable without
    a model, and the image it made is read back: the same integer per pixel and the same
    colour per integer is what says a browser run and a command line run draw one picture.
    """
    labels = np.array([["Quartz", "Calcite"], ["Pyrite", "Quartz"]], dtype=object)
    top5 = np.array([
        [[("Quartz", 0.90), ("Calcite", 0.05)], [("Calcite", 0.40), ("Quartz", 0.35)]],
        [[("Pyrite", 0.70), ("Quartz", 0.20)], [("Quartz", 0.99), ("Calcite", 0.01)]],
    ], dtype=object)
    monkeypatch.setattr(view_predict, "predict", lambda *a, **k: (labels, top5))
    monkeypatch.setattr(view_predict.plt, "show", lambda: None)

    viewed = view_predict.show_prediction_map("run", "path", 2, 2, confidence_threshold=THRESHOLD)
    image = view_predict.plt.gcf().axes[0].images[0]
    their_index = np.asarray(image.get_array())
    their_colours = np.asarray(image.cmap.colors)
    view_predict.plt.close("all")

    drawn = predictions.apply_threshold(labels, top5, THRESHOLD)

    assert viewed is None
    np.testing.assert_array_equal(drawn["index"], their_index)
    np.testing.assert_allclose(np.asarray(drawn["colours"]), their_colours)

    # The pixel that reads Unknown is the grey the heatmap paints an out-of-range area in,
    # rather than another colour from the map.
    assert tuple(their_colours[-1]) == pytest.approx(predictions.UNKNOWN_COLOUR)


def test_the_legend_lists_only_the_minerals_the_map_holds(client, predicted):
    """The legend is the set on this map, so it is shorter than the model's class list."""
    minerals = [row["name"] for row in predicted["minerals"]]

    assert len(minerals) < len(LABELS), "a map that holds every class cannot show the difference"
    assert set(minerals) <= set(LABELS) | {predictions.UNKNOWN}
    assert all(row["colour"].startswith("#") for row in predicted["minerals"])


def test_moving_the_threshold_redraws_without_running_the_model(client, predicted):
    """Nothing predicts here: the slider decides what is drawn, from what was computed once.

    A redraw that reached for the model would take a minute to answer a slider drag, so
    what this checks is that the two arrays the job stored are the whole input to it: the
    threshold moves, the predictions themselves do not, and the map changes anyway.

    The two thresholds are a low one and a high one rather than the default and a higher
    one, because the stand-in model is unsure about every pixel: its confidences sit near
    the middle of the range, so a threshold moved above 0.80 would redraw the same grey
    map and this would pass without the redraw having done anything.
    """
    stored = (Path(jobs.current().directory) / predictions.PREDICTIONS_NAME).read_bytes()
    before = client.post("/api/predict/render", json={"threshold": 0.01}).json()
    figure = Path(jobs.current().directory) / before["figure"]
    low = figure.read_bytes()

    after = client.post("/api/predict/render", json={"threshold": 0.999}).json()

    assert after["figure"] == before["figure"], "the page asks for the map under the name it knows"
    assert after["threshold"] == 0.999
    assert len(after["minerals"]) < len(before["minerals"])
    assert low != figure.read_bytes(), "a threshold that changes nothing would not show this"
    assert (Path(jobs.current().directory) / predictions.PREDICTIONS_NAME).read_bytes() == stored


@pytest.mark.parametrize("value", [0, 1.5, -0.5, "soon", None, ""])
def test_a_threshold_that_is_not_a_confidence_is_refused(client, predicted, value):
    """The slider sends a confidence, and a value outside it would decide nothing."""
    response = client.post("/api/predict/render", json={"threshold": value})

    assert response.status_code == 400
    assert "threshold" in response.json()["detail"]


def test_a_redraw_before_a_prediction_is_refused(client):
    """There is nothing to redraw until a prediction has run."""
    response = client.post("/api/predict/render", json={"threshold": THRESHOLD})

    assert response.status_code == 409
    assert "No prediction has been run yet" in response.json()["detail"]


def test_a_redraw_while_the_prediction_is_still_going_is_refused(client, predicted, tmp_path):
    """A running prediction has no map to redraw yet, so the slider waits rather than guesses."""
    jobs._CURRENT.state = "running"

    response = client.post("/api/predict/render", json={"threshold": THRESHOLD})

    assert response.status_code == 409
    assert "No prediction has been run yet" in response.json()["detail"]


# ------------------------------------------------------------------ the export


def test_saving_the_prediction_map_writes_the_command_lines_own_file(client, predicted):
    """A file under ``outputs`` is read on its own, so it carries the title and the legend.

    The page's own copy deliberately carries neither, since the slider and the legend beside
    it say both. The saved file is drawn again for that reason, rather than copied.
    """
    body = client.post("/api/save", json={
        "kind": "predict", "threshold": THRESHOLD, "name": predictions.EXPORT_NAME,
    }).json()

    written = Path(body["path"])
    assert written.name == predictions.EXPORT_NAME
    assert written.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert written.read_bytes() != (Path(jobs.current().directory) / predicted["figure"]).read_bytes()


def test_the_export_names_the_threshold_it_was_drawn_at(client, predicted):
    """A file read away from the screen cannot be asked which threshold it shows.

    The exported copy carries the threshold in its title for that reason, which is why two
    exports at two thresholds are two files rather than one copied twice.
    """
    low = Path(client.post("/api/save", json={
        "kind": "predict", "threshold": 0.01, "name": "low.png",
    }).json()["path"])
    high = Path(client.post("/api/save", json={
        "kind": "predict", "threshold": 0.999, "name": "high.png",
    }).json()["path"])

    assert low.is_file() and high.is_file()
    assert low.read_bytes() != high.read_bytes()


def test_saving_a_prediction_that_refused_is_refused(client, refused):
    """There are no pixels to draw, and the file is not written as a blank map instead."""
    response = client.post("/api/save", json={"kind": "predict", "threshold": THRESHOLD})

    assert response.status_code == 409
    assert "refused its run" in response.json()["detail"]


def test_saving_a_prediction_needs_a_prediction(client):
    """The Predict screen's Save is refused before anything has been predicted."""
    response = client.post("/api/save", json={"kind": "predict", "threshold": THRESHOLD})

    assert response.status_code == 409
    assert "No prediction has been run yet" in response.json()["detail"]


@pytest.mark.parametrize("name", ["escaped.txt", ".escaped.png", "escaped", "  ", "a/b"])
def test_a_name_that_is_not_a_figure_name_is_refused(client, loaded, name):
    """The name reaches the filesystem, so its shape is fixed before it is written."""
    response = client.post("/api/save", json={"kind": "heatmap", "name": name})

    assert response.status_code == 400
    assert "not a figure name" in response.json()["detail"]


@pytest.mark.parametrize("name", ["../../escaped.png", "/etc/escaped.png"])
def test_a_name_that_climbs_out_of_the_outputs_directory_is_written_under_it(client, loaded, name):
    """What a name may not do is decide where the file goes.

    A path that climbs is reduced to the name it was climbing towards, and that name is
    written under ``outputs`` like any other. Refusing it outright would be worse for the
    person typing it: the file they asked for is still the file they get.
    """
    response = client.post("/api/save", json={"kind": "heatmap", "name": name})

    assert response.status_code == 200, response.text
    written = Path(response.json()["path"])
    assert written.name == "escaped.png"
    assert written.parent.name == "outputs"
    assert written.is_file()


def test_a_name_with_the_outputs_directory_on_its_front_is_written_under_it(client, loaded):
    """The screen offers the whole path the command line prints, so it is reduced to its name."""
    whole = str(Path(predictions.EXPORT_NAME).parent / predictions.EXPORT_NAME)

    body = client.post("/api/save", json={"kind": "heatmap", "name": whole}).json()

    assert Path(body["path"]).name == predictions.EXPORT_NAME
