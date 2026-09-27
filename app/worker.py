"""One job, run in its own process.

Started by :mod:`app.jobs` with a directory holding a ``job.json`` request. It
writes ``state.json`` as it moves through its steps so the server can report where
it is, and it writes its output beside that. Nothing here talks to the server, and
nothing here imports the web framework: a job that only needs the analysis code
should not start a second copy of the server to do its work.

Running in a child process is what makes a job cancellable. The unmixing job is
the one that needs it: scikit-learn's ``NMF`` takes a fixed number of iterations
with no way to interrupt them from inside, so the only place a cancel can land is
the process boundary. A prediction is one pass per row of pixels through the
network, and is cancellable for the same reason.
"""

from __future__ import annotations

import json
import logging
import traceback
from pathlib import Path

from src import logs
from src.cnn.predict import predict
from src.data.loader import get_cumulative_hsi_cube, get_raw_hsi_cube
from src.data.grid import RamanGrid

from app import figures, predictions, session, unmix

logger = logging.getLogger(__name__)

#: The request file the server writes before starting this process.
REQUEST_NAME = "job.json"

#: The file this process keeps its progress in, read by the server while it runs.
STATE_NAME = "state.json"

#: The directory a load job writes its arrays into, moved into place when it finishes.
ARRAYS_DIRNAME = "session"


class Reporter:
    """Keeps ``state.json`` up to date for the server to read.

    Every write is a whole file replacement, so a reader never sees a half-written
    state: it either reads the previous step or the next one.

    Each step is also printed to this process's own output, which is the server's
    console. Printing rather than logging, for the same reason the command line prints
    its dialogue: this is a process entry point saying what it is doing, and the person
    watching the server's output is who that is for. ``logs.configure`` is still called,
    so anything the library under ``src`` reports arrives on the same console.
    """

    def __init__(self, directory: Path):
        self.directory = directory
        self.path = directory / STATE_NAME
        self.state = {"state": "running", "step": "", "detail": "", "error": None, "result": None}

    def _write(self) -> None:
        self.path.write_text(json.dumps(self.state, indent=2), encoding="utf-8")

    def step(self, step: str, detail: str = "") -> None:
        """Records the step being worked on now, and says so on this process's output.

        Args:
            step (str): A short name for the step, shown as it is.
            detail (str): What the step is working through, where a count is known.
        """
        self.state.update(state="running", step=step, detail=detail)
        self._write()
        print(f"{step}{f': {detail}' if detail else ''}", flush=True)

    def done(self, result: dict | None = None) -> None:
        """Records that the job finished, with whatever it produced."""
        self.state.update(state="done", step="finished", detail="", result=result)
        self._write()

    def failed(self, error: str) -> None:
        """Records that the job could not finish, and why."""
        self.state.update(state="error", error=error)
        self._write()
        print(f"the job stopped: {error}", flush=True)


def build(directory: Path, request: dict, reporter: Reporter) -> dict:
    """Reads a map once and writes the session it becomes.

    The whole read happens here, in one pass over the files, and what it leaves
    behind is the arrays the screens memory-map. Progress is reported by step rather
    than by file: ``get_cumulative_hsi_cube`` walks the grid internally with no hook
    to report against, so a per-file count would be invented here rather than
    measured.

    The same function serves a first load and a re-derive. Loading reads the map with
    no preprocessing and no crop, because the pipeline and the spectral range belong to
    the screens that use them; a re-derive carries the pipeline and the range the
    heatmap top bar was set to and replaces the session with the result. Both are the
    same read, so they are the same code.

    Args:
        directory (Path): This job's directory, which the arrays are written into.
        request (dict): ``path``, ``rows`` and ``cols`` of the map to read, with
            ``pipeline_id`` and the ``start`` and ``end`` bounds when re-deriving.
        reporter (Reporter): Where progress is recorded.

    Returns:
        dict: The session record, which the server publishes if the read succeeds.
    """
    path = request["path"]
    rows = int(request["rows"])
    cols = int(request["cols"])
    pipeline_id = int(request.get("pipeline_id", 0))
    start = request.get("start")
    end = request.get("end")

    # Checked here as well as in the route that started this job, because the two are
    # separate processes and the directory can change between them. This raises the
    # grid's own message, which is what the screen shows.
    files = RamanGrid(path, rows, cols).get_sorted_files()

    reporter.step("reading", f"{len(files)} files")
    cube = get_cumulative_hsi_cube(path, rows, cols, pipeline_id, start, end)

    reporter.step("saving", f"{len(session.ARRAY_NAMES)} arrays")
    record = session.write(directory / ARRAYS_DIRNAME, cube, {
        "directory": str(Path(path).expanduser()),
        "label": Path(path).expanduser().name,
        "rows": rows,
        "cols": cols,
        "files": len(files),
        "bands": int(cube.axis.shape[0]),
        "axis_start": float(cube.axis[0]),
        "axis_end": float(cube.axis[-1]),
        "pipeline_id": pipeline_id,
        "crop_start": start,
        "crop_end": end,
    })

    reporter.done(result=record)
    return record


def decompose(directory: Path, request: dict, reporter: Reporter) -> dict:
    """Unmixes a map once, and draws what came out of it.

    The map is read here rather than taken from the session, because the decomposition is
    over raw intensities: the preprocessing it runs behind is its own, and a map the
    heatmap screen derived with a pipeline is not a reason for this to unmix spectra that
    already had one applied. The read is the same call the load job makes, in this
    process, so the map the server is holding is untouched by anything this job does.

    When the estimator names no count there is nothing to decompose with, so the job
    finishes having produced the report and no panels. That is a finished job rather than
    a failed one: the estimator's findings are the answer, and inventing a count to
    decompose with would be the thing the rule exists to prevent.

    Args:
        directory (Path): This job's directory, which the figures are written into.
        request (dict): ``path``, ``rows`` and ``cols`` of the map, ``count`` as the
            component count to use or None to estimate one, and the ``start`` and ``end``
            bounds to crop to before decomposing.
        reporter (Reporter): Where progress is recorded.

    Returns:
        dict: The record the server reports to the screen, holding the count, the
        estimator's report when one was taken, the range that was decomposed, and the
        names of the figures.
    """
    path = request["path"]
    rows = int(request["rows"])
    cols = int(request["cols"])
    count = request.get("count")
    count = None if count is None else int(count)

    reporter.step("reading", f"{rows * cols} raw spectra")
    cube = unmix.crop(get_raw_hsi_cube(path, rows, cols), request.get("start"), request.get("end"))

    if count is None:
        # No detail: the step's own name is what there is to say, and a count is exactly
        # what is not known yet.
        reporter.step("estimating")
    count, estimate = unmix.resolve_count(cube, count)

    if count is None:
        result = {"unmixed": False, "count": None, "estimate": unmix.report(estimate)}
        reporter.done(result=result)
        return result

    reporter.step("decomposing", f"{count} components")
    abundance_maps, phases = unmix.decomposition(cube, count)

    reporter.step("drawing", f"{count} abundance maps")
    drawn = figures.unmix_pngs(phases, cube.spectral_axis, abundance_maps, directory)

    result = {
        "unmixed": True,
        "count": count,
        "estimate": None if estimate is None else unmix.report(estimate),
        "axis_start": float(cube.spectral_axis[0]),
        "axis_end": float(cube.spectral_axis[-1]),
        "bands": int(cube.spectral_data.shape[-1]),
        "spectra": drawn["spectra"],
        "maps": [
            {"component": component, "figure": name}
            for component, name in enumerate(drawn["maps"], start=1)
        ],
    }
    reporter.done(result=result)
    return result


def classify(directory: Path, request: dict, reporter: Reporter) -> dict:
    """Predicts a mineral for every pixel of a map, and draws the map of them.

    The prediction is the framework's own call, so a browser run and a command line run
    classify a spectrum the same way. It reads the map files itself, which is also why the
    run it is given is a full path: this job is a separate process, and it does not share
    the server's idea of where artifacts are kept.

    A refusal is reported as this job's answer rather than as a failure. ``predict``
    documents ``ValueError`` as the two guards it applies, a model that disagrees with its
    own manifest and a map whose measured range does not cover the range the model was
    trained on, and both of them exist because the model would answer anyway. The answer
    is the reason it is not shown, so the job finished and produced it.

    Args:
        directory (Path): This job's directory, which the predictions and the figure go in.
        request (dict): ``run_path`` as the trained run to predict with, and ``path``,
            ``rows`` and ``cols`` of the map to predict over.
        reporter (Reporter): Where progress is recorded.

    Returns:
        dict: The run, the threshold the figure was drawn at and its legend, or
        ``refused`` and the guard's own words.
    """
    run_path = request["run_path"]
    path = request["path"]
    rows = int(request["rows"])
    cols = int(request["cols"])
    run = Path(run_path).name

    reporter.step("predicting")
    try:
        labels, top5 = predict(run_path, path, rows, cols)
    except ValueError as error:
        result = {"refused": True, "run": run, "reason": str(error)}
        reporter.done(result=result)
        return result

    reporter.step("storing", f"{rows * cols} predictions")
    predictions.save_predictions(directory, labels, top5, run)

    # Named apart from the unmixing's own drawing step, since what is drawn here is one
    # map rather than a panel per component.
    reporter.step("mapping", "the mineral map")
    drawn = predictions.render(directory, predictions.DEFAULT_THRESHOLD)

    result = {"refused": False, "run": run, "rows": rows, "cols": cols, **drawn}
    reporter.done(result=result)
    return result


#: Every kind of job this process can run, by the name the request uses. A kind that is
#: not here is a request the server should not have written, so it is refused rather than
#: guessed at. ``derive`` is the same read as ``load``, with a pipeline and a range on it.
HANDLERS = {"load": build, "derive": build, "unmix": decompose, "predict": classify}


def run(directory: str) -> None:
    """Runs one job and records how it went.

    Never raises: an exception here would end the process with a traceback and leave
    the server with a job that stopped and no reason written down, so it is caught and
    put in the state file where the screen can show it.

    Args:
        directory (str): The job's directory, holding the request to run.
    """
    logs.configure()
    job_dir = Path(directory)
    reporter = Reporter(job_dir)

    try:
        request = json.loads((job_dir / REQUEST_NAME).read_text(encoding="utf-8"))
        kind = request.get("kind")
        if kind not in HANDLERS:
            reporter.failed(f"This server has no job called {kind!r}.")
            return
        HANDLERS[kind](job_dir, request, reporter)
    except Exception as error:  # noqa: BLE001  (see the docstring: the reason is the point)
        logger.debug("Job %s failed.", job_dir.name, exc_info=True)
        reporter.failed(f"{type(error).__name__}: {error}" if str(error) else repr(error))
        (job_dir / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
