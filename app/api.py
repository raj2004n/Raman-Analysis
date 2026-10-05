"""The JSON routes the screens call.

Everything the browser asks for that is not the page itself. A route here either
reports something cheap enough to answer from what is already known, or starts a
job and returns its identifier for the screen to poll. Nothing in this module
reads a map file inside a request: the one read that happens is the load job's,
and it happens in its own process.

The request bodies are plain mappings rather than models built from a web
framework's own validation layer. A load request is three fields and the messages
worth showing for it come from the grid itself, which already says which pixel
numbers are missing or repeated, so there is nothing for a second validator to
add except a second way of saying it.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from fastapi import APIRouter, Body, HTTPException, Query
from fastapi.responses import FileResponse

from src.data.grid import RamanGrid
from src.paths import DATA_DIR_ENV, data_root

from app import figures, frames, jobs, predictions, saved_maps, session, unmix, worker

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

#: What one of a job's figures is allowed to be named. The name reaches the filesystem,
#: so its shape is fixed here rather than left to whatever the job wrote: no separators, no
#: leading dot, and a PNG, which is every figure this server draws and no other file in a
#: job's directory.
FIGURE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.png$")


def _publish(job: jobs.Job) -> None:
    """Makes a finished load job's read the current session, and remembers it.

    Called from the job's watcher thread once the child has finished, so a load that
    completes while nobody is looking still becomes the map the next request sees.

    A map read from outside the map root is opened but not saved, because the list names
    its maps by a path relative to the root and there is no such path for it.

    The record carries the grid the map was read with, and the entry keeps it: the shape of
    a directory cannot be worked out from the files in it, so a map opened from the list
    would otherwise have to have its size typed in again.
    """
    record = session.replace_with(job.directory / worker.ARRAYS_DIRNAME)
    saved_maps.remember(
        record.get("label", ""),
        record.get("directory", ""),
        record.get("rows"),
        record.get("cols"),
    )


def _grid_report(path: str, rows: int, cols: int) -> dict:
    """Checks a directory against a grid size without reading any file.

    Args:
        path (str): The map directory, as typed. ``~`` is expanded.
        rows, cols (int): The grid the files are expected to fill.

    Returns:
        dict: Whether the directory holds the grid, the count expected and the count
        found, and the grid's own message when it does not. The message is passed
        through as written: it names the pixel numbers that are missing or repeated and
        any file that was skipped for having no number, which is more than a route
        could say from the outside.
    """
    expected = rows * cols
    try:
        files = RamanGrid(path, rows, cols).get_sorted_files()
    except (FileNotFoundError, ValueError) as error:
        return {"ok": False, "expected": expected, "count": None, "message": str(error)}
    return {"ok": True, "expected": expected, "count": len(files), "message": None}


def _wavenumber_bounds(start, end) -> dict:
    """The two ends of a spectral range, either of which may be left out.

    Both routes that take a range take it the same way: a blank end means the end of the
    map's own range rather than a default number, and a pair that is the wrong way round
    is refused here, before either route has done anything that costs.

    Args:
        start, end: The bounds as they arrived, which may be None, blank or a string.

    Returns:
        dict: ``start`` and ``end`` as floats, each None when it was left out.

    Raises:
        HTTPException: A bound is not a number, or the range starts at or past its end.
    """
    bounds = {}
    for name, value in (("start", start), ("end", end)):
        if value in (None, ""):
            bounds[name] = None
            continue
        try:
            bounds[name] = float(value)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail=f"{name} is not a wavenumber: {value!r}.")

    if bounds["start"] is not None and bounds["end"] is not None and bounds["start"] >= bounds["end"]:
        raise HTTPException(
            status_code=400,
            detail=f"The range must start below where it ends: {bounds['start']} to {bounds['end']}.",
        )
    return bounds


@router.get("/grid")
def grid(
    path: str = Query(...),
    rows: int = Query(..., ge=1),
    cols: int = Query(..., ge=1),
) -> dict:
    """Reports whether a directory holds the grid that was asked for."""
    return _grid_report(path, rows, cols)


@router.post("/map")
def load_map(request: dict = Body(...)) -> dict:
    """Reads a map and makes it the session the screens work on.

    The grid is checked before the job starts, so a directory that cannot be laid out
    is refused here with the grid's own message and no process is started for it. The
    read itself is the job, because it is the long part.
    """
    try:
        path = str(request["path"]).strip()
        rows = int(request["rows"])
        cols = int(request["cols"])
    except (KeyError, TypeError, ValueError) as error:
        raise HTTPException(status_code=400, detail=f"The request is incomplete: {error}.")

    if not path:
        raise HTTPException(status_code=400, detail="No map directory was given.")

    report = _grid_report(path, rows, cols)
    if not report["ok"]:
        raise HTTPException(status_code=400, detail=report["message"])

    try:
        job = jobs.start("load", {"path": path, "rows": rows, "cols": cols}, publish=_publish)
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error))

    return {"job": jobs.as_dict(job), "grid": report}


@router.get("/frame")
def heatmap_frame(
    center: int = Query(0, ge=0),
    width_cm: float = Query(10.0, gt=0),
    pixel: int = Query(1, ge=1),
) -> dict:
    """Draws one band of the loaded map, and the spectrum of one pixel.

    Answered from the session in memory, so moving the position slider is a request
    against arrays the server already holds and never a read of the map files.
    """
    cube = session.load_cube()
    if cube is None:
        raise HTTPException(status_code=409, detail="No map is loaded. Load one first.")

    record = session.metadata()
    key = record.get("session_id")

    try:
        return frames.frame(cube, key, center, width_cm, pixel)
    except ValueError as error:
        # The window does not fit the range. Recoverable: the screen keeps the view it
        # had and shows this message, rather than drawing a band that is not there.
        raise HTTPException(status_code=400, detail=str(error))


@router.post("/derive")
def derive(request: dict = Body(...)) -> dict:
    """Reads the loaded map again, with a pipeline and a range applied.

    The heatmap's pipeline and range are the two things that change what the cube is
    rather than how it is drawn, so changing either is the one control that costs a
    read. The result replaces the session, which is what the screen's own warning says
    it does.
    """
    record = session.metadata()
    if record is None:
        raise HTTPException(status_code=409, detail="No map is loaded. Load one first.")

    try:
        pipeline_id = int(request.get("pipeline_id", 0))
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=400, detail=f"The request is unusable: {error}.")

    if not 0 <= pipeline_id <= 3:
        raise HTTPException(status_code=400, detail=f"Unknown pipeline id: {pipeline_id}.")

    bounds = _wavenumber_bounds(request.get("start"), request.get("end"))

    try:
        job = jobs.start(
            "derive",
            {
                "path": record["directory"],
                "rows": record["rows"],
                "cols": record["cols"],
                "pipeline_id": pipeline_id,
                "start": bounds["start"],
                "end": bounds["end"],
            },
            publish=_publish,
        )
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error))

    return {"job": jobs.as_dict(job)}


@router.get("/unmix/settings")
def unmix_settings() -> dict:
    """What a decomposition will run with, for the screen to show before it is run.

    Served rather than written into the front end, so the values on the screen are the
    ones the code will use: the settings are read from ``NmfConfig`` at the moment they
    are asked for, and the pipeline is the same list of steps the job builds.
    """
    return unmix.settings()


@router.post("/unmix")
def unmix_map(request: dict = Body(...)) -> dict:
    """Unmixes the loaded map, in a child process.

    A decomposition runs to ten thousand iterations by default, so it is the one job the
    design insists can be stopped. It runs in a child for that reason: scikit-learn's NMF
    cannot be interrupted from inside, so the process boundary is the only place a cancel
    can land.

    The range here crops inside the run, which is not what the same control does on the
    heatmap: there it replaces the loaded cube, and here it narrows what this one
    decomposition is taken over and leaves the map the other screens hold alone.

    Nothing is published when the job finishes. An unmixing is a result rather than a
    state the map is in, so it does not become the session.
    """
    record = session.metadata()
    if record is None:
        raise HTTPException(status_code=409, detail="No map is loaded. Load one first.")

    count = None
    if not request.get("estimate", True):
        try:
            count = int(request.get("count"))
        except (TypeError, ValueError):
            raise HTTPException(
                status_code=400,
                detail=f"A fixed phase count has to be a whole number: {request.get('count')!r}.",
            )
        if count < unmix.COUNT_MIN:
            raise HTTPException(
                status_code=400,
                detail=f"A decomposition needs at least {unmix.COUNT_MIN} component, not {count}.",
            )

    bounds = _wavenumber_bounds(request.get("start"), request.get("end"))

    try:
        job = jobs.start("unmix", {
            "path": record["directory"],
            "rows": record["rows"],
            "cols": record["cols"],
            "count": count,
            "start": bounds["start"],
            "end": bounds["end"],
        })
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error))

    return {"job": jobs.as_dict(job)}


@router.get("/runs")
def list_runs() -> dict:
    """The trained runs on disk, newest first, and the directory they were looked for in.

    Answered without loading a model: a run's facts come from its manifest file, so the
    picker does not pay for the network weights of every run it lists.
    """
    return predictions.catalogue()


@router.post("/predict")
def predict_map(request: dict = Body(...)) -> dict:
    """Predicts a mineral for every pixel of the loaded map, in a child process.

    The model and the manifest it was trained with decide how the prediction is made, so
    the run is the one thing this route takes: there is nothing to configure here that the
    run does not already record. A run whose measured range does not cover what the model
    was trained on is refused by ``predict`` itself, and that refusal is the job's own
    result rather than an error, because the model would answer anyway.

    The run is resolved to a path here rather than a name being looked up again in the
    child: the child is a separate process that builds its own idea of where the
    repository keeps things, so what it is given has to stand on its own.
    """
    record = session.metadata()
    if record is None:
        raise HTTPException(status_code=409, detail="No map is loaded. Load one first.")

    name = str(request.get("run") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="No run was given to predict with.")

    try:
        run_path = predictions.resolve_run(name)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))

    try:
        job = jobs.start("predict", {
            "run_path": str(Path(run_path).resolve()),
            "path": record["directory"],
            "rows": record["rows"],
            "cols": record["cols"],
        })
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error))

    return {"job": jobs.as_dict(job), "run": run_path.name}


@router.post("/predict/render")
def predict_render(request: dict = Body(...)) -> dict:
    """Redraws the predicted map at another confidence threshold.

    No model runs here. The predictions were computed by the job and are on disk, so the
    threshold only decides which pixels are drawn as unknown, which is what lets the
    slider be moved rather than answered once. The figure keeps the name the job gave it,
    so a page that already knows the job's identifier asks for it the same way.
    """
    job = _finished_prediction()
    threshold = _threshold(request.get("threshold"))

    try:
        drawn = predictions.render(job.directory, threshold)
    except FileNotFoundError:
        raise HTTPException(
            status_code=409,
            detail="That prediction refused its run, so there are no pixels to draw.",
        )
    return drawn


@router.post("/save")
def save(request: dict = Body(...)) -> dict:
    """Writes what a screen is showing to a file.

    Two screens export. The heatmap is written from the state on screen, so the file
    matches the picture rather than a re-run of the same settings. The prediction map is
    drawn again from the job's own predictions at the threshold the slider is on, and
    carries the title and the legend the command line's saved file carries, because a file
    under ``outputs`` is read away from the screen that made it.

    The unmixing figures are written by the job that produces them instead. There is
    nothing to export until it has run, and what it wrote belongs to the page showing it.
    """
    kind = request.get("kind")
    if kind == "heatmap":
        return _save_heatmap(request)
    if kind == "predict":
        return _save_prediction(request)
    raise HTTPException(status_code=400, detail=f"There is nothing to save for {kind!r}.")


def _output_name(value) -> str:
    """The file name to write under ``outputs``, from what the screen was given.

    The screen offers the whole path the command line prints, so a value with the outputs
    directory on its front is reduced to the name it ends in. Reducing it to the last
    component is also what keeps a name from reaching anywhere else: a path that climbs
    out of the directory arrives here as the name it was climbing towards, and under
    ``outputs`` is where that name is written.

    Raises:
        HTTPException: What arrived is not the name of a file this server draws.
    """
    name = Path(str(value or "")).name
    if not FIGURE_NAME.match(name):
        raise HTTPException(
            status_code=400, detail=f"That is not a figure name to write under outputs: {value!r}."
        )
    return name


def _save_heatmap(request: dict) -> dict:
    """Writes the band on screen, with the window, the pixel and the range it is drawn in."""
    cube = session.load_cube()
    if cube is None:
        raise HTTPException(status_code=409, detail="No map is loaded. Load one first.")

    record = session.metadata()
    key = record.get("session_id")

    try:
        bounds = {name: float(request[name]) for name in ("i_min", "i_max") if request.get(name) is not None}
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=400, detail=f"The intensity range is not a pair of numbers: {error}.")

    name = _output_name(request.get("name") or "heatmap.png")
    try:
        path = figures.heatmap_png(
            cube,
            key,
            int(request.get("center", 0)),
            float(request.get("width_cm", 10.0)),
            int(request.get("pixel", 1)),
            bounds.get("i_min"),
            bounds.get("i_max"),
            name,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    return {"path": str(path)}


def _save_prediction(request: dict) -> dict:
    """Writes the prediction map on screen, at the threshold the slider is on."""
    job = _finished_prediction()
    threshold = _threshold(request.get("threshold"))
    name = _output_name(request.get("name") or predictions.EXPORT_NAME)

    try:
        path = predictions.export(job.directory, threshold, name)
    except FileNotFoundError:
        raise HTTPException(
            status_code=409, detail="That prediction refused its run, so there is nothing to draw."
        )
    return {"path": str(path)}


def _threshold(value) -> float:
    """The confidence threshold, as the slider sends it.

    Raises:
        HTTPException: It is not a number, or not a confidence: the threshold decides which
            pixels are drawn as unknown, and a value outside (0, 1] would draw every pixel
            one way or the other rather than deciding anything.
    """
    try:
        threshold = float(value)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail=f"The threshold is not a number: {value!r}.")
    if not 0.0 < threshold <= 1.0:
        raise HTTPException(
            status_code=400,
            detail=f"The threshold is a confidence, so it lies above 0 and up to 1, not {threshold}.",
        )
    return threshold


def _finished_prediction() -> jobs.Job:
    """The prediction the screen is looking at, if there is one.

    The predictions live with the job that computed them, and the runner holds one job at
    a time, so the job it is holding is the one whose map is on screen.

    Raises:
        HTTPException: No prediction has run, or the one that did is still going.
    """
    job = jobs.current()
    if job is None or job.kind != "predict" or job.state != "done":
        raise HTTPException(status_code=409, detail="No prediction has been run yet.")
    return job


@router.get("/session")
def read_session() -> dict:
    """Reports the map currently loaded, if there is one."""
    record = session.metadata()
    if record is None:
        return {"loaded": False}
    return {"loaded": True, **record}


@router.get("/maps")
def list_saved_maps() -> dict:
    """The maps the interface has opened before, and the root their paths are relative to.

    The root travels with the list because the screen shows it: the full path of a saved
    map is the root and the entry's own path, and the paths in the file mean nothing
    without it. The list is answered whether or not the root is there, since a list that
    can still be read is worth showing when its maps have moved away.
    """
    return {"root": str(data_root()), "entries": saved_maps.entries()}


def _saved_path(request: dict) -> str:
    """The path of the saved map a request names.

    The screen sends back the path it was given, which is the list's own relative form,
    so nothing is resolved here: the path is a key into the list, and a change is refused
    when no entry has it.

    Raises:
        HTTPException: The request names no map.
    """
    stored = str(request.get("path") or "").strip()
    if not stored:
        raise HTTPException(status_code=400, detail="No saved map was named.")
    return stored


def _saved_change(change, *args) -> dict:
    """Runs a change to the saved list and answers with the list as it now stands.

    Every change answers the whole list, so the screen redraws from one response rather
    than asking again for what it has just changed.

    Raises:
        HTTPException: The path names no saved map, or the directory it names is gone
            (404); or the change was refused, which is a path climbed out of the root or a
            name that is not a folder name (400).
    """
    try:
        entries = change(*args)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    return {"root": str(data_root()), "entries": entries}


@router.post("/maps/label")
def rename_saved_label(request: dict = Body(...)) -> dict:
    """Renames what a saved map is called, leaving the directory where it is."""
    return _saved_change(
        saved_maps.rename_label, _saved_path(request), str(request.get("label") or "")
    )


@router.post("/maps/path")
def rename_saved_path(request: dict = Body(...)) -> dict:
    """Renames the directory a saved map names, moving it inside the map root."""
    return _saved_change(
        saved_maps.rename_path, _saved_path(request), str(request.get("name") or "")
    )


@router.post("/maps/delete")
def delete_saved_map(request: dict = Body(...)) -> dict:
    """Removes a saved map: the entry, and the directory it names.

    The confirmation is the screen's, not a field here. By the time a request is sent the
    answer is already the user's, and a flag carried alongside it would only be a second
    copy of that decision to keep in step with the screen.
    """
    return _saved_change(saved_maps.delete, _saved_path(request))


def _map_root() -> Path:
    """The directory the maps are opened from, which the picker may not leave.

    Refused rather than created when it is missing: the directory holds the user's own
    maps, and under a bind mount Docker would create a missing one owned by root.

    Raises:
        HTTPException: The root is not there.
    """
    root = data_root()
    if not root.is_dir():
        raise HTTPException(
            status_code=404,
            detail=(
                f"The map directory {root} is not there. Create it, or name another "
                f"with {DATA_DIR_ENV}."
            ),
        )
    return root


@router.get("/directories")
def directories(path: str = Query("")) -> dict:
    """Lists the directories inside one directory, for the Data screen's picker.

    A browser cannot name a path on the machine the server runs on, so the walk is
    done here. Only directories are listed, because a map is a directory and no file
    can be loaded as one, and dot directories are left out since a map is not hidden.

    The walk is confined to the map root, and a blank path is the root itself. Every map
    the picker can offer then has a path relative to the root, which is what lets a saved
    map be renamed or deleted without reaching anything outside the user's own maps.
    """
    root = _map_root()
    directory = (root / path.strip()).resolve() if path.strip() else root

    if directory != root and root not in directory.parents:
        raise HTTPException(
            status_code=403, detail=f"{directory} is outside the map directory {root}."
        )
    if not directory.is_dir():
        raise HTTPException(status_code=404, detail=f"{directory} is not a directory.")

    try:
        entries = sorted(
            entry.name for entry in directory.iterdir()
            if entry.is_dir() and not entry.name.startswith(".")
        )
    except PermissionError:
        raise HTTPException(status_code=403, detail=f"{directory} cannot be read.")

    return {
        "path": str(directory),
        # Nothing above the root, so the picker's way back up stops there.
        "parent": str(directory.parent) if directory != root else None,
        "directories": entries,
    }


@router.get("/job/{job_id}")
def read_job(job_id: str) -> dict:
    """Reports where a job has got to."""
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="That job is no longer the current one.")
    return jobs.as_dict(job)


@router.get("/job/{job_id}/figure/{name}")
def job_figure(job_id: str, name: str) -> FileResponse:
    """Sends one of a job's own figures to the browser.

    A job's figures belong to that job and are read from its own directory, so a page
    showing a finished run's panels asks for them by the identifier it was given and
    nothing else can name them. The name is checked against the shape a figure has before
    it reaches the filesystem, since it arrives from the browser.
    """
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="That job is no longer the current one.")

    path = job.directory / name
    if not FIGURE_NAME.match(name) or not path.is_file():
        raise HTTPException(status_code=404, detail=f"No figure is called {name!r}.")
    return FileResponse(path, media_type="image/png")


@router.get("/job/{job_id}/predictions")
def job_predictions(job_id: str) -> dict:
    """Sends a prediction job's per-pixel table to the browser.

    Asked for once, when the job finishes, rather than with every move of the threshold
    slider: the pointer reads the five most likely minerals of a pixel out of this table
    many times a second, and a request per hover would be felt. The table is the job's
    own file, so it is the same one a re-render reads.
    """
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="That job is no longer the current one.")
    if job.kind != "predict":
        raise HTTPException(status_code=404, detail=f"A {job.kind} job has no predictions.")

    try:
        return predictions.load_predictions(job.directory)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, detail="That prediction refused its run, so it has no pixels."
        )


@router.post("/job/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    """Stops a running job by ending its process."""
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="That job is no longer the current one.")
    if not jobs.cancel(job_id):
        raise HTTPException(status_code=409, detail="That job is no longer running.")
    return jobs.as_dict(job)
