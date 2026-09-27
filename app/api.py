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
from pathlib import Path

from fastapi import APIRouter, Body, HTTPException, Query

from src.data.grid import RamanGrid

from app import figures, frames, jobs, session, worker

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")


def _publish(job: jobs.Job) -> None:
    """Makes a finished load job's read the current session.

    Called from the job's watcher thread once the child has finished, so a load that
    completes while nobody is looking still becomes the map the next request sees.
    """
    session.replace_with(job.directory / worker.ARRAYS_DIRNAME)


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
        start = request.get("start")
        end = request.get("end")
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=400, detail=f"The request is unusable: {error}.")

    if not 0 <= pipeline_id <= 3:
        raise HTTPException(status_code=400, detail=f"Unknown pipeline id: {pipeline_id}.")

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


@router.post("/save")
def save(request: dict = Body(...)) -> dict:
    """Writes what a screen is showing to a file.

    Only the heatmap is written this way: the unmixing and prediction figures are
    written by the jobs that produce them, because there is nothing to export until
    they have run. Here the screen's own state is the input, so the file matches what
    the screen shows rather than a re-run of the same settings. The intensity range is
    part of that state: the map is drawn masked to it, and a file drawn to the whole
    range the data spans would not be the picture that was on screen.
    """
    cube = session.load_cube()
    if cube is None:
        raise HTTPException(status_code=409, detail="No map is loaded. Load one first.")

    kind = request.get("kind")
    if kind != "heatmap":
        raise HTTPException(status_code=400, detail=f"There is nothing to save for {kind!r}.")

    record = session.metadata()
    key = record.get("session_id")

    try:
        bounds = {name: float(request[name]) for name in ("i_min", "i_max") if request.get(name) is not None}
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=400, detail=f"The intensity range is not a pair of numbers: {error}.")

    try:
        path = figures.heatmap_png(
            cube,
            key,
            int(request.get("center", 0)),
            float(request.get("width_cm", 10.0)),
            int(request.get("pixel", 1)),
            bounds.get("i_min"),
            bounds.get("i_max"),
            str(request.get("name") or "heatmap.png"),
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    return {"path": str(path)}


@router.get("/session")
def read_session() -> dict:
    """Reports the map currently loaded, if there is one."""
    record = session.metadata()
    if record is None:
        return {"loaded": False}
    return {"loaded": True, **record}


@router.get("/directories")
def directories(path: str = Query("~")) -> dict:
    """Lists the directories inside one directory, for the Data screen's picker.

    A browser cannot name a path on the machine the server runs on, so the walk is
    done here. Only directories are listed, because a map is a directory and no file
    can be loaded as one, and dot directories are left out since a map is not hidden.
    """
    directory = Path(path).expanduser()
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
        "parent": str(directory.parent) if directory.parent != directory else None,
        "directories": entries,
    }


@router.get("/job/{job_id}")
def read_job(job_id: str) -> dict:
    """Reports where a job has got to."""
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="That job is no longer the current one.")
    return jobs.as_dict(job)


@router.post("/job/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    """Stops a running job by ending its process."""
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="That job is no longer the current one.")
    if not jobs.cancel(job_id):
        raise HTTPException(status_code=409, detail="That job is no longer running.")
    return jobs.as_dict(job)
