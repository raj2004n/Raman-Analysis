"""Starting a job, watching it, and stopping it.

One job at a time, by design rather than by accident: both heavy jobs are minutes
long, the machine has one user, and a queue would be state to keep for no one. A
second request while one is running is refused with the running job named.

The job itself is a child process, started with the spawn context rather than by
forking the server. A forked child of a threaded process inherits locks and
sockets it must not touch, and matplotlib and OpenBLAS both hold such state, so
spawning is the choice that does not depend on what the server happened to be
doing at the moment. Cancelling is then ``terminate``: the child is killed where
it stands, which is the only way to interrupt an ``NMF`` that cannot be.

A watcher thread joins the child and then publishes whatever the job produced. It
runs outside any request, so a load that finishes while nobody is polling the
status still becomes the session the next request sees.
"""

from __future__ import annotations

import json
import logging
import multiprocessing
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from src.paths import artifacts_dir

from app import worker

logger = logging.getLogger(__name__)

#: The directory holding one directory per job, below ``artifacts``.
JOBS_DIRNAME = "jobs"

#: How often the watcher reads the child's state file while the child is running.
POLL_SECONDS = 0.2

#: The spawn context. Chosen once here so a job is always started the same way, however
#: the server itself was started.
_CONTEXT = multiprocessing.get_context("spawn")


def jobs_dir() -> Path:
    """Returns:
        Path: ``<artifacts>/jobs``, created if it does not already exist.
    """
    directory = artifacts_dir() / JOBS_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)
    return directory


@dataclass
class Job:
    """One run of one job, and what is known about it.

    Attributes:
        id: The identifier the screens poll with.
        kind: Which job this is, as ``app.worker`` names it.
        request: The request the child was started with, kept so a failed job can be
            described from what was asked for rather than from what it managed.
        directory: The job's own directory, holding its request, state and output.
        state: 'running', 'done', 'error' or 'cancelled'.
        step: The step being worked on, or the last one reached.
        detail: What that step is working through, where a count is known.
        error: Why the job stopped, when it stopped for a reason.
        result: What the job produced, when it finished.
    """

    id: str
    kind: str
    request: dict
    directory: Path
    state: str = "running"
    step: str = ""
    detail: str = ""
    error: str | None = None
    result: dict | None = None
    process: multiprocessing.process.BaseProcess | None = field(default=None, repr=False)
    watcher: threading.Thread | None = field(default=None, repr=False)


_LOCK = threading.Lock()
_CURRENT: Job | None = None


def _read_state(job: Job) -> dict:
    """The child's state file as it stands, or an empty mapping before it exists."""
    path = job.directory / worker.STATE_NAME
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        # The child writes the whole file in one call, so a partial read is a race with
        # that write rather than a corrupt state; the next poll gets the whole of it.
        return {}


def _watch(job: Job, publish) -> None:
    """Follows the child to its end, then publishes what it produced.

    Args:
        job (Job): The job being watched.
        publish (callable): Called with the finished job. For a load, this is what
            makes the read map the current session.
    """
    while job.process.is_alive():
        if job.state != "running":
            # Cancelled from another thread. The child is on its way out, so its last
            # state file would otherwise put the step it died on back on the screen.
            break
        state = _read_state(job)
        job.step = state.get("step", job.step)
        job.detail = state.get("detail", job.detail)
        time.sleep(POLL_SECONDS)

    job.process.join()
    state = _read_state(job)

    if job.state == "cancelled":
        logger.info("Job %s was cancelled.", job.id)
        return

    if state.get("state") == "done":
        job.step = state.get("step", "finished")
        job.result = state.get("result")
        if publish is not None:
            # Published before the job is called done, so a caller that sees 'done' is
            # looking at a finished result rather than at one still being put in place.
            try:
                publish(job)
            except Exception as error:  # noqa: BLE001  (the watcher must not die here)
                job.state = "error"
                job.error = f"The job finished but its result could not be used: {error}"
                logger.exception("Publishing job %s failed.", job.id)
                return
        job.state = "done"
        return

    if state.get("state") == "error":
        job.state = "error"
        job.error = state.get("error") or "The job stopped without saying why."
        return

    # The child is gone and wrote neither of the two endings, so it was killed from
    # outside this server: the machine ran out of memory, or something else stopped it.
    job.state = "error"
    job.error = "The job stopped without finishing. See the server's own output for why."


def start(kind: str, request: dict, publish=None) -> Job:
    """Starts a job, refusing while another is running.

    Args:
        kind (str): Which job to run, as ``app.worker`` names it.
        request (dict): What the job should work on, written where the child reads it.
        publish (callable, optional): Called with the finished job, from the watcher
            thread. Used to make a finished read the current session.

    Returns:
        Job: The job that was started.

    Raises:
        RuntimeError: A job is already running. The message names it, since the caller's
            way out is either to wait for that one or to cancel it.
    """
    global _CURRENT

    with _LOCK:
        if _CURRENT is not None and _CURRENT.state == "running":
            raise RuntimeError(
                f"A {_CURRENT.kind} job is already running. Wait for it to finish, or "
                "cancel it, before starting another."
            )

        directory = jobs_dir() / uuid.uuid4().hex[:12]
        directory.mkdir(parents=True)
        (directory / worker.REQUEST_NAME).write_text(
            json.dumps({"kind": kind, **request}, indent=2), encoding="utf-8"
        )

        job = Job(id=directory.name, kind=kind, request=request, directory=directory)
        job.process = _CONTEXT.Process(target=worker.run, args=(str(directory),), daemon=True)
        job.process.start()

        job.watcher = threading.Thread(target=_watch, args=(job, publish), daemon=True)
        job.watcher.start()

        _CURRENT = job
        logger.info("Started %s job %s.", kind, job.id)
        return job


def current() -> Job | None:
    """Returns:
        Job or None: The job the server has run or is running, if there has been one.
    """
    return _CURRENT


def get(job_id: str) -> Job | None:
    """The job with this identifier.

    Args:
        job_id (str): The identifier from a start.

    Returns:
        Job or None: The job, or None when the identifier is not the current one. Only
        one job is kept, so an identifier from before the last one started is not
        answerable and a stale page's poll ends rather than reporting another job's
        progress as its own.
    """
    if _CURRENT is not None and _CURRENT.id == job_id:
        return _CURRENT
    return None


def cancel(job_id: str) -> bool:
    """Stops a running job.

    Args:
        job_id (str): The identifier from a start.

    Returns:
        bool: True when a running job was stopped, False when the identifier names no
        running job, which is not an error: the job may have finished between the
        request and this call.
    """
    job = get(job_id)
    if job is None or job.state != "running" or job.process is None:
        return False

    job.state = "cancelled"
    job.step = "cancelled"
    job.process.terminate()
    job.process.join()
    logger.info("Cancelled job %s.", job.id)
    return True


def as_dict(job: Job) -> dict:
    """The job as the screens read it.

    Args:
        job (Job): The job to describe.

    Returns:
        dict: Its identifier, kind, state, step and detail, with the error or the result
        when it has one.
    """
    return {
        "id": job.id,
        "kind": job.kind,
        "state": job.state,
        "step": job.step,
        "detail": job.detail,
        "error": job.error,
        "result": job.result,
    }
