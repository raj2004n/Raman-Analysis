"""Shared fixtures and the snapshot store.

Run ``pytest`` to compare against the recorded snapshots, or
``pytest --update-golden`` to re-record them. Re-recording is a deliberate act: if a
snapshot moves during a change that claims to preserve behaviour, that is the
finding, not an inconvenience.

The fixture grid is generated into a temporary directory on every run and is never
committed, so no real data is needed to run the suite.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import logs  # noqa: E402  (needs REPO_ROOT on the path)
from src.data.loader import get_raw_hsi_cube  # noqa: E402  (needs REPO_ROOT on the path)
from tests import _golden, synthetic  # noqa: E402


def pytest_addoption(parser):
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="Re-record tests/golden/snapshots.json from the current code.",
    )


class SnapshotStore:
    """Records fingerprints during a session and compares each one as it is checked.

    Comparing immediately means a mismatch fails the test that produced it rather
    than the session teardown. The file is written on teardown, so ``--update-golden``
    records everything the run actually reached.
    """

    def __init__(self, stored: dict, update: bool):
        self.stored = stored
        self.update = update
        self.collected: dict = {}

    def check(self, name: str, value) -> None:
        current = _golden.fingerprint(value)
        self.collected[name] = current

        if self.update:
            return
        if name not in self.stored:
            pytest.fail(
                f"no golden snapshot recorded for {name!r} yet.\n"
                f"    now: {current}\n"
                f"    Record it with: pytest --update-golden",
                pytrace=False,
            )
        recorded = self.stored[name]
        if recorded != current:
            pytest.fail(
                f"golden snapshot {name!r} changed.\n"
                f"    recorded: {recorded}\n"
                f"    now:      {current}\n"
                f"    If the change is intended, re-record with: pytest --update-golden",
                pytrace=False,
            )


@pytest.fixture(autouse=True)
def isolated_logging():
    """Keeps one test's logging configuration out of the next one.

    ``logs.configure`` sets the level of the ``src`` logger and attaches a handler to it,
    and the command line entry point calls it, which several tests exercise. Both of those
    outlive the test that caused them, so without this a later test would have the
    package's progress lines arriving in its captured output, and its own level set by
    whatever ran before it. The whole run is restored rather than the level alone, so a
    handler a test attached deliberately is removed too.
    """
    logger = logging.getLogger(logs.LOGGER_NAME)
    handlers = list(logger.handlers)
    level = logger.level

    yield

    logger.handlers[:] = handlers
    logger.setLevel(level)


@pytest.fixture(scope="session")
def golden(request):
    store = SnapshotStore(_golden.load_snapshots(), request.config.getoption("--update-golden"))
    yield store

    if store.update and store.collected:
        _golden.save_snapshots({**store.stored, **store.collected})


@pytest.fixture(scope="session")
def grid_dir(tmp_path_factory) -> Path:
    """A freshly generated 4x4 grid of .txt spectra. Nothing comes from real data."""
    return synthetic.write_grid(tmp_path_factory.mktemp("fixture_grid"))


@pytest.fixture(scope="session")
def raw_cube(grid_dir):
    return get_raw_hsi_cube(grid_dir, synthetic.ROWS, synthetic.COLS)
