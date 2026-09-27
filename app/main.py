"""The HTTP surface of the browser interface.

One FastAPI application serves two things: the four screens, which are one HTML
file the browser routes between by path, and the JSON routes those screens call.
The analysis itself is not here. Every route either reports something the
framework already knows how to compute or asks a child process to compute it.

Nothing configures logging at import. ``main`` calls ``logs.configure`` so a run
of the server reports the same way a run of the command line does, and importing
this module to build a test client does not touch the root logger.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app import api

logger = logging.getLogger(__name__)

#: Where the front end lives. Everything the browser loads from the repository is
#: under here, which is why the static mount has one directory and not several.
STATIC_DIR = Path(__file__).resolve().parent / "static"

#: The shell page, served as-is for every screen.
SHELL_PATH = STATIC_DIR / "index.html"

#: The screens, in the order the rail lists them. Each is served at ``/<name>``,
#: and the shell reads its own path to decide which one to draw, so adding a
#: screen means adding it here and to the shell rather than writing a new page.
SCREENS = ("data", "heatmap", "unmix", "predict")


def _shell() -> HTMLResponse:
    """Returns:
        HTMLResponse: The one page the four screens are drawn into.
    """
    return HTMLResponse(SHELL_PATH.read_text(encoding="utf-8"))


def create_app() -> FastAPI:
    """Builds the application.

    A function rather than a module-level side effect, so a test can build its
    own instance without sharing the state a served one accumulates.

    Returns:
        FastAPI: The application, with the shell mounted at each screen's path
        and the front end's files at ``/static``.
    """
    application = FastAPI(title="Raman Maps", docs_url=None, redoc_url=None)
    application.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    application.include_router(api.router)

    @application.get("/", include_in_schema=False)
    def index() -> RedirectResponse:
        """Sends the root to the first screen, so a bare localhost works."""
        return RedirectResponse("/data")

    for screen in SCREENS:
        application.get(f"/{screen}", include_in_schema=False)(_shell)

    return application


#: What uvicorn imports. Tests build their own through ``create_app``.
app = create_app()
