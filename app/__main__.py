"""Starts the browser interface on this machine.

    python -m app

The server binds to the loopback address, so the interface is reachable from a
browser on this desktop and from nowhere else. There is no login, no session
cookie and no second machine in the design; a port bound to the loopback address
is what makes that true rather than a promise.
"""

from __future__ import annotations

import uvicorn

from src import logs

#: The address the interface listens on. Loopback only: this is a desktop tool,
#: and binding wider would put a shell around the analysis on the local network.
HOST = "127.0.0.1"

#: The port the design's rail footer names.
PORT = 8000


def main() -> None:
    """Serves the application until interrupted."""
    logs.configure()
    uvicorn.run("app.main:app", host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
