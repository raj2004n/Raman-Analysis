"""How the framework reports what it is doing.

Modules here attach a logger and log to it. Nothing in the package configures logging at
import time, because that would change the root logger of whatever imported it, which is
not a library's decision to make. A caller that wants the output calls ``configure``; both
entry points do, which is what keeps a run's report on a terminal.

The handler goes on the ``src`` logger rather than the root, and writes to stdout rather
than the logging default of stderr, so a report appears in the order it happened relative
to the prompt dialogue the command line prints around it.
"""

from __future__ import annotations

import logging
import sys

#: The logger every module in the package logs under, and the one ``configure`` sets up.
#: Setting the level here rather than on the root is what keeps the package's output a
#: property of the package.
LOGGER_NAME = "src"

#: The record's message and nothing else. The modules log whole sentences, so a level, a
#: timestamp and a logger name would all be noise in a terminal run.
FORMAT = "%(message)s"

#: Marks the handler ``configure`` installed, so a second call can tell it apart from a
#: handler a caller added and leave that one alone.
_HANDLER_MARKER = "_src_stream_handler"


class _StdoutHandler(logging.StreamHandler):
    """A handler that writes to whatever ``sys.stdout`` is when it emits a record.

    ``logging.StreamHandler`` holds the stream it was built with. A handler built before
    something replaces ``sys.stdout`` (a test capturing output, a caller redirecting it)
    then keeps writing to the stream it captured, and the records miss the redirect while
    everything printed around them obeys it. Looking the stream up per record keeps the
    log and the print statements on the same stream, which is the point of choosing
    stdout in the first place.
    """

    def emit(self, record):
        """Writes one record to the ``sys.stdout`` of the moment, not of construction."""
        self.stream = sys.stdout
        super().emit(record)


def configure(level=logging.INFO):
    """Attaches one stdout handler to the ``src`` logger and sets its level.

    Called by the entry points, not at import. Calling it more than once is safe: the
    handler it installed last time is recognised and not added a second time, so a caller
    that configures as well does not make every line print twice. A handler somebody else
    added is left in place.

    Args:
        level (int, optional): The level the package logs at. Defaults to
            ``logging.INFO``, which is where the progress and result lines sit.

    Returns:
        logging.Logger: the ``src`` logger, configured.
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)

    if not any(getattr(handler, _HANDLER_MARKER, False) for handler in logger.handlers):
        handler = _StdoutHandler()
        handler.setFormatter(logging.Formatter(FORMAT))
        setattr(handler, _HANDLER_MARKER, True)
        logger.addHandler(handler)

    return logger
