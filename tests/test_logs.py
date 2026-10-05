"""What ``logs.configure`` installs, and what it deliberately leaves alone.

The point of this module is that the package's report reaches a terminal when someone
runs an entry point and stays out of the way when the package is imported. So these
tests are about the handler: how many there are, where they write, and that a second call
does not double them. Choosing stdout over the logging default is deliberate, since the
command line prints its prompts to stdout and a report that arrived on stderr would not
stay in order with them once the output is piped.
"""

from __future__ import annotations

import io
import logging

import pytest

from src import logs


@pytest.fixture
def logger():
    """The package logger, empty of handlers, so each test starts from nothing."""
    package_logger = logging.getLogger(logs.LOGGER_NAME)
    package_logger.handlers[:] = []
    package_logger.setLevel(logging.NOTSET)

    yield package_logger

    package_logger.handlers[:] = []
    package_logger.setLevel(logging.NOTSET)


def _our_handlers(package_logger):
    return [h for h in package_logger.handlers if getattr(h, "_src_stream_handler", False)]


def test_configure_sets_the_level_and_installs_one_handler(logger):
    """One handler at the level asked for, which is what makes the report visible."""
    logs.configure(logging.INFO)

    assert logger.level == logging.INFO
    assert len(_our_handlers(logger)) == 1


def test_configure_twice_does_not_double_every_line(logger):
    """Both entry points configure, and a test may configure before calling one.

    Adding a handler per call would print every record once per call.
    """
    logs.configure()
    logs.configure()

    assert len(_our_handlers(logger)) == 1


def test_configure_leaves_a_handlers_somebody_else_added(logger):
    """A caller that wants the records somewhere else adds their own and keeps it."""
    theirs = logging.NullHandler()
    logger.addHandler(theirs)

    logs.configure()

    assert theirs in logger.handlers
    assert len(_our_handlers(logger)) == 1


def test_the_records_are_written_to_stdout(logger, monkeypatch):
    """stdout rather than the logging default of stderr, so the report stays in order."""
    logs.configure()
    captured = io.StringIO()
    monkeypatch.setattr("sys.stdout", captured)

    logging.getLogger(f"{logs.LOGGER_NAME}.some.module").info("a line of the report")

    assert captured.getvalue() == "a line of the report\n"


def test_the_handler_follows_a_redirect_of_stdout(logger, monkeypatch):
    """The stream is looked up per record, so redirecting stdout redirects the log too.

    A handler that captured ``sys.stdout`` when it was built would keep writing to the
    stream it captured, and its records would be the only output that ignored the
    redirect.
    """
    logs.configure()
    first = io.StringIO()
    monkeypatch.setattr("sys.stdout", first)
    logging.getLogger(logs.LOGGER_NAME).info("before")

    second = io.StringIO()
    monkeypatch.setattr("sys.stdout", second)
    logging.getLogger(logs.LOGGER_NAME).info("after")

    assert first.getvalue() == "before\n"
    assert second.getvalue() == "after\n"


def test_the_format_is_the_message_alone(logger, monkeypatch):
    """The modules log whole sentences, so a level and a logger name would be noise."""
    logs.configure()
    captured = io.StringIO()
    monkeypatch.setattr("sys.stdout", captured)

    logging.getLogger(f"{logs.LOGGER_NAME}.analysis.rolling_auc").info("Estimated %s phases", 3)

    assert captured.getvalue() == "Estimated 3 phases\n"


def test_a_module_logger_reaches_the_packages_handler(logger, monkeypatch):
    """The handler is on the package logger, so every module under it is reported."""
    logs.configure()
    captured = io.StringIO()
    monkeypatch.setattr("sys.stdout", captured)

    logging.getLogger(f"{logs.LOGGER_NAME}.cnn.train").warning("a problem")

    assert captured.getvalue() == "a problem\n"
