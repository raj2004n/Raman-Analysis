"""The phase decomposition viewer's four jobs: forward its settings, know what "estimate it"
means, stop when there is no count to decompose with, and give the plot its own colours.

The settings used to be an inline dictionary in the viewer, which meant a caller could
not choose a solver, an iteration cap or a seed, and could not see what a run had used.
They come from ``NmfConfig`` now. These tests call the viewer with the decomposition and
the plotting replaced, so they check what the viewer asks for without running an
unmixing or opening a window.
"""

from __future__ import annotations

import logging
from dataclasses import replace

import numpy as np
import pytest

from src import logs
from src.analysis.phase_number_est import MethodEstimate, PhaseEstimate
from src.config import NmfConfig
from src.visualisation import view_phase_decomp


class _Recorder:
    """Stands in for ``ramanspy``'s NMF: records the kwargs, returns fixed output."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def apply(self, cube):
        n = self.kwargs["n_components"]
        return [np.zeros((2, 2)) for _ in range(n)], np.zeros((n, 4))


@pytest.fixture
def viewer(monkeypatch):
    """Replaces the decomposition, the plotting and the estimate, and hands back the recorder."""
    recorded = {}

    def fake_nmf(**kwargs):
        recorded.update(kwargs)
        return _Recorder(**kwargs)

    monkeypatch.setattr(view_phase_decomp.rp.analysis.decompose, "NMF", fake_nmf)
    monkeypatch.setattr(view_phase_decomp.rp.plot, "spectra", lambda *a, **k: None)
    monkeypatch.setattr(view_phase_decomp.rp.plot, "image", lambda *a, **k: None)
    monkeypatch.setattr(view_phase_decomp.plt, "show", lambda: None)
    monkeypatch.setattr(view_phase_decomp, "estimate_phase_number", lambda cube: PhaseEstimate(3, "high", ()))
    return recorded


def test_the_viewer_sends_the_default_settings(raw_cube, viewer):
    """With no config given, the decomposition gets the defaults for every setting."""
    view_phase_decomp.show_spectra_ext_viewer(raw_cube, 3)

    assert viewer == NmfConfig().as_kwargs(3)


def test_the_viewer_sends_the_settings_it_was_given(raw_cube, viewer):
    """A caller's config reaches the decomposition, including a seed and an iteration cap."""
    config = replace(NmfConfig(), max_iter=25, random_state=7, solver="mu")

    view_phase_decomp.show_spectra_ext_viewer(raw_cube, 2, nmf_config=config)

    assert viewer["n_components"] == 2
    assert viewer["max_iter"] == 25
    assert viewer["random_state"] == 7
    assert viewer["solver"] == "mu"


def test_none_means_estimate_the_count(raw_cube, viewer):
    """The new spelling of "I do not know how many phases there are"."""
    view_phase_decomp.show_spectra_ext_viewer(raw_cube, None)

    assert viewer["n_components"] == 3, "the estimate is what the decomposition must be given"


def test_minus_one_is_still_the_same_request(raw_cube, viewer):
    """The command line prompt offers -1 for "estimate it" and has for long enough that it stays."""
    view_phase_decomp.show_spectra_ext_viewer(raw_cube, -1)

    assert viewer["n_components"] == 3


def test_a_given_count_is_never_estimated(raw_cube, viewer, monkeypatch):
    """An explicit count must not be second-guessed by the estimator."""
    def explosion(cube):
        raise AssertionError("the estimator was called for a cube whose count was given")

    monkeypatch.setattr(view_phase_decomp, "estimate_phase_number", explosion)

    view_phase_decomp.show_spectra_ext_viewer(raw_cube, 4)

    assert viewer["n_components"] == 4


def test_an_undetermined_estimate_stops_before_the_decomposition(raw_cube, viewer, monkeypatch, caplog):
    """No count means no unmixing, and saying so is the whole result.

    The estimator reports what each method found, so the user has already been told
    why there is no answer. Decomposing anyway would need a count, and the only counts
    available are the estimates the bounds already threw away.

    The line saying so is a log record now rather than a print, so this reads it from the
    log. That it reaches stdout as well is covered where the handler is set up.
    """
    monkeypatch.setattr(
        view_phase_decomp, "estimate_phase_number",
        lambda cube: PhaseEstimate(
            None, "undetermined",
            (MethodEstimate("PCA 80%", None, "the estimate of 17 is above the limit of 10"),),
        ),
    )

    with caplog.at_level(logging.INFO, logger=logs.LOGGER_NAME):
        view_phase_decomp.show_spectra_ext_viewer(raw_cube, None)

    assert viewer == {}, "the decomposition was built without a count to give it"
    assert "Nothing to unmix without a phase count." in caplog.text


def test_the_colours_are_given_at_the_call(raw_cube, viewer, monkeypatch):
    """The viewer must not leave the colours to ramanspy's default.

    That default reaches matplotlib through ``plt.cm.get_cmap``, which matplotlib 3.11
    removed, so asking for it is what crashed this viewer on a current install. It is
    computed inside ramanspy, so the only way not to depend on it is to pass every colour
    in. One line is drawn per phase, so one colour each is what the plot expects.
    """
    seen = {}

    def fake_spectra(*args, **kwargs):
        seen.update(kwargs)
        return None

    monkeypatch.setattr(view_phase_decomp.rp.plot, "spectra", fake_spectra)
    view_phase_decomp.show_spectra_ext_viewer(raw_cube, 3)

    assert "color" in seen, "ramanspy would otherwise compute a default through a removed API"
    assert len(seen["color"]) == 3, "a colour for each of the three phases"
