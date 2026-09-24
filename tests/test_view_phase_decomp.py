"""The phase decomposition viewer's two jobs: forward its settings, and know what "estimate it" means.

The settings used to be an inline dictionary in the viewer, which meant a caller could
not choose a solver, an iteration cap or a seed, and could not see what a run had used.
They come from ``NmfConfig`` now. These tests call the viewer with the decomposition and
the plotting replaced, so they check what the viewer asks for without running an
unmixing or opening a window.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.config import NmfConfig
from src.visualisation import view_phase_decomp


class _Recorder:
    """Stands in for ``ramanspy``'s NMF: records the kwargs, returns fixed output."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def apply(self, cube):
        n = self.kwargs["n_components"]
        return [np.zeros((2, 2)) for _ in range(n)], np.zeros((n, 4))


class _Lines:
    """Enough of a matplotlib Axes for the viewer's recolouring loop."""

    def __init__(self, count):
        self.lines = [type("Line", (), {"set_color": lambda self, color: None})() for _ in range(count)]

    def get_lines(self):
        return self.lines


@pytest.fixture
def viewer(monkeypatch):
    """Replaces the decomposition, the plotting and the estimate, and hands back the recorder."""
    recorded = {}

    def fake_nmf(**kwargs):
        recorded.update(kwargs)
        return _Recorder(**kwargs)

    monkeypatch.setattr(view_phase_decomp.rp.analysis.decompose, "NMF", fake_nmf)
    monkeypatch.setattr(view_phase_decomp.rp.plot, "spectra", lambda *a, **k: _Lines(recorded.get("n_components", 1)))
    monkeypatch.setattr(view_phase_decomp.rp.plot, "image", lambda *a, **k: None)
    monkeypatch.setattr(view_phase_decomp.plt, "show", lambda: None)
    monkeypatch.setattr(view_phase_decomp, "estimate_phase_number", lambda cube: (3, "high"))
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
