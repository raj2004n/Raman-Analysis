"""The command line hands each viewer what it needs, and the loaders what they ask for.

The three modes are thin, but they are where the loaders and the viewers meet, and the
heatmap mode is where the four values of a cube used to be spread across an unpacking
statement. These tests replace ``prompt_args`` so no input is read, and replace the
viewers so no window opens, which leaves the wiring itself as the thing under test.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts import plot_raman
from src.data.loader import Cube


@pytest.fixture
def prompted(monkeypatch):
    """Records what each viewer was handed, with the prompts and the loaders replaced."""
    recorded = {}

    def fake_prompt_args(mode, kwargs):
        monkeypatch.setattr(plot_raman, "prompt_args", lambda: (recorded["path"], 2, 3, mode, kwargs))

    monkeypatch.setattr(plot_raman, "show_hsi_viewer", lambda *a, **k: recorded.update(viewer=("hsi", a, k)))
    monkeypatch.setattr(plot_raman, "show_spectra_ext_viewer", lambda *a, **k: recorded.update(viewer=("phases", a, k)))
    monkeypatch.setattr(plot_raman, "show_prediction_map", lambda *a, **k: recorded.update(viewer=("predict", a, k)))
    recorded["prompt_args"] = fake_prompt_args
    return recorded


def _cube():
    rows, cols, bands = 2, 3, 4
    return Cube(
        cube=np.zeros((rows, cols, bands)),
        spectra=np.zeros((rows * cols + 1, bands)),
        axis=np.linspace(100, 900, bands),
        pixel_map=np.arange(1, rows * cols + 1).reshape(rows, cols),
    )


def test_the_heatmap_mode_passes_the_whole_cube(prompted, monkeypatch):
    """One object, not four positionally unpacked values plus a grid size."""
    prompted["path"] = "~/data"
    cube = _cube()
    monkeypatch.setattr(plot_raman, "get_cumulative_hsi_cube", lambda *a, **k: cube)
    prompted["prompt_args"]("heatmap", {"pipeline_id": 1, "start": 200.0, "end": 800.0})

    plot_raman.main()

    kind, args, kwargs = prompted["viewer"]
    assert kind == "hsi"
    assert args == (cube,), "the viewer takes the cube and derives the grid size from it"
    assert kwargs == {}


def test_the_heatmap_mode_asks_the_loader_for_the_prompted_arguments(prompted, monkeypatch):
    """The pipeline and the crop bounds reach the loader, in that order."""
    prompted["path"] = "~/data"
    asked = {}

    def fake_loader(path, x, y, pipeline_id, start=None, end=None):
        asked.update(path=path, x=x, y=y, pipeline_id=pipeline_id, start=start, end=end)
        return _cube()

    monkeypatch.setattr(plot_raman, "get_cumulative_hsi_cube", fake_loader)
    prompted["prompt_args"]("heatmap", {"pipeline_id": 2, "start": 200.0, "end": 800.0})

    plot_raman.main()

    assert asked == {"path": "~/data", "x": 2, "y": 3, "pipeline_id": 2, "start": 200.0, "end": 800.0}


def test_the_phase_mode_passes_its_arguments_positionally(prompted, monkeypatch):
    """The raw cube, then the phase count, then the crop bounds."""
    prompted["path"] = "~/data"
    raw = object()
    monkeypatch.setattr(plot_raman, "get_raw_hsi_cube", lambda *a, **k: raw)
    prompted["prompt_args"]("phase_decomp", {"phase_number": -1, "start": 300.0, "end": 700.0})

    plot_raman.main()

    kind, args, _ = prompted["viewer"]
    assert kind == "phases"
    assert args == (raw, -1, 300.0, 700.0)


def test_the_predict_mode_passes_the_run_and_the_threshold_as_keywords(prompted, monkeypatch):
    """The run directory is positional, the threshold and the save path are keywords."""
    prompted["path"] = "~/data"
    prompted["prompt_args"]("predict", {"run_path": "/runs/sample", "confidence_threshold": 0.5, "save_path": None})

    plot_raman.main()

    kind, args, kwargs = prompted["viewer"]
    assert kind == "predict"
    assert args == ("/runs/sample", "~/data", 2, 3)
    assert kwargs == {"confidence_threshold": 0.5, "save_path": None}
