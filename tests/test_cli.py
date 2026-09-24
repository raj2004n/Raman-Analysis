"""The command line hands each viewer what it needs, and the loaders what they ask for.

The three modes are thin, but they are where the loaders and the viewers meet, and the
heatmap mode is where the four values of a cube used to be spread across an unpacking
statement. Most of these tests replace ``prompt_args`` so no input is read, and replace
the viewers so no window opens, which leaves the wiring itself as the thing under test.
The last two let the prompt loop read real piped input instead, since the parsing is
only covered by exercising it.
"""

from __future__ import annotations

import io

import numpy as np
import pytest

from scripts import plot_raman
from src.analysis.phase_number_est import PhaseEstimate
from src.data.loader import Cube
from src.visualisation import view_heatmap, view_phase_decomp
from tests import synthetic


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


# --------------------------------------------------------------------------------------
# Whole modes, with the answers piped in rather than the prompts replaced. Everything
# below the prompt is the real thing: the loader, the cube it builds, and the viewer.
# --------------------------------------------------------------------------------------


def _piped(*lines):
    """The answers a user would type, one per prompt, as the prompt loop would read them."""
    return io.StringIO("".join(f"{line}\n" for line in lines))


def test_the_heatmap_mode_runs_end_to_end_from_piped_answers(grid_dir, monkeypatch):
    """Mode 1 through the prompts, with the figure built from the cube the loader returned."""
    monkeypatch.setattr(view_heatmap.plt, "show", lambda: None)
    monkeypatch.setattr("sys.stdin", _piped(
        f"{grid_dir} {synthetic.ROWS} {synthetic.COLS}", 1, 1, 400.0, 900.0
    ))

    plot_raman.main()

    figure = view_heatmap.plt.gcf()
    assert len(figure.axes) >= 4, "the image, the two spectrum panels and the sliders"
    view_heatmap.plt.close(figure)


def test_an_undetermined_estimate_reaches_the_user_from_piped_answers(grid_dir, monkeypatch, capsys):
    """Mode 2 at ``-1`` when the estimate has no count, all the way from the prompts.

    The estimator is the one thing replaced, with the answer the rule gives when
    nothing survives, since a fixture that happens to be undetermined is not something
    to rely on. The prompts, the loader and the viewer are the real ones, and the
    decomposition is booby-trapped: the path being checked is the one that stops before
    it, and it stops whatever count it would have been given.
    """

    def no_unmixing(**kwargs):
        raise AssertionError(f"the decomposition was built with {kwargs}")

    monkeypatch.setattr(
        view_phase_decomp, "estimate_phase_number",
        lambda cube: PhaseEstimate(None, "undetermined", ()),
    )
    monkeypatch.setattr(view_phase_decomp.rp.analysis.decompose, "NMF", no_unmixing)
    monkeypatch.setattr("sys.stdin", _piped(
        f"{grid_dir} {synthetic.ROWS} {synthetic.COLS}", 2, -1, "", ""
    ))

    plot_raman.main()

    assert "Nothing to unmix without a phase count." in capsys.readouterr().out
