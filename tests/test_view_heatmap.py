"""The heatmap viewer: which pixel a click means, and that it can be built at all.

The viewer is an interactive figure, so most of it is matplotlib plumbing that a test
would only be re-stating. Two things are worth pinning: the mapping from a pointer
position to a pixel number, which is the part a user actually reads off the figure, and
the fact that the figure can be constructed from a cube, which is what every change to
the cube's shape has to keep working.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data.loader import Cube
from src.visualisation import view_heatmap

ROWS, COLS, BANDS = 2, 3, 4


@pytest.fixture
def cube():
    """A 2x3 grid whose pixel numbers are not derivable from the position, so a mix-up shows.

    The bands are scaled per pixel, so the area between two bands differs from pixel to
    pixel. A cube whose pixels all look alike collapses the intensity slider's range to a
    single value, which matplotlib warns about and nothing in the figure then depends on.
    """
    bands = np.linspace(0.0, 1.0, BANDS) ** 2
    per_pixel = np.arange(1, ROWS * COLS + 1, dtype=float).reshape(ROWS, COLS)
    return Cube(
        cube=np.einsum("b,rc->rcb", bands, per_pixel),
        spectra=np.arange((ROWS * COLS + 1) * BANDS, dtype=float).reshape(ROWS * COLS + 1, BANDS),
        axis=np.linspace(100, 900, BANDS),
        pixel_map=np.array([[1, 2, 3], [4, 5, 6]]),
    )


def test_a_click_maps_to_the_pixel_under_it(cube):
    """Data coordinates are pixel centres, so a whole number lands on that pixel."""
    assert view_heatmap.pixel_at(0.0, 0.0, cube.pixel_map) == 1
    assert view_heatmap.pixel_at(2.0, 1.0, cube.pixel_map) == 6


def test_a_click_rounds_to_the_nearest_pixel_centre(cube):
    """A pointer is continuous, the grid is not: over half way means the next pixel."""
    assert view_heatmap.pixel_at(0.4, 0.4, cube.pixel_map) == 1
    assert view_heatmap.pixel_at(0.6, 0.6, cube.pixel_map) == 5


def test_a_click_outside_the_image_is_clamped_to_the_edge(cube):
    """``imshow`` leaves margin inside the axes, so the pointer can report coordinates past the image."""
    assert view_heatmap.pixel_at(99.0, 99.0, cube.pixel_map) == 6
    assert view_heatmap.pixel_at(-4.0, -4.0, cube.pixel_map) == 1


def test_the_clamping_uses_the_grid_it_was_given(cube):
    """A grid that is not square catches a bound taken from the wrong axis.

    Reading rows and columns the wrong way round works on a square fixture, which is what
    the real data is, so this one is deliberately 2 by 3.
    """
    assert view_heatmap.pixel_at(2.4, 1.4, cube.pixel_map) == 6, "the last column, last row"
    assert view_heatmap.pixel_at(2.4, 0.0, cube.pixel_map) == 3, "the last column, first row"


def test_the_viewer_can_be_built_from_a_cube(cube, monkeypatch):
    """Constructing the figure is the part every change to the cube has to keep working.

    ``plt.show`` is replaced, so the figure is built and then discarded rather than opened.
    """
    monkeypatch.setattr(view_heatmap.plt, "show", lambda: None)

    view_heatmap.show_hsi_viewer(cube)

    figure = view_heatmap.plt.gcf()
    assert len(figure.axes) >= 4, "the image, the two spectrum panels and the sliders"
    view_heatmap.plt.close(figure)
