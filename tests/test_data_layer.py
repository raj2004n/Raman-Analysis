"""Golden snapshots for the data layer: ``src/data/grid.py`` and ``src/data/loader.py``.

These pin the numbers the loaders produce today, so a later change to file
discovery, traversal order, or the shape of what a load returns can be shown to
have preserved them. The fixture is 4x4 with three phases laid out by
``(row + col) % 3``, which makes the traversal order a meaningful claim rather
than a coincidence.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data.grid import RamanGrid
from src.data.loader import get_cumulative_hsi_cube, get_raw_hsi_cube
from tests import synthetic

# Pipeline 0 is raw data; pipeline 1 is the first preprocessing protocol. The other
# two protocols are exercised the same way but add runtime without adding coverage
# of anything the first two do not already touch.
PIPELINES_SNAPSHOTTED = (0, 1)


def test_traversal_follows_the_documented_route(grid_dir, golden):
    """``RamanGrid.traverse`` walks the bottom row, steps up, and walks back."""
    grid = RamanGrid(grid_dir, synthetic.ROWS, synthetic.COLS)
    route = [(row, col) for _, row, col in grid.traverse()]

    golden.check("grid.traversal_route", np.array(route))
    golden.check("grid.sorted_file_count", len(grid.get_sorted_files()))

    _, _, _, pixel_map = get_cumulative_hsi_cube(grid_dir, synthetic.ROWS, synthetic.COLS, 0)
    golden.check("grid.pixel_map", pixel_map)

    assert np.array_equal(pixel_map, synthetic.expected_pixel_map()), (
        "traversal no longer matches the boustrophedon route the loaders assume: "
        f"got {pixel_map.tolist()}, expected {synthetic.expected_pixel_map().tolist()}"
    )


def test_raw_cube_shape_and_axis(grid_dir, golden, raw_cube):
    golden.check("data.raw_cube.shape", np.array(raw_cube.spectral_data.shape))
    golden.check("data.raw_cube.spectral_axis", np.asarray(raw_cube.spectral_axis))
    golden.check("data.raw_cube.intensities", np.asarray(raw_cube.spectral_data))

    assert raw_cube.spectral_data.shape == (
        synthetic.ROWS,
        synthetic.COLS,
        synthetic.spectral_axis().size,
    )


def test_cumulative_cube_snapshots(grid_dir, golden):
    """The four values ``get_cumulative_hsi_cube`` returns, per pipeline."""
    for pipeline_id in PIPELINES_SNAPSHOTTED:
        auc_cube, spectra_list, cropped_axis, pixel_map = get_cumulative_hsi_cube(
            grid_dir, synthetic.ROWS, synthetic.COLS, pipeline_id
        )
        prefix = f"data.cumulative.p{pipeline_id}"
        golden.check(f"{prefix}.auc_cube", auc_cube)
        golden.check(f"{prefix}.spectra_list", spectra_list)
        golden.check(f"{prefix}.cropped_axis", cropped_axis)
        golden.check(f"{prefix}.pixel_map", pixel_map)


def test_cropping_narrows_the_axis(grid_dir, golden):
    """A cropped run must land on the same axis positions as an uncropped one."""
    for label, start, end in (("full", None, None), ("window", 400.0, 900.0)):
        auc_cube, _, cropped_axis, _ = get_cumulative_hsi_cube(
            grid_dir, synthetic.ROWS, synthetic.COLS, 0, start, end
        )
        golden.check(f"data.crop.{label}.cropped_axis", cropped_axis)
        golden.check(f"data.crop.{label}.auc_cube", auc_cube)


def test_raw_cube_matches_the_snapshot(grid_dir, golden):
    cube = get_raw_hsi_cube(grid_dir, synthetic.ROWS, synthetic.COLS)
    golden.check("data.raw_cube.default", np.asarray(cube.spectral_data))


# --------------------------------------------------------------------------------------
# File discovery. A data directory is written by an instrument, so it can pick up a notes
# file or lose a spectrum, and neither should end in a stack trace from the sort call.
# --------------------------------------------------------------------------------------


def test_an_unnumbered_file_is_ignored(tmp_path):
    """A stray .txt next to a complete grid is skipped, not fatal.

    Sorting the files by pixel number used to compare None with int and raise TypeError
    as soon as one filename had no [n] in it, which made a notes file in the sample
    directory break every mode.
    """
    directory = synthetic.write_grid(tmp_path / "grid", 2, 2)
    (directory / "readme_backup.txt").write_text("notes about this sample\n")

    files = RamanGrid(directory, 2, 2).get_sorted_files()

    assert len(files) == 4
    assert "readme_backup.txt" not in [file.name for file in files]


def test_an_unnumbered_file_is_named_when_the_grid_is_incomplete(tmp_path):
    """The stray file is reported, since it may be the spectrum that is missing."""
    directory = synthetic.write_grid(tmp_path / "grid", 2, 2)
    (directory / "FullGrid[4](-18004, -324, -14679).txt").unlink()
    (directory / "readme_backup.txt").write_text("notes about this sample\n")

    with pytest.raises(ValueError) as raised:
        RamanGrid(directory, 2, 2).get_sorted_files()

    message = str(raised.value)
    assert "readme_backup.txt" in message
    assert "found 3" in message


def test_a_gap_in_the_pixel_numbers_is_an_error(tmp_path):
    """Numbers 1, 2, 4 cannot be laid out on a 2x2 grid, so this stops rather than guesses."""
    directory = synthetic.write_grid(tmp_path / "grid", 2, 2)
    (directory / "FullGrid[3](-18003, -323, -14678).txt").unlink()

    with pytest.raises(ValueError, match="Pixel numbers with no file"):
        RamanGrid(directory, 2, 2).get_sorted_files()


def test_a_repeated_pixel_number_is_an_error(tmp_path):
    """Two files claiming pixel 3 would shift every pixel after them by one position."""
    directory = synthetic.write_grid(tmp_path / "grid", 2, 2)
    source = directory / "FullGrid[3](-18003, -323, -14678).txt"
    (directory / "FullGrid[3](-18009, -329, -14684).txt").write_text(source.read_text())

    with pytest.raises(ValueError, match="more than one file"):
        RamanGrid(directory, 2, 2).get_sorted_files()


def test_a_directory_with_no_numbered_files_says_so(tmp_path):
    """Only unnumbered files present: the error names them instead of the sort failing."""
    directory = tmp_path / "grid"
    directory.mkdir()
    (directory / "readme_backup.txt").write_text("notes about this sample\n")

    with pytest.raises(FileNotFoundError, match="readme_backup.txt"):
        RamanGrid(directory, 2, 2).get_sorted_files()
