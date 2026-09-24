"""Synthetic data generator for the test suite.

Writes a grid of ``.txt`` files in the format the loaders expect: two tab-separated
columns (Raman shift, intensity) with no header, named ``FullGrid[n](-x, -y, -z).txt``.

Nothing here is read from real data. The cube holds three known phases - each a sum
of Gaussians at fixed Raman shifts - laid out across the grid, so the phase
estimator has a defensible right answer to be wrong about.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROWS = 4
COLS = 4

RAMAN_SHIFT_START = 200.0
RAMAN_SHIFT_STOP = 1200.0
RAMAN_SHIFT_STEP = 1.0

# Fixed on purpose. The fixture is input data rather than a computed result, so it
# must be identical on every machine for a snapshot comparison to mean anything.
FIXTURE_SEED = 20260924

# (centre cm^-1, width cm^-1, height) - three distinguishable phases.
PHASES: dict[int, list[tuple[float, float, float]]] = {
    0: [(320.0, 12.0, 1.0), (640.0, 20.0, 0.6)],
    1: [(510.0, 15.0, 1.0), (900.0, 25.0, 0.7)],
    2: [(420.0, 10.0, 0.9), (1080.0, 30.0, 0.8), (760.0, 18.0, 0.5)],
}

N_PHASES = len(PHASES)


def spectral_axis() -> np.ndarray:
    return np.arange(RAMAN_SHIFT_START, RAMAN_SHIFT_STOP + RAMAN_SHIFT_STEP, RAMAN_SHIFT_STEP)


def phase_at(row: int, col: int) -> int:
    """Which phase a pixel shows. Deterministic, so the layout is reproducible."""
    return (row + col) % N_PHASES


def spectrum_for(axis: np.ndarray, phase: int, rng: np.random.Generator) -> np.ndarray:
    """A sum of Gaussians plus a little noise, clipped at zero like a real spectrum."""
    y = np.zeros_like(axis, dtype=float)
    for centre, width, height in PHASES[phase]:
        y += height * np.exp(-0.5 * ((axis - centre) / width) ** 2)
    y += rng.normal(0.0, 0.01, size=axis.size)
    return np.clip(y, 0.0, None)


def write_grid(directory: Path, rows: int = ROWS, cols: int = COLS) -> Path:
    """Write ``rows * cols`` spectra into ``directory`` and return it."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    axis = spectral_axis()
    rng = np.random.default_rng(FIXTURE_SEED)

    n = 0
    for row in range(rows):
        for col in range(cols):
            n += 1
            y = spectrum_for(axis, phase_at(row, col), rng)
            name = f"FullGrid[{n}](-{18000 + n}, -{320 + n}, -{14675 + n}).txt"
            np.savetxt(directory / name, np.column_stack([axis, y]), delimiter="\t", fmt="%.6f")

    return directory


def expected_pixel_map(rows: int = ROWS, cols: int = COLS) -> np.ndarray:
    """The pixel numbering ``RamanGrid.traverse`` is expected to produce.

    Start at the bottom-left corner, walk right along the row, step up, walk left,
    and so on. Returns an ``(rows, cols)`` array of 1-based pixel numbers. This is an
    independent re-implementation of the traversal rather than a call into it, so the
    two can be compared; if they ever disagree, one of them is wrong.
    """
    pixel_map = np.zeros((rows, cols), dtype=int)
    pixel = 1
    cur_x, cur_y, step = rows - 1, 0, 1
    for _ in range(rows * cols):
        pixel_map[cur_x, cur_y] = pixel
        pixel += 1
        if cur_y == cols - 1 and step != -1:  # right edge: move up, reverse
            cur_x -= 1
            step *= -1
        elif cur_y == 0 and step != 1:  # left edge: move up, reverse
            cur_x -= 1
            step *= -1
        else:
            cur_y += step
    return pixel_map
