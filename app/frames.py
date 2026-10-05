"""What the heatmap draws, worked out from the session already in memory.

The band image and the two spectra are the parts of the heatmap the pointer touches
constantly, so they are sent as arrays and drawn in the browser rather than rendered
as a picture. A slider move is then a message of a few kilobytes instead of a new
figure, and the band follows the slider without a pause.

The windowing is not reimplemented here. ``query_rolling_auc`` is the one place it is
written down: it turns a width in cm⁻¹ into a number of bands with a floor of two and
a cap the caller passes, and each band of the result is one subtraction between two
bands of the cumulative cube. Keeping the screen on that function is what makes a
window of 10 cm⁻¹ mean the same thing wherever it is measured, down to which bands it
spans.
"""

from __future__ import annotations

import logging

import numpy as np

from src.analysis.rolling_auc import query_rolling_auc

logger = logging.getLogger(__name__)

#: The windowing of the last width that was asked for, kept because a slider move
#: changes the band but not the width. Without it every move would rebuild the whole
#: windowed cube to slice one band out of it.
_CACHE: dict = {"key": None, "auc": None, "starts": None, "ends": None, "step": None}


def _window(cube, width_cm: float, session_key):
    """The windowed cube for a width, rebuilt only when the width or map changes.

    Args:
        cube (Cube): The session's cube, its axis and its spectra.
        width_cm (float): The window width in cm⁻¹.
        session_key: Anything that changes when a different map is loaded, so a width
            that happens to match the last one does not reuse another map's bands.

    Returns:
        tuple: The areas, the wavenumber each window starts at, the wavenumber each
        ends at, and the number of bands the window spans.
    """
    key = (session_key, float(width_cm))
    if _CACHE["key"] != key:
        auc, starts, ends, step = query_rolling_auc(cube.cube, cube.axis, width_cm)
        _CACHE.update(key=key, auc=auc, starts=starts, ends=ends, step=step)
    return _CACHE["auc"], _CACHE["starts"], _CACHE["ends"], _CACHE["step"]


def _limits(auc_cube):
    """The intensity range the band image is drawn in, and the log range beside it.

    The areas span orders of magnitude and some of them can be negative, so the log
    range is taken over the areas shifted by whatever it takes to bring the smallest
    of them above zero. Both ranges are reported: the slider works in whichever scale
    it is showing, and the image is drawn in plain units either way.
    """
    i_min = float(np.min(auc_cube))
    i_max = float(np.max(auc_cube))
    correction = abs(min(i_min, 0.0))
    corrected = auc_cube + correction + 1e-10
    return {
        "i_min": i_min,
        "i_max": i_max,
        "correction": correction,
        "log_min": float(np.log(np.min(corrected))),
        "log_max": float(np.log(np.max(corrected))),
    }


def frame(cube, session_key, center: int, width_cm: float, pixel: int) -> dict:
    """Everything the heatmap needs to draw one band and one pixel's spectrum.

    Args:
        cube (Cube): The session's cube. Its arrays are memory-mapped, so this reads
            the bands it needs and no more.
        session_key: What identifies the map and how it was derived, so a cached
            window is not reused across a re-derive.
        center (int): Which band of the windowed cube to draw. Clamped, because a
            narrower window leaves fewer bands than a wider one and a slider that was
            at the end of the old range would otherwise point past the end.
        width_cm (float): The window width in cm⁻¹.
        pixel (int): Which pixel's spectrum to send. Clamped to the map.

    Returns:
        dict: The band as an array of areas, both ends of the window it covers, the
        spectrum of the chosen pixel, the axis, and the ranges the controls work in.

    Raises:
        ValueError: The window is wider than the spectral range, from
            ``query_rolling_auc``. The message is the code's own, since the way out is
            to ask for a smaller window and the message says what the current one is.
    """
    axis = np.asarray(cube.axis)
    auc_cube, starts, ends, step = _window(cube, width_cm, session_key)

    index = int(np.clip(center, 0, auc_cube.shape[-1] - 1))
    pixel = int(np.clip(pixel, 1, cube.spectra.shape[0] - 1))

    mean_step = float(np.mean(np.diff(axis)))
    spectral_range = float(axis[-1] - axis[0])

    return {
        "rows": int(auc_cube.shape[0]),
        "cols": int(auc_cube.shape[1]),
        "axis": [float(value) for value in axis],
        "band": np.asarray(auc_cube[:, :, index], dtype=float).tolist(),
        # Which pixel is at which position, so the browser can turn a point on the
        # image back into a pixel number the same way the viewer does. It is a property
        # of the grid shape rather than of the derivation, but it is small enough to
        # send with the band rather than cached separately.
        "pixel_map": np.asarray(cube.pixel_map, dtype=int).tolist(),
        "index": index,
        "bands": int(auc_cube.shape[-1]),
        "window_start": float(starts[index]),
        "window_end": float(ends[index]),
        "window_bands": int(step),
        "width_cm": float(width_cm),
        "spectrum": np.asarray(cube.spectra[pixel], dtype=float).tolist(),
        "pixel": pixel,
        "limits": _limits(auc_cube),
        "mean_step": mean_step,
        "range": spectral_range,
        "window_min": 2 * mean_step,
        "window_max": spectral_range * 0.5,
    }
