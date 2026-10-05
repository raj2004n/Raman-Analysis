"""The areas under a spectrum over a sliding window, for every position of that window.

This is the measurement the browser's Heatmap screen is built on, and the one place it is
written down: a width in cm⁻¹ becomes a number of bands, and each position's area is one
subtraction between two bands of a cumulative cube. The matplotlib figure that used to be
drawn on top of it is retired, and its code is kept in ``legacy/``.
"""

import numpy as np


def query_rolling_auc(cumulative_cube, raman_shift, window_width_cm):
    """The area under every spectrum over a window, for every position of that window.

    The window slides one band at a time. Its width arrives in cm⁻¹ and becomes a number
    of bands, never fewer than two, so a window narrower than the spacing between bands
    still spans two of them. Each window's area is then one subtraction between two bands
    of the cumulative cube rather than a fresh integral.

    Args:
        cumulative_cube: The cumulative integral of each spectrum, shaped
            (rows, columns, bands).
        raman_shift: The wavenumbers, one per band.
        window_width_cm (float): The window width in cm⁻¹.

    Returns:
        tuple: The areas, shaped (rows, columns, bands - step); the wavenumber each
        window starts at; the wavenumber each ends at; and the number of bands the window
        spans, which the screens keep to describe the window they are showing.

    Raises:
        ValueError: The window is wider than the spectral range, so no position of it
            falls inside the data. The message gives both widths, since the caller's way
            out is to ask for a smaller window.
    """
    spectral_axis = np.asarray(raman_shift) # as array for future operations
    n = len(spectral_axis)
    mean_step = np.mean(np.diff(spectral_axis))
    idx_step = max(2, int(window_width_cm // mean_step)) # corresponds to window width in the units of index

    # corresponds to the number of available shifts
    n_bands = n - idx_step
    if n_bands <= 0: # if <= 0 rolling window is too wide
        raise ValueError(
            f"Window width {window_width_cm:.1f} cm⁻¹ exceeds the spectral range "
            f"({spectral_axis[-1] - spectral_axis[0]:.1f} cm⁻¹). Choose a smaller window."
        )

    # get area under cruve by subtracting the cumulative values
    auc_cube = cumulative_cube[:, :, idx_step:] - cumulative_cube[:, :, :n_bands]

    band_starts = spectral_axis[:n_bands]
    band_ends   = spectral_axis[idx_step:]

    return auc_cube, band_starts, band_ends, idx_step
