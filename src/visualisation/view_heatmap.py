import logging

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from .theme import *
from matplotlib.widgets import Slider, TextBox, RangeSlider, Button

logger = logging.getLogger(__name__)


def _apply_intensity_mask(image, i_min, i_max):
    """Applies intensity mask and greys out values outside the range.

    Args:
        image: 2D array of AUC values for a single band.
        i_min: Minimum intensity value.
        i_max: Maximum intensity value.

    Returns:
        RGBA image array with out-of-range pixels greyed out.
    """
    cmap = plt.get_cmap("magma")
    norm = Normalize(vmin=i_min, vmax=i_max, clip=True)
    rgba = cmap(norm(image))

    outside = (image < i_min) | (image > i_max)
    rgba[outside] = [0.15, 0.15, 0.18, 1]
    return rgba


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
        spans, which the viewer keeps to describe the window it is showing.

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


def pixel_at(xdata, ydata, pixel_map):
    """The pixel number at a data coordinate on the image.

    Matplotlib reports coordinates outside the drawn image when the pointer is inside the
    axes but in the margin ``imshow`` leaves, and a coordinate can land half a pixel past
    the last one, so the position is rounded and then clamped to the grid rather than
    trusted. Rounding before clamping is what makes a click on the right half of the last
    pixel report that pixel rather than one past it.
    """
    rows, cols = pixel_map.shape
    col = np.clip(int(xdata + 0.5), 0, cols - 1)
    row = np.clip(int(ydata + 0.5), 0, rows - 1)
    return pixel_map[row, col]


def show_hsi_viewer(cube):
    """Shows a cube as an image with the spectrum under the pointer, and the window under it.

    Args:
        cube (Cube): The cumulative cube, the spectra behind it, their axis, and the pixel
            map. The grid size comes from the pixel map rather than from a second pair of
            arguments that could disagree with it.
    """
    apply_theme()

    # set the initial window width to 10 cm
    initial_window_width_cm = 10

    raman_shift_arr = np.asarray(cube.axis)
    spectral_range  = raman_shift_arr[-1] - raman_shift_arr[0]
    mean_step       = float(np.mean(np.diff(raman_shift_arr))) 

    # initialise the current state using a dicitonary (fast look up times)
    state = {
        "ln_scale":    False,
        "auc_cube":    None,
        "band_starts": None,
        "band_ends":   None,
        "idx_step":    None,
        "i_min":       None,
        "i_max":       None,
        "correction":  None,
        "log_v_min":   None,
        "log_v_max":   None,
    }

    def _recompute_auc(window_width_cm):
        """Rebuilds the windowed cube, and the intensity range that describes it.

        Args:
            window_width_cm (float): The width the window slider is now on.

        Raises:
            ValueError: The window is wider than the spectral range, from
                ``query_rolling_auc``. Nothing is stored in that case, so the viewer is
                left holding the cube and the scale it had.
        """
        auc_cube, band_starts, band_ends, idx_step = query_rolling_auc(
            cube.cube, raman_shift_arr, window_width_cm
        )
        # update relevant states
        state["auc_cube"]    = auc_cube
        state["band_starts"] = band_starts
        state["band_ends"]   = band_ends
        state["idx_step"]    = idx_step

        # update intensity slider min and max
        i_min = float(np.min(auc_cube))
        i_max = float(np.max(auc_cube))
        
        # re-compute the correction term based on the new width
        correction = abs(min(i_min, 0))
        auc_corrected = auc_cube + correction + 1e-10
        state["i_min"]      = i_min
        state["i_max"]      = i_max
        state["correction"] = correction
        state["log_v_min"]  = float(np.log(np.min(auc_corrected)))
        state["log_v_max"]  = float(np.log(np.max(auc_corrected)))

    _recompute_auc(initial_window_width_cm)

    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor(BG)

    # plots
    ax_image       = fig.add_axes([0.09, 0.56, 0.65, 0.40])
    ax_spectrum    = fig.add_axes([0.09, 0.33, 0.65, 0.18])
    ax_ln_spectrum = fig.add_axes([0.09, 0.07, 0.65, 0.18])

    # raman shift slider
    ax_pos_slider   = fig.add_axes([0.09, 0.015, 0.65, 0.030])
    ax_width_slider = fig.add_axes([0.09, 0.052, 0.65, 0.030])

    # intensity slider
    ax_intensity_slider = fig.add_axes([0.80, 0.32, 0.03, 0.63])

    # switch scale button and pixel text box
    ax_button           = fig.add_axes([0.80, 0.22, 0.16, 0.06])
    ax_box              = fig.add_axes([0.80, 0.14, 0.16, 0.05])

    for ax in fig.axes:
        ax.set_facecolor((1, 0, 0, 0.05))

    ax_image.set_title("Raman Image")
    ax_image.set_axis_off()
    ax_spectrum.set_title("Intensity Spectra")
    ax_spectrum.set_xlabel(r"Raman Shift cm$^{-1}$")
    ax_spectrum.set_ylabel("Intensity")
    ax_ln_spectrum.set_title("Intensity Spectra (ln scale)")
    ax_ln_spectrum.set_xlabel(r"Raman Shift cm$^{-1}$")
    ax_ln_spectrum.set_ylabel("ln(Intensity)")

    # correct intensity range
    if state["correction"] > 0:
        logger.info("Intensity correction applied: +%.4f to shift all values positive", state["correction"])

    # initial plots
    image = ax_image.imshow(
        _apply_intensity_mask(state["auc_cube"][:, :, 0], state["i_min"], state["i_max"]),
        aspect="equal", origin="upper"
    )
    first_spectrum       = cube.spectra[cube.pixel_map[0, 0]]
    (pixel_spectrum,)    = ax_spectrum.plot(raman_shift_arr, first_spectrum, color=ACCENT)
    (pixel_ln_spectrum,) = ax_ln_spectrum.plot(
        raman_shift_arr,
        np.log(np.maximum(first_spectrum + state["correction"] + 1e-10, 1e-10)),
        color=ACCENT
    )

    scalar_mappable = cm.ScalarMappable(norm=Normalize(state["i_min"], state["i_max"]), cmap="magma")
    scalar_mappable.set_array([])
    cbar = fig.colorbar(scalar_mappable, ax=ax_image)
    cbar.set_ticks(np.linspace(state["i_min"], state["i_max"], 5))
    cbar.ax.yaxis.set_tick_params(color=FG, labelsize=8)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=FG)

    # lines show the current window start/end on both spectrum panels
    lower_limit_line    = ax_spectrum.axvline(state["band_starts"][0], color="#E07B54", linestyle="--", alpha=0.7)
    upper_limit_line    = ax_spectrum.axvline(state["band_ends"][0],   color="#E07B54", linestyle="--", alpha=0.7)
    lower_ln_limit_line = ax_ln_spectrum.axvline(state["band_starts"][0], color="#E07B54", linestyle="--", alpha=0.7)
    upper_ln_limit_line = ax_ln_spectrum.axvline(state["band_ends"][0],   color="#E07B54", linestyle="--", alpha=0.7)

    # widgets
    n_bands_init = state["auc_cube"].shape[-1]
    band_indices = np.arange(n_bands_init)

    pos_slider = Slider(
        ax=ax_pos_slider, label="Raman Shift",
        valmin=0, valmax=n_bands_init - 1,
        valinit=0, valstep=band_indices,
    )
    pos_slider.label.set_color(FG)
    pos_slider.valtext.set_color(FG)
    pos_slider.track.set_color(SLIDER_TRACK)
    pos_slider.poly.set_color(SLIDER_ACTIVE)
    pos_slider.valtext.set_text(
        f"{state['band_starts'][0]:.0f}–{state['band_ends'][0]:.0f} cm⁻¹"
    )
    ax_pos_slider.set_facecolor(WIDGET_PANEL)

    width_slider = Slider(
        ax=ax_width_slider, label="Window (cm⁻¹)",
        valmin=2 * mean_step,
        valmax=spectral_range * 0.5,   # cap at half the full range
        valinit=initial_window_width_cm,
        valstep=mean_step,
    )
    width_slider.label.set_color(FG)
    width_slider.valtext.set_color(FG)
    width_slider.track.set_color(SLIDER_TRACK)
    width_slider.poly.set_color(SLIDER_ACTIVE)
    ax_width_slider.set_facecolor(WIDGET_PANEL)

    intensity_slider = RangeSlider(
        ax=ax_intensity_slider, label="Intensity",
        valmin=state["i_min"], valmax=state["i_max"],
        valinit=(state["i_min"], state["i_max"]),
        orientation="vertical",
    )
    intensity_slider.label.set_color(FG)
    intensity_slider.valtext.set_color(FG)
    intensity_slider.track.set_color(SLIDER_TRACK)
    intensity_slider.poly.set_color(SLIDER_ACTIVE)
    ax_intensity_slider.set_facecolor(WIDGET_PANEL)

    scale_button = Button(ax_button, "Switch to ln scale", color=BUTTON_COLOR, hovercolor=BUTTON_HOVER)
    scale_button.label.set_color(FG)
    scale_button.label.set_fontsize(8)
    ax_button.set_facecolor(WIDGET_PANEL)

    text_box = TextBox(
        ax_box, "Pixel:",
        textalignment="center",
        color=WIDGET_SURFACE,
        hovercolor=WIDGET_EDGE,
    )
    text_box.label.set_color(FG)
    text_box.text_disp.set_color(FG)
    text_box.set_val(str(cube.pixel_map[0, 0]))
    ax_box.set_facecolor(WIDGET_PANEL)

    hover_text = ax_image.text(
        0.01, 0.99, "", transform=ax_image.transAxes,
        va="top", ha="left", color=FG, fontsize=8,
        bbox=dict(boxstyle="round,pad=0.2", facecolor=GRID, edgecolor=FG, alpha=0.85),
    )

    fig.canvas.draw()
    background = fig.canvas.copy_from_bbox(fig.bbox)

    # helpers
    def _get_image_range():
        """The intensity range to draw in, in plain intensity units.

        The slider carries ln values while the ln scale is on, so its ends are turned
        back into intensities by undoing the log and the correction that was added before
        it. The image and the colour bar are in plain units either way.
        """
        v_min, v_max = intensity_slider.val
        if state["ln_scale"]:
            return np.exp(v_min) - state["correction"], np.exp(v_max) - state["correction"]
        return v_min, v_max

    def _update_pos_slider_range():
        """Clamp position slider to the new number of bands after a width change."""
        n_bands = state["auc_cube"].shape[-1]
        new_indices = np.arange(n_bands)
        pos_slider.valmax = n_bands - 1
        pos_slider.valstep = new_indices
        pos_slider.ax.set_xlim(0, n_bands - 1)
        # clamp current value
        clamped = int(min(pos_slider.val, n_bands - 1))
        pos_slider.set_val(clamped)

    # callbacks
    def update(_val):
        """Draws the band the position slider is on, and the spectrum of the chosen pixel.

        Both the slider and the text box that names a pixel call this, so the argument
        matplotlib passes is ignored and the current contents of both widgets are read
        instead. A text box holding something that is not a number leaves the display as
        it was, so a half-typed pixel number does not blank the spectrum.
        """
        index = int(pos_slider.val)
        index = min(index, state["auc_cube"].shape[-1] - 1)

        try:
            pixel = int(text_box.text)
        except ValueError:
            return

        i_min_cur, i_max_cur = _get_image_range()

        image.set_data(_apply_intensity_mask(state["auc_cube"][:, :, index], i_min_cur, i_max_cur))
        scalar_mappable.set_clim(i_min_cur, i_max_cur)
        cbar.set_ticks(np.linspace(i_min_cur, i_max_cur, 5))
        cbar.update_normal(scalar_mappable)

        spectrum = cube.spectra[pixel]
        pixel_spectrum.set_ydata(spectrum)
        pixel_ln_spectrum.set_ydata(
            np.log(np.maximum(spectrum + state["correction"] + 1e-10, 1e-10))
        )

        x_start = state["band_starts"][index]
        x_end   = state["band_ends"][index]
        for line in [lower_limit_line, lower_ln_limit_line]:
            line.set_xdata([x_start, x_start])
        for line in [upper_limit_line, upper_ln_limit_line]:
            line.set_xdata([x_end, x_end])
        pos_slider.valtext.set_text(f"{x_start:.0f}–{x_end:.0f} cm⁻¹")

        ax_spectrum.relim()
        ax_spectrum.autoscale_view()
        ax_ln_spectrum.relim()
        ax_ln_spectrum.autoscale_view()

        fig.canvas.restore_region(background)
        for ax in [ax_image, ax_spectrum, ax_ln_spectrum]:
            ax.redraw_in_frame()
        fig.canvas.blit(fig.bbox)

    def on_width_changed(_val):
        """Recompute auc_cube for the new window width, then reset intensity scale."""
        try:
            _recompute_auc(width_slider.val)
        except ValueError as e:
            # A window the intensity range cannot be recomputed for is recoverable: the
            # cube keeps the scale it had and the slider stays where the user left it.
            # Logged rather than printed, because a caller reading stdout has no way to
            # tell this apart from output it asked for.
            logger.warning("Could not recompute the intensity range: %s", e)
            return

        # reset intensity slider to cover the new cube's full range
        new_i_min = state["i_min"]
        new_i_max = state["i_max"]
        if state["ln_scale"]:
            new_lo = state["log_v_min"]
            new_hi = state["log_v_max"]
        else:
            new_lo, new_hi = new_i_min, new_i_max

        intensity_slider.valmin = new_lo
        intensity_slider.valmax = new_hi
        intensity_slider.ax.set_ylim(new_lo, new_hi)
        intensity_slider.set_val((new_lo, new_hi))

        _update_pos_slider_range()
        update(None)

    def on_resize(_event):
        """Retakes the copy of the canvas that ``update`` blits over.

        Redrawing only the axes on top of a saved copy of the canvas is what keeps a
        slider drag from redrawing the whole figure, and the copy is only valid at the
        size it was taken at. A resize redraws the canvas, so the stale copy has to be
        replaced or the saved region no longer matches the figure under it.
        """
        nonlocal background
        fig.canvas.draw()
        background = fig.canvas.copy_from_bbox(fig.bbox)

    def on_scale_toggle(_event):
        """Switches the intensity slider between plain and ln scale.

        The two ends are converted so the same intensities stay in view across the
        switch, and the slider's bounds become the range that belongs to the scale it is
        now on. The ln scale is there because the areas span orders of magnitude; the
        correction is what keeps an area of zero or below off the log.
        """
        v_min_cur, v_max_cur = intensity_slider.val
        state["ln_scale"] = not state["ln_scale"]

        if state["ln_scale"]:
            new_min = np.log(max(v_min_cur + state["correction"] + 1e-10, 1e-10))
            new_max = np.log(max(v_max_cur + state["correction"] + 1e-10, 1e-10))
            intensity_slider.ax.set_ylim(state["log_v_min"], state["log_v_max"])
            intensity_slider.valmin = state["log_v_min"]
            intensity_slider.valmax = state["log_v_max"]
            intensity_slider.label.set_text("ln(Intensity)")
            scale_button.label.set_text("Switch to standard scale")
        else:
            new_min = np.exp(v_min_cur) - state["correction"]
            new_max = np.exp(v_max_cur) - state["correction"]
            intensity_slider.ax.set_ylim(state["i_min"], state["i_max"])
            intensity_slider.valmin = state["i_min"]
            intensity_slider.valmax = state["i_max"]
            intensity_slider.label.set_text("Intensity")
            scale_button.label.set_text("Switch to ln scale")

        intensity_slider.set_val((new_min, new_max))
        fig.canvas.draw_idle()

    def on_hover(event):
        """Names the pixel the pointer is over, and clears the name off the image.

        The label is only rewritten when its text changes, because a redraw per mouse
        move would otherwise be paid for on every pixel the pointer crosses while it
        stays within one pixel of the image.
        """
        new_text = ""
        if event.inaxes == ax_image:
            new_text = f"Pixel {pixel_at(event.xdata, event.ydata, cube.pixel_map)}"
        if hover_text.get_text() != new_text:
            hover_text.set_text(new_text)
            fig.canvas.draw_idle()

    def on_click(event):
        """Moves the spectrum to the pixel that was double-clicked on the image.

        A single click does nothing: panning and zooming the image also send click
        events, and following those would move the spectrum off the pixel being looked
        at. The position slider is left alone, so the band on screen does not change.
        """
        if event.inaxes != ax_image or not event.dblclick:
            return
        pixel = pixel_at(event.xdata, event.ydata, cube.pixel_map)
        text_box.set_val(str(pixel))

        spectrum = cube.spectra[pixel]
        pixel_spectrum.set_ydata(spectrum)
        pixel_ln_spectrum.set_ydata(
            np.log(np.maximum(spectrum + state["correction"] + 1e-10, 1e-10))
        )
        ax_spectrum.relim()
        ax_spectrum.autoscale_view()
        ax_ln_spectrum.relim()
        ax_ln_spectrum.autoscale_view()
        fig.canvas.draw_idle()

    # -------------------------------------------------------- updates
    text_box.on_submit(update)
    pos_slider.on_changed(update)
    width_slider.on_changed(on_width_changed)
    intensity_slider.on_changed(update)
    scale_button.on_clicked(on_scale_toggle)
    fig.canvas.mpl_connect("resize_event", on_resize)
    fig.canvas.mpl_connect("motion_notify_event", on_hover)
    fig.canvas.mpl_connect("button_press_event", on_click)

    plt.show()