"""Figures drawn on the server and written as files.

The band image and the two spectra are drawn in the browser, because the pointer
touches them constantly and a request per slider move would be felt. What is left is
what a person asks for once and then looks at: the end-member spectra, the abundance
maps, the prediction map, and the heatmap exported as a picture. Those are drawn here
with matplotlib and written as PNGs.

The heatmap export is built from the same payload the screen draws from, so the file
and the screen show the same band, the same window and the same pixel rather than two
nearly equal renderings of the same state. The unmixing's two panels are drawn through
RamanSPy's own plotting calls, which are the ones the command line draws them with, and
written into the job's directory rather than into ``outputs``: they are one job's
result, and nothing but that job's own page asks for them. The prediction map is drawn
both ways, once into the job's directory for the page and once into ``outputs`` when it
is exported, where it carries the title and the legend a file read on its own needs.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import ramanspy as rp
from matplotlib import colormaps
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patches import Patch

from src.paths import output_path
from src.visualisation.theme import ACCENT, BG, FG, apply_theme

from app import frames

logger = logging.getLogger(__name__)

#: The colormap the maps are drawn in, and the one the browser draws them in too.
CMAP = "magma"

#: What an area outside the intensity range is drawn as, instead of being clipped to
#: the end of the colormap. Clipping would show an out-of-range pixel as the brightest
#: one in the map, which is the opposite of what it is.
OUTSIDE = [0.15, 0.15, 0.18, 1.0]

#: Where the two dashed lines marking the window are drawn on a spectrum.
WINDOW_LINE = "#E07B54"

#: The title a prediction figure carries when it names its own threshold, which an
#: exported file does: read away from the screen, the threshold is the one thing about
#: the picture that cannot be seen.
THRESHOLD_TITLE = "Mineral Prediction Map (confidence threshold: {percent:.0f}%)"

#: The file names one unmixing's panels are written under, in the job's own directory.
ENDMEMBER_NAME = "endmembers.png"
ABUNDANCE_NAME = "abundance-{component}.png"


def masked_band(band, i_min, i_max):
    """The band as an RGBA image, with everything outside the range greyed.

    Args:
        band (np.ndarray): Areas, shaped (rows, cols).
        i_min, i_max (float): The intensity range the map is drawn in.

    Returns:
        np.ndarray: The image, shaped (rows, cols, 4).
    """
    image = colormaps[CMAP](Normalize(vmin=i_min, vmax=i_max, clip=True)(band))
    image[(band < i_min) | (band > i_max)] = OUTSIDE
    return image


def heatmap_png(cube, session_key, center: int, width_cm: float, pixel: int,
                i_min: float | None = None, i_max: float | None = None,
                name: str = "heatmap.png") -> Path:
    """Draws the heatmap as it stands and writes it beside the other outputs.

    Args:
        cube (Cube): The session's cube.
        session_key: What identifies the map and its derivation, for the window cache.
        center (int): Which band is on screen.
        width_cm (float): The window width in cm⁻¹.
        pixel (int): The pixel whose spectrum is drawn.
        i_min, i_max (float or None): The intensity range the screen is drawing the map
            in, so the file carries the picture that was on screen rather than the whole
            range the data happens to span. Left out, the range is the data's own.
        name (str): The file name under ``outputs/``.

    Returns:
        Path: Where the figure was written.

    Raises:
        ValueError: The window is wider than the spectral range, from ``frames``.
    """
    apply_theme()
    data = frames.frame(cube, session_key, center, width_cm, pixel)

    band = np.asarray(data["band"])
    axis = np.asarray(data["axis"])
    spectrum = np.asarray(data["spectrum"])
    limits = data["limits"]

    if i_min is None or i_max is None:
        i_min, i_max = limits["i_min"], limits["i_max"]

    fig = plt.figure(figsize=(12.5, 9))
    fig.patch.set_facecolor(BG)
    ax_image = fig.add_axes([0.07, 0.55, 0.55, 0.38])
    ax_spectrum = fig.add_axes([0.07, 0.31, 0.55, 0.17])
    ax_ln = fig.add_axes([0.07, 0.07, 0.55, 0.17])

    ax_image.set_title(
        f"{data['window_start']:.0f} – {data['window_end']:.0f} cm⁻¹ "
        f"(window {width_cm:.0f} cm⁻¹, {data['window_bands']} bands)"
    )
    ax_image.set_axis_off()
    ax_image.imshow(
        masked_band(band, i_min, i_max),
        aspect="equal", origin="upper", interpolation="nearest",
    )

    # The colour bar is built from a scalar mappable rather than from the image, because
    # the image is already RGBA: there is no scalar in it for a bar to be drawn from, and
    # one taken from it would come out in whichever colormap is the default rather than in
    # the one the picture is drawn in.
    mappable = ScalarMappable(norm=Normalize(vmin=i_min, vmax=i_max), cmap=CMAP)
    mappable.set_array([])
    colour_bar = fig.colorbar(mappable, ax=ax_image, fraction=0.046)
    colour_bar.set_ticks(np.linspace(i_min, i_max, 5))
    colour_bar.set_label("Area under the spectrum", color=FG, fontsize=9)
    colour_bar.ax.yaxis.set_tick_params(color=FG, labelsize=8)
    plt.setp(colour_bar.ax.yaxis.get_ticklabels(), color=FG)

    for ax, values, label in (
        (ax_spectrum, spectrum, "Intensity"),
        (ax_ln, np.log(np.maximum(spectrum + limits["correction"] + 1e-10, 1e-10)), "ln(Intensity)"),
    ):
        ax.plot(axis, values, color=ACCENT, linewidth=1)
        for edge in (data["window_start"], data["window_end"]):
            ax.axvline(edge, color=WINDOW_LINE, linestyle="--", alpha=0.7)
        ax.set_ylabel(label)
        ax.set_xlim(axis[0], axis[-1])

    ax_spectrum.set_title(f"Intensity Spectra, pixel {pixel}")
    ax_ln.set_title("Intensity Spectra (ln scale)")
    ax_ln.set_xlabel(r"Raman Shift cm$^{-1}$")

    path = output_path(name)
    fig.savefig(path, dpi=110, facecolor=BG)
    plt.close(fig)
    logger.info("Wrote %s.", path)
    return path


def unmix_pngs(phases, axis, abundance_maps, directory) -> dict:
    """Draws an unmixing's end-member spectra and its abundance maps.

    One file per panel rather than one figure holding all of them: the screen lays the
    maps out in a grid of its own, and a panel then sizes in the page like any other
    image. The viewer's rule for how many go in a row, a square grid up to ten and one
    per row beyond that, is a layout rather than a drawing, so it is the screen's.

    Each map is drawn scaled to its own component, which is what the viewer's default
    does and what its colour bar states. Two panels are therefore not comparable by
    colour alone, which the screen says beside them.

    Args:
        phases (list): One end-member spectrum per component.
        axis (np.ndarray): The wavenumbers the spectra sit on.
        abundance_maps (list): One abundance map per component, shaped like the grid.
        directory (Path): The job's own directory, which the files are written into.

    Returns:
        dict: The name of the spectra figure under ``spectra``, and one name per component
        under ``maps``, in component order. Names rather than full paths, because they are
        what the page asks the server for and a path would carry the server's directories
        into the browser.
    """
    apply_theme()
    directory.mkdir(parents=True, exist_ok=True)

    # The colours are given at the call rather than left to RamanSPy's default, which asks
    # matplotlib for a colormap through a name that a current matplotlib removed. The
    # registry is the documented replacement, and one line is drawn per phase, so one
    # colour each is what the default produced here.
    colours = list(colormaps[CMAP](np.linspace(0.15, 0.85, len(phases))))
    ax = rp.plot.spectra(
        phases, axis,
        plot_type="single stacked",
        label=[f"Spectra {index + 1}" for index in range(len(phases))],
        color=colours,
    )
    ax.figure.savefig(directory / ENDMEMBER_NAME, dpi=110, facecolor=BG)
    plt.close(ax.figure)

    maps = []
    for component, abundance_map in enumerate(abundance_maps, start=1):
        fig, panel = plt.subplots(figsize=(4, 4))
        fig.patch.set_facecolor(BG)
        rp.plot.image(abundance_map, title=f"Spectra {component}", cmap=CMAP, ax=panel)
        name = ABUNDANCE_NAME.format(component=component)
        fig.savefig(directory / name, dpi=110, facecolor=BG)
        plt.close(fig)
        maps.append(name)

    logger.info("Wrote %s and %s maps.", ENDMEMBER_NAME, len(maps))
    return {"spectra": ENDMEMBER_NAME, "maps": maps}


def prediction_png(index, colours, path, threshold=None, legend=None, size=(6, 6),
                   pad_inches=None) -> Path:
    """Draws the predicted mineral per pixel, one colour each.

    The colours are handed in already assigned rather than chosen here, because which
    mineral gets which colour depends on the set the map holds, and that set is what the
    legend beside the figure lists. Both come from the same list, so a swatch and a pixel
    cannot disagree.

    Args:
        index (np.ndarray): The position in ``colours`` of each pixel's mineral, shaped
            like the grid.
        colours (list): One RGBA per mineral, in the order the index refers to them.
        path (Path): Where to write the figure. Its directory is created.
        threshold (float, optional): Named in a title when given. Left out for the copy
            the page shows, whose slider already states it.
        legend (list, optional): Mineral names to label the colours with, in the same
            order. Left out for the copy the page shows, which draws its own legend.
        size (tuple): The figure's size in inches.
        pad_inches (float, optional): Margin kept around the figure. Left out, matplotlib's
            own default applies, which is what an exported file wants so that a legend
            outside the axes is not cropped. The page's copy passes zero, because the page
            turns a pointer position into a pixel by measuring against the image and a
            border of no pixels of the map would be counted as if it were one.

    Returns:
        Path: Where the figure was written.
    """
    apply_theme()
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, panel = plt.subplots(figsize=size)
    fig.patch.set_facecolor(BG)
    panel.set_axis_off()
    # A single colour would leave matplotlib a range of one to scale, so the top of the
    # range is at least one whatever the map holds. Only the first colour is used either
    # way, and a map of one mineral is a map of one colour.
    panel.imshow(
        index,
        cmap=ListedColormap(colours),
        vmin=0,
        vmax=max(len(colours) - 1, 1),
        interpolation="nearest",
    )

    if threshold is not None:
        panel.set_title(THRESHOLD_TITLE.format(percent=threshold * 100))

    if legend:
        panel.legend(
            handles=[Patch(facecolor=colour, label=name) for colour, name in zip(colours, legend)],
            bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8,
            title="Minerals", title_fontsize=9, framealpha=0.9,
        )

    save = {"dpi": 150, "facecolor": BG, "bbox_inches": "tight"}
    if pad_inches is not None:
        save["pad_inches"] = pad_inches
    fig.savefig(path, **save)
    plt.close(fig)
    logger.info("Wrote %s.", path)
    return path
