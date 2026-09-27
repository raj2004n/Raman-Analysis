"""What the prediction screen needs around the model rather than from it.

The predictions themselves are the framework's own: the job calls the same
``predict`` the command line calls, so the browser cannot classify a spectrum
differently. What is here is everything else the screen shows: the runs on disk and
what each of them recorded, the confidence mask that turns a best guess below the
threshold into ``Unknown``, and the figure.

Two things are arranged differently from the viewer rather than computed
differently. The legend is drawn in the page rather than into the figure, so that
the notes it carries about ``Unknown`` and about the minerals the model has but the
map does not can sit with it, and the five most likely minerals reach the page as
data so the hover card is built without a request per move of the pointer. Both come
from the same arrays the viewer draws its tooltip from.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import to_hex

from src.paths import output_path

from app import figures

#: The confidence the command line offers for the threshold, and the one the slider
#: starts at. Below it a pixel reads ``Unknown``.
DEFAULT_THRESHOLD = 0.80

#: What a pixel reads when its best guess is not confident enough to name a mineral.
UNKNOWN = "Unknown"

#: The grey ``Unknown`` pixels are drawn in. Deliberately not from the colormap: a pixel
#: that reads ``Unknown`` is the absence of an answer rather than another mineral, and it
#: is the same grey the heatmap paints an out-of-range area in.
UNKNOWN_COLOUR = (0.15, 0.15, 0.18, 1.0)

#: The file a job's per-pixel predictions are written to, and the figure it draws.
PREDICTIONS_NAME = "predictions.json"
MAP_NAME = "prediction-map.png"

#: What the command line calls the file it saves, and so what the screen offers.
EXPORT_NAME = "prediction_map.png"

#: Each row of the facts card, and how it reads a manifest. A reader that answers None
#: is one a run may never have recorded, which the card shows rather than leaving out.
FACT_READERS = (
    ("input", lambda manifest: f"{manifest.input_size} points per spectrum"),
    ("classes", lambda manifest: f"{manifest.num_classes} minerals"),
    ("range", lambda manifest: f"{manifest.x_min:.1f} – {manifest.x_max:.1f} cm⁻¹"),
    ("metrics", lambda manifest: _metrics(manifest.metrics)),
    ("dataset", lambda manifest: _dataset(manifest)),
    ("seed", lambda manifest: None if manifest.seed is None else str(manifest.seed)),
)


def _metrics(metrics) -> str | None:
    """What the model scored, as the run recorded it, or None if it recorded nothing."""
    if not metrics:
        return None
    return ", ".join(
        f"{name} {value:.3f}" if isinstance(value, (int, float)) else f"{name} {value}"
        for name, value in metrics.items()
    )


def _dataset(manifest) -> str | None:
    """The dataset a run was trained on, with enough of its digest to tell it apart.

    The digest is truncated because the whole of it is sixty-four characters of hex that
    nobody compares by eye. What it is for is telling two datasets of the same name apart,
    and a few characters do that.
    """
    if not manifest.dataset:
        return None
    name = Path(manifest.dataset).name
    if not manifest.dataset_hash:
        return name
    return f"{name} · {manifest.dataset_hash[:11]}…"


def facts(manifest) -> list[dict]:
    """What a run recorded, as the rows the facts card lists.

    Every row is present whether or not the run recorded it. Fields were added to the
    manifest after runs had already been written, and a manifest written before one
    existed still loads: showing the row as not recorded is what keeps a run that never
    had a fact apart from a run that was trained without one.

    Args:
        manifest (RunManifest | None): The run's manifest, or None when the file could not
            be read as one.

    Returns:
        list[dict]: One ``label`` and ``value`` per row, with ``value`` None where the run
        recorded nothing for it.
    """
    rows = []
    for label, read in FACT_READERS:
        value = None
        if manifest is not None:
            value = read(manifest)
        rows.append({"label": label, "value": value})
    return rows


def catalogue() -> dict:
    """The runs on disk, newest first, and where they were looked for.

    The directory comes back with the list so that an empty one can say where it looked.
    A person whose runs are somewhere else is otherwise looking at a page that says there
    are none and nothing about where it decided that.

    A manifest is read as JSON here rather than through ``load_run``, which would load
    every model in the list to answer a question about what they were trained with.

    Returns:
        dict: ``runs`` as entries holding a name, a directory and its facts rows,
        ``directory`` naming where runs are looked for, and ``threshold`` as the confidence
        the slider starts at, so the screen's default is the one the code uses.
    """
    # Imported here rather than at the top of the module: ``src.cnn.runs`` imports Keras,
    # and a server whose Predict screen is never opened should not carry a deep learning
    # framework in memory to answer questions about the other three screens.
    from src.cnn import runs

    entries = []
    for name in runs.list_runs():
        directory = runs.run_directory(name)
        try:
            manifest = runs.RunManifest.from_json(
                (directory / runs.MANIFEST_NAME).read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError, TypeError, ValueError, KeyError):
            # A run.json edited by hand, or left half-written by a run that was killed.
            # The directory is still a run, and nothing is claimed about what it recorded.
            manifest = None
        entries.append({
            "name": name,
            "directory": str(directory),
            "classes": None if manifest is None else int(manifest.num_classes),
            "facts": facts(manifest),
        })

    return {
        "runs": entries,
        "directory": str(runs.runs_root()),
        "default_threshold": DEFAULT_THRESHOLD,
    }


def resolve_run(name: str) -> Path:
    """The run directory a name or a path from the picker names.

    Args:
        name (str): A run name from the catalogue, or a path to a run directory. The
            command line offers both, so the screen does too.

    Returns:
        Path: The run directory.

    Raises:
        FileNotFoundError: Nothing at that name or path holds a manifest, so it is not a
            run. Refused here rather than inside a job, because a job that would only
            spend a process finding out has nothing to report that this does not.
    """
    from src.cnn import runs

    candidate = Path(name).expanduser()
    if not candidate.is_dir():
        candidate = runs.run_directory(name)
    if not (candidate / runs.MANIFEST_NAME).is_file():
        raise FileNotFoundError(
            f"{candidate} holds no {runs.MANIFEST_NAME}, so it is not a run directory."
        )
    return candidate


def confidence(labels, top5) -> np.ndarray:
    """The probability of each pixel's best guess, which is what the threshold reads.

    Read from the top five rather than from an array of its own, because that is where the
    viewer reads it: the first entry of a pixel's list is its top prediction.
    """
    return np.array([[float(entry[0][1]) for entry in row] for row in top5])


def apply_threshold(labels, top5, threshold: float) -> dict:
    """The minerals drawn at a threshold, and the colours the legend names them in.

    A pixel whose best guess falls below the threshold reads ``Unknown`` rather than as
    that mineral. That is the whole effect of the slider: the predictions are already
    computed, so moving it changes what is drawn and nothing else.

    The colours follow the viewer's rule, assigned in sorted order of the minerals present
    rather than from a fixed table, because the set on the map changes as the threshold
    moves and the legend has to be rebuilt with it. Only minerals this map holds are given
    a colour, so the legend is shorter than the model's class list. A map of three holds
    three far-apart colours rather than three neighbouring ones.

    Args:
        labels (np.ndarray): The best guess per pixel, as ``predict`` returns it.
        top5 (np.ndarray): The five most likely per pixel, as ``predict`` returns it.
        threshold (float): Below this, a pixel reads ``Unknown``.

    Returns:
        dict: ``labels`` masked to the threshold, ``minerals`` in the order they are
        drawn, ``colours`` as RGBA in that order, and ``index`` giving each pixel the
        position of its colour, which is what the figure is drawn from.
    """
    masked = np.array(labels, dtype=object).copy()
    masked[confidence(labels, top5) < threshold] = UNKNOWN

    present = np.unique(masked)
    minerals = sorted(str(name) for name in present if name != UNKNOWN)
    if UNKNOWN in present:
        minerals.append(UNKNOWN)

    sampled = [
        colormaps["tab20"](position / max(len(minerals) - 1, 1))
        for position in range(len(minerals))
    ]
    if minerals and minerals[-1] == UNKNOWN:
        sampled[-1] = UNKNOWN_COLOUR

    positions = {name: position for position, name in enumerate(minerals)}
    return {
        "labels": masked,
        "minerals": minerals,
        "colours": sampled,
        "index": np.vectorize(positions.get)(masked),
    }


def legend(drawn: dict) -> list[dict]:
    """The mineral rows the page draws beside the map.

    Names and hex colours rather than the viewer's figure-level patches, because the
    legend is drawn in the page here so that the notes about ``Unknown`` and about the
    minerals the model has but the map does not can sit with it.
    """
    return [
        {"name": name, "colour": to_hex(colour)}
        for name, colour in zip(drawn["minerals"], drawn["colours"])
    ]


def save_predictions(directory, labels, top5, run: str) -> Path:
    """Writes the per-pixel predictions beside the job's other files.

    Written as JSON because a mineral name is not a number: the labels are strings and the
    pairs are strings with probabilities, and an array of those is an array of objects that
    only a pickled file could round-trip. What reads it back is either the page, which
    wants text, or the re-render, which wants the same two arrays.

    Args:
        directory (Path): The job's own directory.
        labels (np.ndarray): The best guess per pixel.
        top5 (np.ndarray): The five most likely per pixel.
        run (str): The run the prediction was made with, recorded so the file says which
            model produced it.

    Returns:
        Path: Where the table was written.
    """
    payload = {
        "run": str(run),
        "labels": [[str(label) for label in row] for row in labels],
        "top5": [
            [[[str(label), float(probability)] for label, probability in entry] for entry in row]
            for row in top5
        ],
    }
    path = Path(directory) / PREDICTIONS_NAME
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def load_predictions(directory) -> dict:
    """Reads back the table ``save_predictions`` wrote.

    Raises:
        FileNotFoundError: The job has no table, which is what a job that refused the run
            leaves behind, since it had nothing to predict.
    """
    return json.loads((Path(directory) / PREDICTIONS_NAME).read_text(encoding="utf-8"))


def _stored(directory, threshold: float) -> dict:
    """The job's predictions, masked to a threshold."""
    stored = load_predictions(directory)
    return apply_threshold(
        np.array(stored["labels"], dtype=object),
        np.array(stored["top5"], dtype=object),
        threshold,
    )


def render(directory, threshold: float, name: str = MAP_NAME) -> dict:
    """Draws the mineral map at a threshold, into the job's own directory.

    No model runs here. The predictions were computed when the job ran, and the threshold
    only decides which of them are drawn as ``Unknown``, which is what makes the slider
    something to move rather than something to answer once.

    Args:
        directory (Path): The job's directory, holding the predictions.
        threshold (float): Below this, a pixel is drawn as ``Unknown``.
        name (str): The file name to write inside that directory.

    Returns:
        dict: The figure's name, the threshold it was drawn at, and the legend rows the
        page draws beside it.
    """
    drawn = _stored(directory, threshold)
    entries = legend(drawn)
    # No title and no legend in the figure: the page's slider says which threshold is
    # being drawn, and its legend carries the notes this one cannot. No margin either,
    # so the file is the map edge to edge and the page can turn a pointer position into
    # a pixel by measuring against it.
    figures.prediction_png(drawn["index"], drawn["colours"], Path(directory) / name, pad_inches=0)
    return {"figure": name, "threshold": float(threshold), "minerals": entries}


def export(directory, threshold: float, name: str = EXPORT_NAME) -> Path:
    """Writes the map as a file that can be read away from the screen.

    Drawn at the size, and with the title and the legend, that the command line saves it
    with, because a file under ``outputs`` is looked at on its own: a bare map with no key
    would be a picture of colours, and the threshold it was drawn at is the one thing
    about it that cannot be seen.

    Args:
        directory (Path): The job's directory, holding the predictions.
        threshold (float): Below this, a pixel is drawn as ``Unknown``.
        name (str): The file name under ``outputs``.

    Returns:
        Path: Where the figure was written.
    """
    drawn = _stored(directory, threshold)
    path = output_path(name)
    figures.prediction_png(
        drawn["index"], drawn["colours"], path,
        threshold=threshold, legend=drawn["minerals"], size=(12, 8),
    )
    return path
