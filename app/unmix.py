"""The unmixing a screen asks for, and what it runs with.

The decomposition itself is not reimplemented here. ``rp.analysis.decompose.NMF``
and ``estimate_phase_number`` are the framework's own, and they are called with the
settings and the fixed preprocessing the command line has always used, so an
unmixing asked for in the browser decomposes the same cube the same way.

What is here is the little that the browser needs around them: the settings as a
mapping the screen can print instead of a second copy of the numbers in the front
end, the count resolved from the screen's choice of estimating it or fixing it, and
the estimator's findings turned into plain data. A ``PhaseEstimate`` holds
dataclasses, which the job's state file cannot carry, and every method's vote is
kept, rejections and their reasons included, because the reason a method was
discarded is the part a reader of the screen needs.
"""

from __future__ import annotations

import logging
from dataclasses import fields

import ramanspy as rp

from src.analysis.phase_number_est import PhaseEstimate, estimate_phase_number
from src.config import EstimatorConfig, NmfConfig

logger = logging.getLogger(__name__)

#: The preprocessing an unmixing runs behind, in the order it runs it, named as the
#: screen shows them. The names are here as well as in :func:`pipeline` so the card
#: that says what will run cannot drift from what does.
PIPELINE_STEPS = ("WhitakerHayes", "ASLS", "MinMax")

#: The fewest components a decomposition can be asked for. One component is a
#: decomposition of a kind; none is not, so the screen's stepper stops here.
COUNT_MIN = 1


def pipeline():
    """The fixed preprocessing an unmixing runs behind.

    Smoothing is deliberately not among the steps: a smoothed spectrum distorts the
    baseline the step after it is fitting.

    Returns:
        ramanspy Pipeline: The three steps, in the order they are applied.
    """
    return rp.preprocessing.Pipeline([
        rp.preprocessing.despike.WhitakerHayes(),
        rp.preprocessing.baseline.ASLS(),
        rp.preprocessing.normalise.MinMax(),
    ])


def settings() -> dict:
    """The settings a decomposition runs with, as the screen shows them.

    Read from the config classes rather than written out again in the front end, so a
    value changed in one of them reaches the screen without a second place to change.

    Returns:
        dict: The ``NmfConfig`` fields in the order they are declared, the steps of the
        fixed pipeline, the smallest count that can be asked for, and how many estimates
        have to survive before the estimator will name a count at all.
    """
    config = NmfConfig()
    return {
        "settings": {field.name: getattr(config, field.name) for field in fields(config)},
        "pipeline": list(PIPELINE_STEPS),
        "count_min": COUNT_MIN,
        "min_votes": EstimatorConfig().min_votes,
    }


def crop(cube, start, end):
    """The cube narrowed to a wavenumber range.

    Args:
        cube: The cube to narrow.
        start, end: The bounds to keep. Either may be None, which keeps that side whole.

    Returns:
        The cropped cube, or the cube itself when neither bound was given. Cropping to
        the whole range would copy the cube for nothing.
    """
    if start is None and end is None:
        return cube
    return rp.preprocessing.misc.Cropper(region=(start, end)).apply(cube)


def resolve_count(cube, count) -> tuple[int | None, PhaseEstimate | None]:
    """How many components to decompose into, and the estimate when one was taken.

    A count the caller gives is used as given, with nothing checking that the data
    supports it. ``None`` means estimate it, which is what ``-1`` means on the command
    line. When the estimator names no count there is nothing to decompose with, so the
    count comes back as None and the estimate is the whole result rather than one that
    was rounded up to something to show.

    Args:
        cube: The cube to estimate from.
        count (int or None): The count the caller fixed, or None to estimate it.

    Returns:
        tuple: The count to decompose with, which is None when the estimator abstained,
        and the estimate as it came back, which is None when the caller fixed a count.
    """
    if count is not None:
        return int(count), None

    estimate = estimate_phase_number(cube)
    return estimate.n, estimate


def decomposition(cube, count: int):
    """Decomposes a cube into end-member spectra and their abundance maps.

    The preprocessing runs here rather than on the cube the caller holds, because the
    decomposition's own pipeline is not the heatmap screen's: the same map read with a
    pipeline on that screen is not a reason for this one to unmix preprocessed data.

    Args:
        cube: The cube to decompose, with no preprocessing on it.
        count (int): How many components to look for.

    Returns:
        tuple: One abundance map per component, shaped like the grid, and one end-member
        spectrum per component.
    """
    nmf = rp.analysis.decompose.NMF(**NmfConfig().as_kwargs(count))
    return nmf.apply(pipeline().apply(cube))


def report(estimate: PhaseEstimate) -> dict:
    """The estimator's findings as the screen draws them.

    Every method is listed, including the ones that contributed nothing, since a reader
    seeing only the number that survived cannot tell a method that agreed from one that
    was thrown out for disagreeing. How many of them counted is reported beside how many
    ran, because that pair is what decides whether there is a count at all.

    Args:
        estimate (PhaseEstimate): The estimate to report.

    Returns:
        dict: The count, which is None when the methods did not agree on one, the
        confidence, every method's vote with its rejection reason, and the two counts the
        rule was applied with.
    """
    counted = [vote for vote in estimate.votes if vote.votes]
    return {
        "n": estimate.n,
        "confidence": estimate.confidence,
        "votes": [
            {
                "name": vote.name,
                "value": vote.value,
                "rejected": vote.rejected,
                "counted": vote.votes,
            }
            for vote in estimate.votes
        ],
        "counted": len(counted),
        "methods": len(estimate.votes),
        "min_votes": EstimatorConfig().min_votes,
    }
