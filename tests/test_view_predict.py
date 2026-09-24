"""The prediction viewer's hover annotation.

The annotation indexed the label and confidence arrays before testing the row and
column against the map bounds, and those indices come from rounding the pointer
position, which lands on exactly ``width`` or ``length`` along the far edges. So
hovering the right or bottom edge raised IndexError, and the range check on the next
line could not help because the damage was already done.

The decision is now a plain function, which is what makes these tests possible: the
callback that used it needed a loaded model and an open figure window.
"""

from __future__ import annotations

import numpy as np

from src.visualisation.view_predict import _hover_annotation

THRESHOLD = 0.80


def _viewer_state(labels):
    """The three arrays the annotation reads, built from a grid of mineral names."""
    masked_labels = np.array(labels, dtype=object)
    confidence_map = np.full(masked_labels.shape, 0.95)
    predicted_top5_map = np.array(
        [[[("Quartz", 0.95), ("Calcite", 0.02)] for _ in row] for row in labels],
        dtype=object,
    )
    return masked_labels, confidence_map, predicted_top5_map


def test_hovering_the_far_edges_returns_nothing():
    """Rounding up to exactly the width or the length is the case that used to raise."""
    state = _viewer_state([["Quartz", "Calcite"], ["Calcite", "Quartz"]])

    # The two edges the pointer can round onto, and the two it can round past.
    assert _hover_annotation(0, 2, *state, THRESHOLD) is None
    assert _hover_annotation(2, 0, *state, THRESHOLD) is None
    assert _hover_annotation(2, 2, *state, THRESHOLD) is None

    # Negative indices would wrap around in numpy rather than raise, so they are
    # bounded by the same check.
    assert _hover_annotation(-1, 0, *state, THRESHOLD) is None
    assert _hover_annotation(0, -1, *state, THRESHOLD) is None


def test_a_supported_pixel_returns_its_labels():
    state = _viewer_state([["Quartz", "Calcite"], ["Calcite", "Quartz"]])

    lines, mineral = _hover_annotation(1, 0, *state, THRESHOLD)

    assert mineral == "Calcite"
    assert lines[0] == "Pixel (1, 0)"
    assert lines[1] == ""
    assert lines[2] == "1. Quartz: 95.0%"
    assert lines[3] == "2. Calcite: 2.0%"


def test_a_low_confidence_pixel_reads_as_unknown():
    """Below the threshold the annotation says so instead of showing the top five."""
    masked_labels, confidence_map, predicted_top5_map = _viewer_state(
        [["Quartz", "Calcite"], ["Calcite", "Quartz"]]
    )
    masked_labels[0, 0] = "Unknown"
    confidence_map[0, 0] = 0.42

    lines, mineral = _hover_annotation(
        0, 0, masked_labels, confidence_map, predicted_top5_map, THRESHOLD
    )

    assert mineral == "Unknown"
    assert "Confidence: 42.0% (below threshold)" in lines
