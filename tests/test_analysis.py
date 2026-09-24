"""Golden snapshots and characterisation tests for ``src/analysis/phase_number_est.py``.

Two jobs here:

* Pin the numbers the estimator produces today, so splitting it into pure method
  functions plus a combination rule can be shown to preserve them.
* Record the current aggregation behaviour in executable form. The rejected-estimate
  test below asserts today's result on purpose, so that changing the rule is a
  deliberate edit to a failing test rather than a silent change of output.

The method values are read by calling the private helpers directly rather than by
parsing stdout, because the module's prints are expected to be replaced with logging
later and these snapshots should not depend on them.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pytest
import ramanspy as rp

from src.analysis import phase_number_est
from src.config import NmfConfig
from tests import synthetic

N_COMPONENTS = 3


def _preprocessed(cube):
    """The same two-step pipeline ``estimate_phase_number`` applies before estimating."""
    pipeline = rp.preprocessing.Pipeline(
        [
            rp.preprocessing.despike.WhitakerHayes(),
            rp.preprocessing.baseline.ASLS(),
        ]
    )
    return pipeline.apply(cube)


def _quietly(callable_, *args, **kwargs):
    """``estimate_phase_number`` prints 19 lines. Capture them so the run stays readable."""
    with contextlib.redirect_stdout(io.StringIO()):
        return callable_(*args, **kwargs)


@pytest.fixture(scope="module")
def prepared(raw_cube):
    return _preprocessed(raw_cube)


def test_phase_estimate_is_pinned(raw_cube, golden):
    """The headline number: how many distinct spectra the estimator reports."""
    np.random.seed(synthetic.FIXTURE_SEED)
    predicted_n, confidence = _quietly(phase_number_est.estimate_phase_number, raw_cube)

    golden.check("analysis.phase_estimate.predicted_n", predicted_n)
    golden.check("analysis.phase_estimate.confidence", confidence)


def test_each_method_is_pinned(prepared, golden):
    """The three methods separately, so a change in the rule is distinguishable from a change in a method."""
    np.random.seed(synthetic.FIXTURE_SEED)
    n_80, n_elbow = phase_number_est._n_by_pca(prepared)
    ns_hfc = phase_number_est._n_by_hfc(prepared)

    golden.check("analysis.method.pca_80", n_80)
    golden.check("analysis.method.pca_elbow", n_elbow)
    golden.check("analysis.method.nw_hfc", np.asarray(ns_hfc))


def test_confidence_rule_is_pinned():
    """A pure function, so this is real unit coverage rather than a snapshot."""
    assert phase_number_est._determine_confidence((3, 3, 3)) == "high"
    assert phase_number_est._determine_confidence((3, 3, 1)) == "moderate"
    assert phase_number_est._determine_confidence((4, 3, 1)) == "low"

    # Fewer than three estimates cannot be 'high', however much they agree, and a
    # single estimate has nothing to agree with.
    assert phase_number_est._determine_confidence((3, 3)) == "moderate"
    assert phase_number_est._determine_confidence((3, 4)) == "low"
    assert phase_number_est._determine_confidence((3,)) == "low"


def test_missing_elbow_does_not_crash_and_does_not_vote(monkeypatch, raw_cube):
    """``kneed`` returns None when the variance curve has no knee at all.

    That None used to reach the ``> 10`` rejection check on the next line and raise
    ``TypeError``, which is the default path for the viewer and the CLI when the phase
    count is left at ``-1``.

    The method now contributes no estimate rather than an estimate of 1, so two
    methods agreeing on 3 report ``moderate``: unanimity between two is weaker
    evidence than unanimity between three.
    """
    monkeypatch.setattr(phase_number_est, "_n_by_pca", lambda cube: (3, None))
    monkeypatch.setattr(phase_number_est, "_n_by_hfc", lambda cube: np.array([3, 3, 3]))

    predicted_n, confidence = _quietly(phase_number_est.estimate_phase_number, raw_cube)

    assert predicted_n == 3
    assert confidence == "moderate", (
        "a missing elbow leaves two estimates that agree, which must not be reported "
        "with the same confidence as three that agree"
    )


def test_rejected_estimate_is_counted_as_a_vote(monkeypatch, raw_cube, golden):
    """Characterises the current combination rule around the internal threshold of 10.

    When an estimate exceeds 10 it is overwritten with ``1`` (``:93`` and ``:100``), and
    that ``1`` is then carried into the combination at ``:122`` alongside the values that
    were not rejected. So a rejected estimate is not excluded: it is rewritten and then
    counted, and it pulls the reported confidence down with it.

    The three methods are stubbed here so that two of them agree on 3 while the third
    returns 17 and is rejected. All methods genuinely agree on a value of 3, yet the
    result is reported as ``moderate`` rather than ``high``, which is the effect the
    rejected value has on the answer.
    """
    monkeypatch.setattr(phase_number_est, "_n_by_pca", lambda cube: (17, 3))
    monkeypatch.setattr(phase_number_est, "_n_by_hfc", lambda cube: np.array([3, 3, 3]))

    predicted_n, confidence = _quietly(phase_number_est.estimate_phase_number, raw_cube)

    golden.check("analysis.rejected_estimate.predicted_n", predicted_n)
    golden.check("analysis.rejected_estimate.confidence", confidence)

    assert predicted_n == 3
    assert confidence == "moderate", (
        "the rejected estimate is currently counted as a vote of 1, so three methods "
        "that agree on 3 still report only moderate confidence"
    )


def test_rejection_threshold_is_ten(monkeypatch, raw_cube, golden):
    """Pins the boundary of the rejection rule, which the fixture alone does not reach.

    On the generated fixture the PCA estimate lands at 4, well below the threshold, so
    nothing there exercises the rejection branch. This test stubs an estimate exactly at
    the boundary and one just above it.

    The threshold is an exclusive comparison, so an estimate of exactly 10 is kept and
    one of 11 is replaced with 1. With the other methods agreeing on 3, a kept estimate
    dominates the final ``max`` while a rejected one leaves 3 as the answer, which makes
    the two cases tell apart from the returned value alone.
    """
    monkeypatch.setattr(phase_number_est, "_n_by_hfc", lambda cube: np.array([3, 3, 3]))

    monkeypatch.setattr(phase_number_est, "_n_by_pca", lambda cube: (10, 3))
    kept, _ = _quietly(phase_number_est.estimate_phase_number, raw_cube)
    golden.check("analysis.threshold.at_ten", kept)

    monkeypatch.setattr(phase_number_est, "_n_by_pca", lambda cube: (11, 3))
    rejected, _ = _quietly(phase_number_est.estimate_phase_number, raw_cube)
    golden.check("analysis.threshold.above_ten", rejected)

    assert kept == 10, "an estimate of exactly 10 is currently kept, not rejected"
    assert rejected == 3, "an estimate of 11 is currently replaced with 1, leaving 3 as the answer"


def test_a_variance_curve_with_no_knee_survives_the_method(monkeypatch, raw_cube):
    """The same failure, driven through the real ``_n_by_pca`` rather than stubbed.

    ``kneed`` is replaced so that the method actually returns None, which is what the
    previous test assumes but does not exercise.
    """

    class NoKnee:
        def __init__(self, *args, **kwargs):
            self.elbow = None

    monkeypatch.setattr(phase_number_est.kneed, "KneeLocator", NoKnee)

    prepared = _preprocessed(raw_cube)
    n_80, n_elbow = phase_number_est._n_by_pca(prepared)
    assert n_elbow is None

    predicted_n, confidence = _quietly(phase_number_est.estimate_phase_number, raw_cube)

    assert predicted_n in set(range(1, 12))
    assert confidence in {"high", "moderate", "low"}


def test_nmf_phase_spectra_are_pinned(raw_cube, golden):
    """The NMF end members, with the settings the viewer uses.

    The settings come from ``NmfConfig``, which is where the viewer reads them, so
    changing a default there moves this snapshot rather than passing unnoticed.
    Running the decomposition directly rather than through the viewer keeps the
    figures out of it; the viewer passing its config on is covered separately.

    A note on reproducibility: ``random_state`` is ``None``, but the ``nndsvda``
    initialisation is deterministic, so this path is stable in practice. That is worth
    knowing, because it means a missing seed is not what makes this particular result
    hard to reproduce.
    """
    settings = NmfConfig().as_kwargs(N_COMPONENTS)
    pipeline = rp.preprocessing.Pipeline(
        [
            rp.preprocessing.despike.WhitakerHayes(),
            rp.preprocessing.baseline.ASLS(),
            rp.preprocessing.normalise.MinMax(),
        ]
    )

    np.random.seed(synthetic.FIXTURE_SEED)
    nmf = rp.analysis.decompose.NMF(**settings)
    abundance_maps, phases = _quietly(nmf.apply, pipeline.apply(raw_cube))

    golden.check("analysis.nmf.phase_spectra", getattr(phases, "spectral_data", phases))
    golden.check("analysis.nmf.abundance_maps", np.asarray(abundance_maps))


def test_the_nmf_config_holds_the_settings_the_viewer_used_to_inline():
    """The defaults are the values that were in the viewer's dictionary.

    Pinned as a whole mapping rather than key by key, so a renamed or dropped setting
    is a failure here instead of a kwarg that silently stops being passed.
    """
    assert NmfConfig().as_kwargs(3) == {
        "n_components": 3,
        "init": "nndsvda",
        "solver": "cd",
        "beta_loss": "frobenius",
        "tol": 1e-4,
        "max_iter": 10000,
        "random_state": None,
        "alpha_W": 0.0,
        "alpha_H": "same",
        "l1_ratio": 0.0,
        "verbose": 0,
        "shuffle": False,
    }
