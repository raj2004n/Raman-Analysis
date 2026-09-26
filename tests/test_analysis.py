"""Golden snapshots and behaviour tests for ``src/analysis/phase_number_est.py``.

Three jobs here:

* Pin the numbers the estimator produces, so splitting it into per-method reports
  plus a combination rule can be shown to have preserved them.
* Pin the rule itself, in the cases the fixture does not reach: a rejected estimate,
  a missing estimate, and the bounds. The rule changed on purpose, so its tests
  state what the new behaviour is rather than recording the old one.
* Pin where the bounds live, since they are fields on ``EstimatorConfig`` now and a
  change to a default there is a change to what every run decides.

The method values are read by calling the private helpers directly rather than by
parsing stdout, because the module's prints are expected to be replaced with logging
later and these snapshots should not depend on them.
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import fields

import numpy as np
import pytest
import ramanspy as rp

from src.analysis import phase_number_est
from src.analysis.phase_number_est import MethodEstimate
from src.config import PROJECT_SEED, EstimatorConfig, NmfConfig
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
    estimate = _quietly(phase_number_est.estimate_phase_number, raw_cube)

    golden.check("analysis.phase_estimate.predicted_n", estimate.n)
    golden.check("analysis.phase_estimate.confidence", estimate.confidence)


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

    The method contributes no estimate rather than an estimate of 1, so two methods
    agreeing on 3 report ``moderate``: unanimity between two is weaker evidence than
    unanimity between three.
    """
    monkeypatch.setattr(phase_number_est, "_n_by_pca", lambda cube: (3, None))
    monkeypatch.setattr(phase_number_est, "_n_by_hfc", lambda cube: np.array([3, 3, 3]))

    estimate = _quietly(phase_number_est.estimate_phase_number, raw_cube)

    assert estimate.n == 3
    assert estimate.confidence == "moderate", (
        "a missing elbow leaves two estimates that agree, which must not be reported "
        "with the same confidence as three that agree"
    )
    assert estimate.votes[1].rejected == "the method produced no estimate"
    assert not estimate.votes[1].votes


def test_a_rejected_estimate_is_not_a_vote(monkeypatch, raw_cube):
    """An estimate above the bound leaves the vote rather than joining it as a 1.

    The old rule wrote 1 over a rejected value, and that 1 then counted: it was one of
    the numbers the answer was taken from, and it decided the mean the NW-HFC estimate
    was compared against. The stub here is the case that ended in a confident wrong
    answer rather than a refusal, because with both PCA estimates rejected the vote
    was three copies of that 1 and the run reported one phase with high confidence.

    Neither the 1 nor the rejected values are anywhere in the answer now. The NW-HFC
    check has nothing left to compare against, so it abstains as well, and the result
    says the count is undetermined instead of naming one.
    """
    monkeypatch.setattr(phase_number_est, "_n_by_pca", lambda cube: (17, 14))
    monkeypatch.setattr(phase_number_est, "_n_by_hfc", lambda cube: np.array([3, 3, 3]))

    estimate = _quietly(phase_number_est.estimate_phase_number, raw_cube)

    assert estimate.n is None
    assert estimate.confidence == "undetermined"
    assert [vote.votes for vote in estimate.votes] == [False, False, False]
    assert "above the limit of 10" in estimate.votes[0].rejected
    assert estimate.votes[2].rejected == "there was no PCA estimate to compare against"


def test_a_rejection_leaves_the_other_methods_to_decide(monkeypatch, raw_cube, golden):
    """With one PCA estimate rejected, the methods that remain still answer.

    The rejected estimate is absent from the vote rather than present as a 1, so the
    two methods that did report an estimate are the ones the confidence is read from.
    Two agreeing cannot be 'high', so this reports moderate.

    This stub is the one these snapshots were recorded with, and the answer is the same
    before and after the rule changed: the old rule took the largest of [1, 3, 3] and
    the new one the largest of [3, 3], and both are 3 with two distinct values among
    them. A snapshot cannot show this change, which is why the assertions below read the
    votes and the rejection reason instead, and why the case above, where both PCA
    estimates fall to the bound, is the one that moved.
    """
    monkeypatch.setattr(phase_number_est, "_n_by_pca", lambda cube: (17, 3))
    monkeypatch.setattr(phase_number_est, "_n_by_hfc", lambda cube: np.array([3, 3, 3]))

    estimate = _quietly(phase_number_est.estimate_phase_number, raw_cube)

    golden.check("analysis.rejected_estimate.predicted_n", estimate.n)
    golden.check("analysis.rejected_estimate.confidence", estimate.confidence)

    assert estimate.n == 3
    assert estimate.confidence == "moderate"
    assert [vote.name for vote in estimate.votes if vote.votes] == ["PCA elbow", "NW-HFC"]
    assert "the estimate of 17 is above the limit of 10" in estimate.votes[0].rejected


def test_the_bound_rejects_only_above_its_limit():
    """The comparison is exclusive, so the limit itself is still an estimate."""
    cfg = EstimatorConfig()

    assert phase_number_est._check_bound("PCA 80%", cfg.reject_above, cfg).votes
    assert "above the limit of 10" in phase_number_est._check_bound("PCA 80%", cfg.reject_above + 1, cfg).rejected


def test_the_bound_can_be_moved_by_a_caller(monkeypatch, raw_cube):
    """A tighter bound reaches the whole rule, not only the check that reads it.

    The fixture's own PCA estimate is 4, so a limit of 3 rejects it, and the run that
    reports 4 phases by default reports that the count is undetermined instead.
    """
    np.random.seed(synthetic.FIXTURE_SEED)
    estimate = _quietly(
        phase_number_est.estimate_phase_number, raw_cube, EstimatorConfig(reject_above=3)
    )

    assert estimate.votes[0].value is None
    assert "the estimate of 4 is above the limit of 3" in estimate.votes[0].rejected
    assert estimate.n is None
    assert estimate.confidence == "undetermined"


def test_the_bound_seen_from_the_answer(monkeypatch, raw_cube, golden):
    """The limit and one above it, through the whole rule rather than the check alone.

    On the generated fixture nothing reaches the bound, so both sides of it are stubbed
    here. An estimate of exactly 10 is kept and decides the answer, since 10 is larger
    than the 3 the other methods report; an estimate of 11 is rejected and the two
    methods that remain report 3. Two agreeing cannot be 'high', so the run whose
    estimate was thrown away is also the less confident one.
    """
    monkeypatch.setattr(phase_number_est, "_n_by_hfc", lambda cube: np.array([3, 3, 3]))

    monkeypatch.setattr(phase_number_est, "_n_by_pca", lambda cube: (10, 3))
    kept = _quietly(phase_number_est.estimate_phase_number, raw_cube)
    golden.check("analysis.threshold.at_ten", kept.n)

    monkeypatch.setattr(phase_number_est, "_n_by_pca", lambda cube: (11, 3))
    rejected = _quietly(phase_number_est.estimate_phase_number, raw_cube)
    golden.check("analysis.threshold.above_ten", rejected.n)

    assert (kept.n, kept.confidence) == (10, "low")
    assert (rejected.n, rejected.confidence) == (3, "moderate")
    assert rejected.votes[0].value is None, "the estimate of 11 is not in the vote"


def test_the_consistency_check_rejects_a_gap_at_its_tolerance():
    """The NW-HFC estimate has to be within the tolerance of the largest PCA estimate."""
    cfg = EstimatorConfig()
    surviving = [MethodEstimate("PCA 80%", 5)]

    assert phase_number_est._consistent_with_pca("NW-HFC", np.array([6]), surviving, cfg).votes
    assert "beyond the tolerance of 2" in phase_number_est._consistent_with_pca(
        "NW-HFC", np.array([7]), surviving, cfg
    ).rejected


def test_the_consistency_check_compares_against_the_largest_pca_estimate():
    """The largest, not the smallest: the fixtures' two PCA estimates rarely differ much.

    With estimates of 3 and 5, an NW-HFC estimate of 6 is within the tolerance of the
    larger and not within the tolerance of the smaller, so which one is used decides
    whether the method votes.
    """
    cfg = EstimatorConfig()
    surviving = [MethodEstimate("PCA 80%", 5), MethodEstimate("PCA elbow", 3)]

    assert phase_number_est._consistent_with_pca("NW-HFC", np.array([6]), surviving, cfg).votes


def test_the_consistency_check_starts_from_the_mean_of_the_pca_estimates():
    """HFC-VD reports one count per false alarm rate, and the one nearest the mean is taken.

    The mean rather than the largest: this method is asked for something that agrees
    with what the PCA methods found on average, and the candidate nearest the largest
    of them can be a count nothing else is anywhere near.
    """
    cfg = EstimatorConfig()
    surviving = [MethodEstimate("PCA 80%", 9), MethodEstimate("PCA elbow", 3)]

    chosen = phase_number_est._consistent_with_pca("NW-HFC", np.array([4, 10]), surviving, cfg)

    assert "the nearest estimate of 4" in chosen.rejected, (
        "4 is two from the mean of 6, while 10 is four from it"
    )


def test_combine_takes_the_largest_estimate_that_survived():
    """The rule the estimator replaced, stated on its own rather than through a cube."""
    votes = [MethodEstimate("a", 3), MethodEstimate("b", 5), MethodEstimate("c", 4)]

    estimate = phase_number_est.combine(votes, EstimatorConfig())

    assert estimate.n == 5
    assert estimate.confidence == "low", "three different numbers, so no two agree"
    assert estimate.votes == tuple(votes), "every method is reported, not only the winner"


def test_combine_needs_the_minimum_number_of_votes():
    """One survivor is not a decision, however confident that one method sounds.

    Reporting the largest of the surviving estimates would answer 3 here, from a run
    where the other method was thrown away for being out of range.
    """
    votes = [
        MethodEstimate("a", 3),
        MethodEstimate("b", None, "the estimate of 40 is above the limit of 10"),
    ]

    estimate = phase_number_est.combine(votes, EstimatorConfig())

    assert estimate.n is None
    assert estimate.confidence == "undetermined"
    assert [vote.name for vote in estimate.votes if vote.votes] == ["a"]


def test_combine_decides_once_the_minimum_is_met():
    """Two survivors is enough to name a count, and agreement between two is moderate."""
    votes = [
        MethodEstimate("a", 3),
        MethodEstimate("b", 3),
        MethodEstimate("c", None, "the method produced no estimate"),
    ]

    estimate = phase_number_est.combine(votes, EstimatorConfig())

    assert estimate.n == 3
    assert estimate.confidence == "moderate"


def test_combine_reads_the_minimum_from_the_config():
    """The minimum is a field, so a caller who wants to decide from one method can."""
    votes = [MethodEstimate("a", 3)]

    assert phase_number_est.combine(votes, EstimatorConfig(min_votes=1)).n == 3
    assert phase_number_est.combine(votes, EstimatorConfig()).n is None


def test_the_estimator_config_holds_the_bounds_that_were_in_the_rule():
    """The defaults are the values that stood in the body of the combination rule.

    Pinned as a whole, so a bound cannot be added, renamed or dropped here without
    this failing: a new field changes what every run decides, so it is worth a look.
    """
    cfg = EstimatorConfig()

    assert [field.name for field in fields(cfg)] == ["reject_above", "min_votes", "hfc_tolerance"]
    assert (cfg.reject_above, cfg.min_votes, cfg.hfc_tolerance) == (10, 2, 2)


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

    estimate = _quietly(phase_number_est.estimate_phase_number, raw_cube)

    # one method left, and the NW-HFC estimate is nowhere near it, so there is no
    # count to name rather than a count taken from that one surviving number
    assert estimate.n is None
    assert estimate.confidence == "undetermined"


def test_nmf_phase_spectra_are_pinned(raw_cube, golden):
    """The NMF end members, with the settings the viewer uses.

    The settings come from ``NmfConfig``, which is where the viewer reads them, so
    changing a default there moves this snapshot rather than passing unnoticed.
    Running the decomposition directly rather than through the viewer keeps the
    figures out of it; the viewer passing its config on is covered separately.

    A note on reproducibility: ``random_state`` is the project seed, and this snapshot
    is the same as it was when that setting was ``None``. The ``nndsvda``
    initialisation is computed from the data rather than drawn, so the seed makes an
    otherwise open setting explicit without changing what is decomposed. That was
    checked rather than assumed, which is why the value is pinned here.
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

    ``random_state`` is the project seed rather than the ``None`` the viewer passed. The
    decomposition is unaffected either way with this initialisation, which is why the
    pinned end members above did not move; what changed is that the setting is now
    written down, so the seed a result came from is not something the process decides.
    """
    assert NmfConfig().as_kwargs(3) == {
        "n_components": 3,
        "init": "nndsvda",
        "solver": "cd",
        "beta_loss": "frobenius",
        "tol": 1e-4,
        "max_iter": 10000,
        "random_state": PROJECT_SEED,
        "alpha_W": 0.0,
        "alpha_H": "same",
        "l1_ratio": 0.0,
        "verbose": 0,
        "shuffle": False,
    }
