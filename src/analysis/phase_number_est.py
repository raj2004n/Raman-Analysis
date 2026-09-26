import logging
from dataclasses import dataclass

import kneed
import numpy as np
import ramanspy as rp
from pysptools import material_count
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.config import EstimatorConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MethodEstimate:
    """What one method found, or why it contributes no number.

    A method ends up with nothing to contribute in two ways: it produced no estimate
    at all, or it produced one and a bound threw it away. Both are recorded the same
    way, an empty value with the reason beside it, because from the vote's point of
    view they are the same thing. A rejection used to be written over with a 1 that
    then voted with the real answers, which made a discarded guess indistinguishable
    from a method that had genuinely counted one phase.

    Attributes:
        name: The method, as it is reported to the user.
        value: The number of phases this method suggests, or None.
        rejected: Why this estimate does not vote, or None when it does. A method
            that produced nothing carries its reason here too.
    """

    name: str
    value: int | None = None
    rejected: str | None = None

    @property
    def votes(self) -> bool:
        """Whether this estimate is one of the numbers the answer is decided from."""
        return self.value is not None and self.rejected is None


@dataclass(frozen=True)
class PhaseEstimate:
    """The phase count, the confidence in it, and what every method reported.

    Attributes:
        n: How many phases, or None when too few methods produced an estimate for
            there to be an answer.
        confidence: 'high', 'moderate' or 'low' when there is a count, and
            'undetermined' when there is not.
        votes: Every method's report, in the order the methods ran, the rejections
            and their reasons included, so a caller can show what was discarded and
            why rather than only the number that survived.
    """

    n: int | None
    confidence: str
    votes: tuple[MethodEstimate, ...]


def _n_by_pca(hsi_cube):
    # get matrix containing spectral data
    X = hsi_cube.spectral_data

    # prepare data to shape (rows * cols, band_lenght), and scale
    rows, cols, b = X.shape
    X = X.reshape(rows * cols, b).astype(np.float64)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # principal components that capture 80% variance
    pca_80 = PCA(n_components=0.8)
    pca_80.fit_transform(X)
    n_80 = pca_80.n_components_

    # principal components by knee method
    pca_elbow = PCA(n_components=10)
    pca_elbow.fit_transform(X)
    ev = pca_elbow.explained_variance_
    x = np.arange(1, 11)
    kn = kneed.KneeLocator(x, ev, curve='convex', direction='decreasing')
    # None when the curve has no knee, so this method has no estimate to contribute
    n_elbow = kn.elbow

    """
    # Kaiser's method: Only keep those whose eigenvalues greater than 1.
    # In practice, principal components with eigenvales like 0.95 may still
    # contain significant variance. Hence, this method is not the only one being used.
    pca_kaiser = PCA()
    pca_kaiser.fit_transform(X)
    ev = pca_kaiser.explained_variance_
    # pick out principal components with eigenvalue >= 1
    n_kaiser = np.sum(ev >= 1.0)
    """
    return n_80, n_elbow

def _n_by_hfc(hsi_cube):
    # get spectral matrix
    X = hsi_cube.spectral_data
    # sclae spectral matrix
    X = (X - X.min()) / (X.max() - X.min())

    hfcvd = material_count.HfcVd()
    fars = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    ns_hfc = hfcvd.count(X, far=fars, noise_whitening=True)
    return ns_hfc

def _determine_confidence(ns):
    """
    How far the estimates agree with each other.

    Args:
        ns (Sequence[int]): The estimates that counted. A method that produced no
            estimate is not in this sequence.

    Possible confidence values:
    - 'high' three or more estimates, all the same
    - 'moderate' some of them the same, or fewer than three of them
    - 'low' no two of them the same

    A pair that agrees cannot reach 'high'. Three methods agreeing is stronger
    evidence than two agreeing, and callers read 'high' as a number to rely on.
    """
    values = list(ns)
    if len(values) < 2:
        return 'low'
    if len(set(values)) == 1:
        return 'high' if len(values) >= 3 else 'moderate'
    if len(set(values)) < len(values):
        return 'moderate'
    return 'low'


def _check_bound(name, value, cfg):
    """The estimate as a vote, or a rejection saying which bound it broke.

    A method that returned nothing is rejected here as well: there is no number to
    count, which is the same situation as a number that was thrown away, and the
    reason records which of the two happened.
    """
    if value is None:
        return MethodEstimate(name, None, "the method produced no estimate")

    value = int(value)
    if value > cfg.reject_above:
        return MethodEstimate(
            name, None,
            f"the estimate of {value} is above the limit of {cfg.reject_above}",
        )
    return MethodEstimate(name, value)


def _consistent_with_pca(name, ns_hfc, surviving_pca, cfg):
    """The NW-HFC estimate nearest what the surviving PCA estimates agree on.

    HFC-VD reports one count per false alarm rate, so there are seven candidates and
    the one nearest the PCA estimates is taken as this method's answer. That answer
    only votes if it is close to the largest PCA estimate, which is what keeps a
    method that disagrees with the others from deciding the result. With no PCA
    estimate surviving there is nothing to agree with and the check cannot be made,
    so this abstains rather than counting as agreement.
    """
    if not surviving_pca:
        return MethodEstimate(name, None, "there was no PCA estimate to compare against")

    values = [estimate.value for estimate in surviving_pca]
    target = float(np.mean(values))
    value = int(ns_hfc[np.abs(np.asarray(ns_hfc) - target).argmin()])

    largest = max(values)
    gap = abs(value - largest)
    if gap < cfg.hfc_tolerance:
        return MethodEstimate(name, value)
    return MethodEstimate(
        name, None,
        f"the nearest estimate of {value} is {gap} from the largest PCA estimate of "
        f"{largest}, beyond the tolerance of {cfg.hfc_tolerance}",
    )


def combine(votes, cfg):
    """The phase count the estimates agree on, or None when they do not agree enough.

    A rejected estimate is not a number this rule can use, so the answer is the
    largest estimate that survived. Fewer than ``cfg.min_votes`` survivors is not a
    count with weak support, it is no count: reporting the largest of one or two
    numbers is how a run ends up naming a phase count that nothing in the data
    supports.

    Args:
        votes: One ``MethodEstimate`` per method, in the order the methods ran.
        cfg (EstimatorConfig): The bounds the rule is applied with.

    Returns:
        PhaseEstimate: the count, its confidence, and the reports it was decided from.
    """
    survivors = [estimate for estimate in votes if estimate.votes]
    if len(survivors) < cfg.min_votes:
        return PhaseEstimate(None, "undetermined", tuple(votes))

    values = [estimate.value for estimate in survivors]
    return PhaseEstimate(max(values), _determine_confidence(values), tuple(votes))


def _report(estimate, cfg):
    """Logs what each method reported and what was decided from it.

    Every method is named, including the ones that contributed nothing, so a reader can
    see a discarded estimate and its reason rather than only the number that survived.
    """
    logger.info("-----------------------------")
    logger.info("Final estimates of number of distinct spectra:")
    for vote in estimate.votes:
        if vote.votes:
            logger.info("%s: %s", vote.name, vote.value)
        else:
            logger.info("%s: no vote, %s", vote.name, vote.rejected)
    logger.info("-----------------------------")

    if estimate.n is None:
        survivors = sum(1 for vote in estimate.votes if vote.votes)
        logger.info(
            "Best estimate of number of distinct spectra is undetermined: "
            "%s of %s methods gave an estimate, fewer than the %s needed to decide",
            survivors, len(estimate.votes), cfg.min_votes,
        )
    else:
        logger.info(
            "Best estimate of number of distinct spectra is %s with %s confidence",
            estimate.n, estimate.confidence,
        )


def estimate_phase_number(hsi_cube, cfg=None):
    """Estimates how many distinct spectra the cube holds, and how far to trust that.

    Three methods estimate the count and are combined by ``combine``. Each one
    reports rather than votes silently, so a method that was rejected or found
    nothing at all is visible in the result instead of being replaced by a number
    that then counted.

    Args:
        hsi_cube: The cube to estimate from.
        cfg (EstimatorConfig, optional): The bounds to apply. Defaults to
            EstimatorConfig(), whose values are the ones this rule used to spell out
            in its own body.

    Returns:
        PhaseEstimate: the count, which is None when the methods did not agree on
        one, the confidence, and every method's report.
    """
    if cfg is None:
        cfg = EstimatorConfig()

    pipeline_pca = rp.preprocessing.Pipeline([
    rp.preprocessing.despike.WhitakerHayes(),
    rp.preprocessing.baseline.ASLS(),
    ])

    pipeline_hfc = rp.preprocessing.Pipeline([
    rp.preprocessing.despike.WhitakerHayes(),
    rp.preprocessing.baseline.ASLS(),
    ])

    hsi_cube_pca = pipeline_pca.apply(hsi_cube)
    hsi_cube_hfc = pipeline_hfc.apply(hsi_cube)

    n_80, n_elbow = _n_by_pca(hsi_cube_pca)
    ns_hfc        = _n_by_hfc(hsi_cube_hfc)

    logger.info("-----------------------------")
    logger.info("PCA 80%% estimate: %s", n_80)
    logger.info("-----------------------------")
    logger.info("PCA elbow estimate: %s", n_elbow)
    if n_elbow is None:
        logger.info("No knee in the PC variance curve, so this method has nothing to report.")
    logger.info("-----------------------------")
    logger.info("NW-HFC estimates: %s", ns_hfc)

    # the two PCA estimates are checked against the bound first, since the NW-HFC
    # check compares against the PCA estimates that survived it
    pca_votes = [
        _check_bound("PCA 80%", n_80, cfg),
        _check_bound("PCA elbow", n_elbow, cfg),
    ]
    surviving_pca = [estimate for estimate in pca_votes if estimate.votes]
    hfc_vote = _consistent_with_pca("NW-HFC", ns_hfc, surviving_pca, cfg)

    estimate = combine([*pca_votes, hfc_vote], cfg)
    _report(estimate, cfg)
    return estimate
