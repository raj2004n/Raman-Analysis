import kneed
import numpy as np
import ramanspy as rp
from pysptools import material_count
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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


def estimate_phase_number(hsi_cube):

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

    # catch any assumptions that are too high
    print("-----------------------------")
    print(f"PCA 80% estimate: {n_80}")
    if n_80 > 10:
        n_80 = 1
        print(f"Rejected PCA 80% estimate.")
        print(f"Reason: Estimate was too high (>10).")

    print("-----------------------------")
    print(f"PCA Elbow estimate: {n_elbow}")
    if n_elbow is None:
        print("No elbow found in the PCA variance curve, so no estimate from this method.")
    elif n_elbow > 10:
        n_elbow = 1
        print(f"Rejected PCA elbow estimate.")
        print(f"Reason: Estimate was too high (>10).")

    print("-----------------------------")
    print(f"NW-HFC estimates: {ns_hfc}")
    # pick ns_hfc which is closest to mean pca
    # A method that found nothing is left out of the vote entirely. Note that a
    # rejected estimate is not: it has already been replaced by 1 above and votes.
    ns = np.array([n for n in (n_80, n_elbow) if n is not None])
    mean_pca = ns.mean()
    idx = np.abs(ns_hfc - mean_pca).argmin()
    n_hfc = ns_hfc[idx]

    # add n_vd to consideration if it is not too far from max
    if abs(n_hfc - np.max(ns)) < 2:
        ns = np.append(ns, n_hfc)
    else:
        ns = np.append(ns, 1)
        print("Rejected predictions from NW-HFC.")
        print("Reason: Estimates are too far from PCA predictions.")
    

    predicted_n = np.max(ns)
    confidence = _determine_confidence(ns)

    print("-----------------------------")
    print(f"Final estimates of number of distinct spectra:")
    print(f"PCA 80%: {n_80}")
    print(f"PCA Elbow: {n_elbow if n_elbow is not None else 'no estimate'}")
    print(f"NWHFC: {ns_hfc}")
    print("-----------------------------")
    
    print(f"Best estimate of number of distinct spectra is {predicted_n} with {confidence} confidence")

    return predicted_n, confidence