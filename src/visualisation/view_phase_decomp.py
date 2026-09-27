import logging
import math

import numpy as np
import ramanspy as rp
import matplotlib.pyplot as plt
from matplotlib import colormaps
from .theme import apply_theme
from src.analysis.phase_number_est import estimate_phase_number
from src.config import NmfConfig

logger = logging.getLogger(__name__)

def show_spectra_ext_viewer(hsi_cube, n_components=None, start=None, end=None, nmf_config=None):
    """Unmixes the cube and shows the end-member spectra and their abundance maps.

    Args:
        hsi_cube: The cube to decompose.
        n_components: How many phases to look for, or None to estimate it from the
            cube. -1 is accepted as the same thing, since that is the value the
            command line prompt has always offered for "estimate it". When the
            estimate comes back undetermined there is no count to decompose with, so
            the estimator's report is the whole result and nothing is unmixed.
        start, end: Wavenumber bounds to crop to before unmixing. Either may be None.
        nmf_config (NmfConfig, optional): The decomposition settings. Defaults to
            NmfConfig(), whose values are the ones this viewer used to inline.
    """
    apply_theme()

    if n_components == -1:
        n_components = None

    if start is not None or end is not None:
        cropper = rp.preprocessing.misc.Cropper(region=(start, end))
        hsi_cube = cropper.apply(hsi_cube)

    # estimate number of phases if requested
    if n_components is None:
        estimate = estimate_phase_number(hsi_cube)
        if estimate.n is None:
            logger.info("Nothing to unmix without a phase count.")
            return
        n_components = estimate.n
        logger.info("Estimated %s phases with %s confidence.", n_components, estimate.confidence)

    if nmf_config is None:
        nmf_config = NmfConfig()
    nmf = rp.analysis.decompose.NMF(**nmf_config.as_kwargs(n_components))
    
    # smoothing distorts potential baseline correction
    pipeline_nmf = rp.preprocessing.Pipeline([
        rp.preprocessing.despike.WhitakerHayes(),
        rp.preprocessing.baseline.ASLS(),
        rp.preprocessing.normalise.MinMax()
    ])
    hsi_cube_nmf = pipeline_nmf.apply(hsi_cube)

    abundance_maps, phases = nmf.apply(hsi_cube_nmf)
    
    # The colours are given at the call rather than left to ramanspy's default, which asks
    # matplotlib for a colormap through plt.cm.get_cmap, removed in matplotlib 3.11. The
    # registry is the documented replacement and has been since 3.5. One line is drawn per
    # phase, so one colour each is what the default produced here.
    colors = list(colormaps["magma"](np.linspace(0.15, 0.85, len(phases))))

    rp.plot.spectra(
        phases, hsi_cube.spectral_axis,
        plot_type="single stacked",
        label=[f"Spectra {i + 1}" for i in range(len(phases))],
        color=colors,
    )

    plt.show()
    n = len(abundance_maps)
    cmap="magma"
    if n <= 10:
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        axes = np.array(axes).flatten()

        for i, (abundance_map, ax) in enumerate(zip(abundance_maps, axes)):
            rp.plot.image(
                abundance_map,
                title=f"Spectra {i + 1}",
                cmap=cmap,
                ax=ax
            )

        for ax in axes[n:]:
            ax.set_visible(False)

        plt.tight_layout()
        plt.show()

    else:
        # too many - show individually
        for i, abundance_map in enumerate(abundance_maps):
            rp.plot.image(
                abundance_map,
                title=f"Spectra {i + 1}",
                cmap=cmap
            )
            plt.show()