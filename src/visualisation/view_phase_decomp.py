import math
import numpy as np
import ramanspy as rp
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from .theme import apply_theme, BG, FG, GRID, ACCENT
from src.analysis.phase_number_est import estimate_phase_number

def show_spectra_ext_viewer(hsi_cube, n_components, start=None, end=None):
    apply_theme()
    
    if start is not None or end is not None:
        cropper = rp.preprocessing.misc.Cropper(region=(start, end))
        hsi_cube = cropper.apply(hsi_cube)

    # estimate number of phases if requested
    if n_components == -1:
        n_components, confidence = estimate_phase_number(hsi_cube)
        print(f"Estimated {n_components} phases with {confidence} confidence.")
    
    kwargs = {
        "init"          : "nndsvda",
        "solver"        : "cd",
        "beta_loss"     : "frobenius",
        "tol"           : 1e-4,
        "max_iter"      : 10000,
        "random_state"  : None,
        "alpha_W"       : 0.0,
        "alpha_H"       : "same",
        "l1_ratio"      : 0.0,
        "verbose"       : 0,
        "shuffle"       : False
    }
    nmf = rp.analysis.decompose.NMF(n_components=n_components, **kwargs)
    
    # smoothing distorts potential baseline correction
    pipeline_nmf = rp.preprocessing.Pipeline([
        rp.preprocessing.despike.WhitakerHayes(),
        rp.preprocessing.baseline.ASLS(),
        rp.preprocessing.normalise.MinMax()
    ])
    hsi_cube_nmf = pipeline_nmf.apply(hsi_cube)

    abundance_maps, phases = nmf.apply(hsi_cube_nmf)
    
    ax = rp.plot.spectra(
        phases, hsi_cube.spectral_axis,
        plot_type="single stacked",
        label=[f"Spectra {i + 1}" for i in range(len(phases))]
    )

    magma = cm.get_cmap("magma")
    lines = ax.get_lines()
    colors = magma(np.linspace(0.15, 0.85, len(lines)))

    for line, color in zip(lines, colors):
        line.set_color(color)

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