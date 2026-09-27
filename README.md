# Raman Surface Map Analysis Framework

A Python framework for automated analysis of Raman surface mapping data in condensed matter systems. Developed as a Senior Honours Project at the University of Edinburgh (School of Physics and Astronomy).

---

## Overview

Raman surface mapping produces large volumes of spectral data - hundreds to thousands of individual spectra - that are difficult to interpret manually. This framework provides four analytical tools to address this:

| Module | Description |
|--------|-------------|
| **Spatial Visualiser** | Interactive heatmap for exploring spectral intensity distributions across a sample surface |
| **Distinct Spectra Estimation** | PCA and noise-whitened HFC methods to estimate the number of spectrally distinct sources |
| **Spectral Extraction** | Non-negative matrix factorisation (NMF) to extract and spatially map distinct spectral components |
| **Compound Prediction** | 1D CNN (reconstructed from Liu et al., 2017) for Raman spectrum classification; achieves **90.7% test accuracy** on the RRUFF mineral database |

---

## Visualisations

### Interactive Spatial Analysis
| Sample A | Sample B |
| :--- | :--- |
| ![Sample A Heatmap](images/heatmap_sb.png) | ![Sample B Heatmap](images/heatmap_fc.png) |
| *Figure 1a: Spectral distribution (Sample A)* | *Figure 1b: Spectral distribution (Sample B)* |

### Spectral Extraction (NMF)
| Extracted Spectra (Sample A) | Spatial Abundance (Sample A) |
| :--- | :--- |
| ![Spectra A](images/sb2.1.png) | ![Map A](images/sb2.2.png) |

| Extracted Spectra (Sample B) | Spatial Abundance (Sample B) |
| :--- | :--- |
| ![Spectra B](images/fc2.1.png) | ![Map B](images/fc2.2.png) |

---

## Project Structure

```
.
├── app/                   # The browser interface: HTTP routes, jobs and the front end
├── src/
│   ├── analysis/          # Phase number estimation (PCA, NWHFC)
│   ├── cnn/               # CNN model, training, evaluation, and prediction
│   ├── data/              # Data loading and grid utilities
│   ├── visualisation/     # Heatmap, phase decomposition, and prediction viewers
│   ├── config.py          # Frozen configs: seed, NMF, estimator and training settings
│   ├── logs.py            # The one place logging is configured, called by entry points
│   └── paths.py           # Repository, artifact and output directories
├── artifacts/
│   ├── models/            # Saved Keras model files
│   ├── weights/           # Saved model weights
│   ├── encoders/          # Label encoders for compound classes
│   ├── metadata/          # Wavenumber range files
│   └── runs/<name>/       # One directory per trained run: model, encoder, run.json
├── images/                # Figures used by this README
├── notebooks/             # Exploratory scripts and examples
├── patches/               # Fixes for the two unmaintained dependencies, as diffs
├── scripts/               # Entry-point plotting scripts
├── tests/                 # Test suite, with snapshots under tests/golden/
├── outputs/               # Generated figures and plots
├── pyproject.toml         # Dependencies, extras and packaging
└── requirements.txt       # Pinned versions of the last verified environment
```

---

## Installation

```bash
git clone https://github.com/raj2004n/Raman-Deep-Learning.git
cd Raman-Deep-Learning
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/python -m scripts.apply_dependency_patches
```

Python 3.12 or later is required. `pyproject.toml` declares the dependencies;
`requirements.txt` pins the exact versions the suite was last verified against, so use it
in place of the `pip install -e '.[dev]'` line when a bit-for-bit reproduction matters.

The patch step is required, not a repair for a broken install. Two of the dependencies,
RamanSPy and PySptools, are no longer maintained and are broken against a current numpy
and scipy: `np.trapz` was removed from numpy, and PySptools passes a scalar where it
expects a sequence. Neither has a release to move to, and no older version of either
avoids it. `patches/` keeps the lines that fix them as unified diffs, each with a comment
saying what breaks and why, and `scripts/apply_dependency_patches.py` puts them onto the
installed copies. Running it again is safe: a patch already in place is reported and not
reapplied, and `--check` reports what it would do without writing anything.

The `[dev]` extra adds pytest, which is what the suite runs on. The `[web]` extra adds
FastAPI and uvicorn, which serve the browser interface, and the tests that drive it need
both extras:

```bash
.venv/bin/pip install -e '.[dev,web]'
```

The `[notebooks]` extra adds torch, which one notebook needs and the framework itself does
not. The web server is kept out of the core dependencies on purpose: a trained run records
the core list among its facts, and a model was not trained against a web server.

Training calls `keras.utils.plot_model`, which shells out to Graphviz's `dot` binary, so
that has to be on the system as well as the `pydot` package pip installs:

```bash
sudo apt install graphviz
```

---

## Tests

```bash
.venv/bin/python -m pytest tests
```

The snapshots under `tests/golden/` are compared on every run. Re-record them only when a
change to the numbers is intended and understood:

```bash
.venv/bin/python -m pytest tests --update-golden
```

---

## Interactive CLI Usage

To launch the analysis framework, execute the following command from the root directory:

```bash
python3 -m scripts.plot_raman
```

---

### Example Session

```
Enter path, grid rows and grid columns (or '?' for help):
e.g. ~/Code/Data_SH/SB008 10 13
> ~/Code/Data_SH/SB008 10 13
Found 130 .txt file(s) in /home/raj/Code/Data_SH/SB008.

Select mode (or '?' for help):
  [1] Heatmap
  [2] Unmixing
  [3] Predict (incomplete)
> 1

Pipeline? [0] None  [1] P1  [2] P2  [3] P3 (default: 0, or '?' for help)
> 0

Start of spectra in cm⁻¹ (press Enter to skip, or '?' for help)
> 200

End of spectra in cm⁻¹ (press Enter to skip, or '?' for help)
> 1200
```

---

## Methods

### Preprocessing Pipeline

Applied prior to estimation and extraction:

- **Whitaker-Hayes despiking** - removes cosmic ray artefacts
- **Asymmetric least squares (ASLS) baseline correction** - removes fluorescence background
- **Min-max normalisation** - scales spectra to [0, 1] for NMF compatibility

### Distinct Spectra Estimation

Two methods are implemented and compared:

- **PCA (scree method)** - identifies the elbow in the eigenvalue spectrum; found to be the most reliable estimator for condensed matter Raman data
- **PCA (80% explained variance)** - tends to overestimate; sensitive to SNR and spectral range
- **Noise-whitened HFC (NWHFC)** - hyperspectral virtual dimensionality method; evaluated across false alarm rates from 10⁻² to 10⁻⁷. Found to be volatile and unreliable without sample-specific tuning for this data type

### NMF Extraction

- **Initialisation:** NNDSVDA (non-negative double SVD, average variant) for dense data
- **Solver:** Coordinate descent
- **Loss:** Frobenius norm (least-squares); note that Raman noise is Poisson-distributed, so KL-divergence loss may be more appropriate for future work
- **Max iterations:** 10,000

### CNN Architecture

Reconstructed from Liu et al., *Analyst*, 2017. A LeNet-variant 1D CNN:

- Three convolutional blocks (16 → 32 → 64 kernels, sizes 21 → 11 → 5), each followed by batch normalisation, Leaky ReLU, and max-pooling (stride 2)
- Fully connected layer (2048 neurons) with batch normalisation and dropout (0.5)
- Output layer with softmax over *C* classes
- Trained with class-weighted categorical cross-entropy and Adam optimiser
- Data augmentation: spectral shift, noise injection, and linear combination of same-class spectra (4.4× training set increase)

**Result:** 90.7% test accuracy on 493 test spectra across 533 classes (cf. 93.3% reported by Liu et al.). However, the model overfitted.

---

## Results

| Method | Result |
|--------|--------|
| Spatial visualiser | Successfully captures surface heterogeneity; rolling window enables region-specific analysis |
| PCA scree | Correctly predicted 2 distinct spectra for both test samples in the fingerprint region |
| NMF extraction | Successfully separated diamond anvil cell background from sample signal |
| CNN (poor unoriented) | **90.7% test accuracy**, 200 epochs, trained on NVIDIA RTX 3060 |
| CNN (excellent oriented) | **98.1% test accuracy** |

---

## Dependencies

Key libraries used:

- [RamanSPy](https://github.com/bagheria-lab/ramanspy) - Raman data structures, NMF, preprocessing pipelines
- [scikit-learn](https://scikit-learn.org/) - PCA, StandardScaler
- [PySptools](https://pysptools.sourceforge.io/) - NWHFC virtual dimensionality
- [Keras](https://keras.io/) - CNN implementation
- [NumPy](https://numpy.org/) - numerical operations

See `pyproject.toml` for the declared dependencies and `requirements.txt` for the pinned
versions of the last verified environment. Two of the packages above need the changes in
`patches/` to run against a current numpy and scipy; the Installation section covers it.

---

## References

- Liu et al., "Deep convolutional neural networks for Raman spectrum recognition: a unified solution", *Analyst*, 2017
- Chang & Du, "Estimation of number of spectrally distinct signal sources in hyperspectral imagery", *IEEE TGRS*, 2004
- Lee & Seung, "Algorithms for non-negative matrix factorization", *NeurIPS*, 2000
- Georgiev et al., "RamanSPy: An open-source Python package for integrative Raman spectroscopy data analysis", *Analytical Chemistry*, 2024

---

## Author

**Raj Negi** - MPhys Computational Physics, University of Edinburgh
[github.com/raj2004n](https://github.com/raj2004n)