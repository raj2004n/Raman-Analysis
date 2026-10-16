# Raman Surface Map Analysis Framework

A Python framework for automated analysis of Raman surface mapping data in condensed matter systems. Developed as a Senior Honours Project at the University of Edinburgh (School of Physics and Astronomy).

## What it is

This framework is a toolkit for processing and interpreting large-scale Raman surface maps. In condensed matter physics, surface mapping generates thousands of spectra across a sample; interpreting this manually is impractical. This project automates the pipeline from raw data to mineral identification and spatial mapping.

It combines traditional signal processing (PCA, NMF) with deep learning (1D CNN) to estimate how many spectral components a map holds, extract them, and identify the minerals in it.

---

## Quick Installation

```bash
git clone https://github.com/raj2004n/Raman-Deep-Learning.git
cd Raman-Deep-Learning
python -m venv .venv
.venv/bin/pip install -e '.[dev,web]'
.venv/bin/python -m scripts.apply_dependency_patches
```

Python 3.12+ is required. The `apply_dependency_patches` step is essential as it fixes breaking changes in the `RamanSPy` and `PySptools` libraries.

---

## How it works

The framework operates as a four-stage analytical pipeline, accessible via a browser interface (`python -m app`):

1. **Spatial Visualisation**: Loads Raman maps and generates interactive heatmaps, allowing users to explore spectral intensity distributions across the sample surface.
2. **Distinct Spectra Estimation**: Uses PCA (Scree method) and Noise-whitened HFC to estimate how many unique chemical phases are present in the data.
3. **Spectral Extraction**: Employs Non-negative Matrix Factorisation (NMF) to decompose the map into pure spectral components and their corresponding spatial abundance maps.
4. **Compound Prediction**: Applies a 1D CNN (reconstructed from Liu et al., 2017) fitted over the RRUFF archive to classify the minerals in the map.

---

## Overview

Raman surface mapping produces large volumes of spectral data - hundreds to thousands of individual spectra - that are difficult to interpret manually. This framework provides four analytical tools to address this:

| Module | Description |
|--------|-------------|
| **Spatial Visualiser** | Interactive heatmap for exploring spectral intensity distributions across a sample surface |
| **Distinct Spectra Estimation** | PCA and noise-whitened HFC methods to estimate the number of spectrally distinct sources |
| **Spectral Extraction** | Non-negative matrix factorisation (NMF) to extract and spatially map distinct spectral components |
| **Compound Prediction** | 1D CNN (reconstructed from Liu et al., 2017), fitted over the RRUFF archive by five-fold cross-validation and applied to a map pixel by pixel |

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
├── configs/               # Settings files: one YAML per job, plus the shared seed and README.md
├── src/
│   ├── analysis/          # Phase number estimation (PCA, NWHFC)
│   ├── cnn/               # CNN model, training, evaluation, and prediction
│   ├── data/              # Data loading and grid utilities
│   ├── visualisation/     # Heatmap, phase decomposition, and prediction viewers
│   ├── config.py          # The config schema, and the loader that reads configs/*.yaml
│   ├── logs.py            # The one place logging is configured, called by entry points
│   └── paths.py           # Repository, artifact and output directories
├── artifacts/
│   ├── runs/<name>/       # One directory per trained run: model, encoder, run.json, figures
│   ├── jobs/              # One directory per browser job, and its outputs
│   ├── sessions/          # The one map the interface currently holds
│   ├── saved_maps.json    # The maps the interface has opened before
│   └── mlruns/            # MLflow's file store, when runs are tracked
├── images/                # Figures used by this README
├── notebooks/             # Exploratory scripts and examples
├── patches/               # Fixes for the two unmaintained dependencies, as diffs
├── scripts/               # Applies the dependency patches under patches/
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

The `[notebooks]` extra adds what runs a notebook rather than the framework: the packages
that execute one to a file with its outputs in it. The web server is kept out of the core
dependencies on purpose: a trained run records the core list among its facts, and a model
was not trained against a web server.

The `[tracking]` extra adds MLflow, which mirrors a training run into a searchable store:

```bash
.venv/bin/pip install -e '.[dev,tracking]'
```

It is an extra for the same reason as the web server, and it is the larger of the two: it
pulls a server stack of its own. Without it, and without `MLFLOW_TRACKING_URI` set to point
at a server, tracking is a file store under `artifacts/mlruns`, which needs nothing running.
Training says so and continues untracked when the extra is not installed, since the run
directory it writes is the record either way.

Beside the run directory's own facts, a tracked run holds the diagnostics that only a
searchable store wants: how the split balanced the classes and the weights it gave them,
the held-out scores broken down class by class with the confusion matrix, which spectra
were held out, how long the fit took, what the machine was doing while it ran, and the commit
the code was at. The model and the encoder stay in the run directory, which is still what a
prediction loads. So does the readable half of the report: `misplacements.csv`, one row per
pair of classes the model mixed, and `confusion_matrix.png` beside it, both written whichever
way the run was tracked.

The machine's own numbers are sampled for the length of the run and land under `system/` in
MLflow: CPU and memory, disk and network, and the card with its utilisation, memory and power.
Two things there are worth knowing. The sampling is MLflow's, not this framework's, at one
sample every ten seconds, which `MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL` changes. And it is
switched on for every tracked run, because a run cannot be compared with the ones before it
without it; an environment that has already answered the question, by setting
`MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING` either way, keeps its answer.

Installing the extra also holds pandas below 3 and protobuf below 7, which is MLflow's own
cap on them and not a choice made here. Both are below what the core environment pins, so
the extra is deliberately left out of `requirements.txt`, whose pins describe the
environment the browser interface was verified in; a tracked environment resolves those two
a little lower than the pinned file says.

---

## Configuration

The settings a job runs with live in YAML files under `configs/`, so changing what a run
uses is changing a value rather than editing code:

| File | Holds |
| :--- | :--- |
| `configs/project.yaml` | The seed that training and decomposition share |
| `configs/train.yaml` | One training run, with the network under `architecture` |
| `configs/unmix.yaml` | The decomposition under `nmf`, the phase-count bounds under `estimator` |

`src/config.py` holds the schema for each: a frozen dataclass per file section with a name
and a type for every setting, and the loader that reads a file into one. A setting a file
leaves out is an error rather than a silent default. The seed is the exception: a job file
that does not name its own takes the one from `project.yaml` beside it, so the number a
training run and an unmixing are repeatable from stays the same.

`configs/README.md` says what every setting in the three files means and where the shipped
value came from. The files themselves carry no comments, so that the explanation lives in
one place rather than drifting across three.

Training reads `configs/train.yaml` by default and takes `--config PATH` to read another,
so a second dataset or a second network is a second file rather than an edit:

```bash
.venv/bin/python -m src.cnn.train --config configs/train.yaml --epochs 100
```

Every flag is an override on the file, so a setting no flag offers is still read from it.

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

## Browser Interface

The framework is driven from a browser, which serves itself on this machine:

```bash
.venv/bin/python -m app
```

That opens `http://127.0.0.1:8000`, where the four tools are four screens: **Data** reads a
map, **Heatmap** explores it, **Unmixing** extracts its components and **Predict** applies a
trained run to it. The address is bound to the loopback interface, so this is a desktop tool
and not something reachable from the local network.

The screens share one map at a time. Loading another one releases the map that was held,
and the map is read from disk once and then kept in memory: moving the heatmap's window or
its intensity controls redraws without touching the files again. Changing the pipeline or
the range in the heatmap's top bar is the exception, since those change what the loader
produces rather than how it is drawn.

Reading a map, unmixing it and predicting from it are minutes-long jobs. Each runs in its
own child process, so the interface stays responsive and a job can be cancelled, which kills
it. One job runs at a time.

The predictions are the framework's own: the Predict screen calls `src.cnn.predict`
unchanged, and colours the map by the rule the viewer uses, so a pixel cannot
come out as a different mineral here than it does elsewhere. Two things are arranged
differently rather than computed differently. The legend is drawn in the page, which lets
the notes about `Unknown` and about the minerals a model has but a map does not sit beside
it, and the five most likely minerals reach the page as data, so the hover card is built
without a request per movement of the pointer.

A map or a run the guards refuse leaves the refusal on the screen, in the guard's own
words, rather than as a message that fades. That is deliberate: the model would have
answered anyway, and an answer drawn under a warning nobody read is worse than no answer.

Training has no screen. It stays a script, since it is run once per model rather than looked
at.

The command line that came first has been retired now that the browser covers the same three
analyses. It read its answers from the terminal and opened a matplotlib window per view, so
one view had to be closed before the next one opened. It is kept readable on disk outside the
repository, and git history holds the last version of it.

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

Reconstructed from Liu et al., *Analyst*, 2017, as a 1D LeNet variant, and built in
`src/cnn/model.py`. Its shape and its optimiser are settings in the `architecture` block of the
training config, so a second network is a second file rather than an edit. As shipped, the
network is:

- Three convolutional blocks, 16 → 32 → 64 filters with kernels 21 → 11 → 5 points wide, each a
  convolution followed by a Leaky ReLU and a max pool of stride 2. The convolutions step one
  point at a time and pad nothing, so each block shortens the axis: 1392 points go in and 167
  points by 64 maps come out of the last block. No batch normalisation is used anywhere,
  though the paper describes it after each layer.
- A head holding almost all of the network's 22 million parameters, in one dense layer. The
  last block is flattened to 10,688 features, which a dense layer of 2048 units reads through a
  tanh activation; a second pool halves those 2048 to 1024, dropout at 0.5 is applied, and an
  output layer gives one logit per class. Softmax is applied when predicting rather than inside
  the model, so the loss reads the logits.
- Trained with a class-weighted cross-entropy, since a few minerals account for much of the
  archive, and with Adam.

A fit is a five-fold cross-validation rather than one split: each fold holds out its share of
every class and is fitted on the rest, and the score reported is the mean over the five with
the spread beside it. Inside a fold the fit stops on the loss over a fifth of its own training
half and keeps the weights of the best epoch, so the held-out spectra score the model without
deciding how long it trains.

Five augmentations are drawn from a measured spectrum: a shift along the axis, noise
proportional to intensity, an added background, additive Gaussian noise, and a redraw through a
random selection of the spectrum's own points. Copies are never composed, so every transform is
applied to the measured spectrum rather than to another transform's output, and mixtures of two
same-substance spectra can be added beside them. `configs/train.yaml` names the mode: `uniform`
gives every spectrum the same number of copies, while `balanced_augmentation` counts the spectra
sharing a mineral and a laser wavelength and copies the sparse groups more, so the rarest
minerals at their wavelength bring the most data to the fit.

**Result.** Over the archive's `excellent_unoriented` and `fair_unoriented` folders, 816
held-out spectra in 58 classes, the best recorded run, which draws more shifted and noisy copies
than the shipped file does, scores 86.2% ± 4.1% top-1 and 95.2% top-3. Much of that is the
augmentation: without it the same corpus scores 67.8%. The score is also sensitive to the axis
moving, which is what the sweep in the config measures: it holds at ±15 points and falls to 37%
at ±30. Over the corpus the paper released, the run that follows its protocol most closely scores
81.5% ± 3.1% across 63 classes, against the 93.3% the paper reports for its own model.

---

## Results

| Method | Result |
|--------|--------|
| Spatial visualiser | Successfully captures surface heterogeneity; rolling window enables region-specific analysis |
| PCA scree | Correctly predicted 2 distinct spectra for both test samples in the fingerprint region |
| NMF extraction | Successfully separated diamond anvil cell background from sample signal |
| CNN (archive, unoriented tiers) | 86.2% ± 4.1% top-1 and 95.2% top-3 held out over 816 spectra in 58 classes |
| CNN (paper's released corpus) | 81.5% ± 3.1% top-1 over 63 classes, against the 93.3% the paper reports |

---

## Dependencies

Key libraries used:

- [RamanSPy](https://github.com/bagheria-lab/ramanspy) - Raman data structures, NMF, preprocessing pipelines
- [scikit-learn](https://scikit-learn.org/) - PCA, StandardScaler
- [PySptools](https://pysptools.sourceforge.io/) - NWHFC virtual dimensionality
- [PyTorch](https://pytorch.org/) - CNN implementation, the framework the paper is written in
- [MLflow](https://mlflow.org/) - run tracking, from the optional `[tracking]` extra
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
[raj2004n@gmail.com]
