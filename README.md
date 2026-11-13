# Spectral Moment Embedding

Spectral Moment Embedding (SME) turns the covariance structure of a multichannel
time series into a feature vector. It forms covariance-matrix powers and collects
their diagonal, upper-triangular or row-sum entries, without eigendecomposition.

**Paper:** Jeet Bandhu Lahiri and Siddharth Panwar,
[Spectral moment embedding: a matrix-moment generalisation of characteristic response analysis for covariance-based signal classification](https://www.nature.com/articles/s41598-026-67565-z).
*Scientific Reports*, published 13 November 2026.
[DOI: 10.1038/s41598-026-67565-z](https://doi.org/10.1038/s41598-026-67565-z)

## Install

Use Python 3.10 or later.

```bash
git clone https://github.com/jeetblahiri/sme.git
cd sme
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

Check the core implementation without downloading data:

```bash
python run_all_experiments.py --test-only
```

## Try SME

Each row of `X` is a time sample; each column is a channel.

```python
import numpy as np
from sme_core import compute_cov, sme_diagonal, sme_vech

X = np.random.default_rng(42).standard_normal((500, 4))
C = compute_cov(X)
features = sme_diagonal(C, p_max=3)  # 12 features: diag(C), diag(C^2), diag(C^3)
full_covariance = sme_vech(C, p_max=1)  # All 10 distinct covariance entries
```

Compute one feature vector per signal window, then fit a classifier. Fit feature
scaling on training data and keep subjects or source recordings in separate folds.

## Experiments

The benchmark covers motor imagery EEG, sleep staging, ECG arrhythmia, bearing
faults and inertial activity recognition. Download the datasets separately:

| Task | Dataset | Local folder |
| --- | --- | --- |
| E1: motor imagery | [EEGMMIDB](https://physionet.org/content/eegmmidb/1.0.0/) | `midb/` |
| E2: sleep staging | [Sleep-EDF Expanded](https://physionet.org/content/sleep-edfx/1.0.0/) | `sleep_edf/sleep-cassette/` |
| E3: arrhythmia | [MIT-BIH](https://physionet.org/content/mitdb/1.0.0/) | `mit-bih/` |
| E4: bearing faults | [CWRU](https://engineering.case.edu/bearingdatacenter) | `cwru/` |
| E5: human activity | [UCI HAR v2.1](https://archive.ics.uci.edu/dataset/341/smartphone+based+recognition+of+human+activities+and+postural+transitions) | `uci_har/` |

## Citation

```bibtex
@article{lahiri2026sme,
  author  = {Lahiri, Jeet Bandhu and Panwar, Siddharth},
  title   = {Spectral moment embedding: a matrix-moment generalisation of
             characteristic response analysis for covariance-based signal classification},
  journal = {Scientific Reports},
  year    = {2026},
  doi     = {10.1038/s41598-026-67565-z}
}
```

The code and cached results archived during peer review are available on
[Zenodo (10.5281/zenodo.20904924)](https://doi.org/10.5281/zenodo.20904924).
