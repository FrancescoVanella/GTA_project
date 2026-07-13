# Degradation as Trajectory Geometry — NASA C-MAPSS
Anomaly detection project for *Geometric Learning, Time-Variant Data Analysis, and Anomaly Detection*.

**Authors:** Luciano Selimaj, Francesco Vanella.

**Dataset:** NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) turbofan engine
degradation data, provided by the NASA Prognostics Center of Excellence (PCoE) and released into
the public domain. All results in this repository are derived exclusively from that dataset.

**Core idea:** Engine degradation is a trajectory in a PCA latent space fitted on healthy
cycles only. Four scores are derived form this space: position, velocity, net displacement, direction. A split-conformal layer turns scores into p-values with a guarantee on the percentage of false positives that is distribution free.

## Results 
- **Preprocessing dominates models**: per-regime normalization on FD004 moves AUROC from ~0.5 to ~0.99 (notebook 08, A1, 5 seeds).
- **Unsupervised fault-mode identification**: FD003 late-life directions cluster 44/56 (silhouette 0.82, cos between clusters −0.65); one cluster aligns perfectly (cos 1.00) with FD001's HPC direction (notebook 06).
- **Conformal alarms**: guarantee holds on FD001 (FPR 0.006 ≤ α=0.01, detection 100%); mild violation on FD004 diagnosed as an exchangeability failure, mitigation tested (notebook 07). Median lead time ~85–90 cycles before failure.
- **Official test files (touched once)**: 100% of near-failure engines flagged at an 8% nuisance rate (notebook 09).

## Structure
```
data/raw/          # C-MAPSS files (public domain, NASA PCoE)
notebooks/01...09  # run top-to-bottom, in order
src/               # loaders, scores, conformal, metrics (imported by notebooks)
results/           # tables (csv) and figures (png)
```

## Reproduce
```
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace notebooks/0*.ipynb  # or run in Jupyter
```
Notebooks regenerate everything from `data/raw`; processed parquets are cached in `data/processed`.
Pre-registered decisions (θ=30, healthy=RUL>120, buffer 30–100, 5 seeds) are frozen in notebook 02
and `src/loaders.py` — never changed silently.

The VAE row (A4b) is part of the standard run and requires `torch`.

## Data & citation
The C-MAPSS dataset is provided by the **NASA Prognostics Center of Excellence (PCoE)** and is
publicly available at https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data.

