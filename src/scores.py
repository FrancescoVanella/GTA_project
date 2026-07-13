"""The geometric model: PCA latent space fitted on healthy data + four scores.

Position      p_t = Mahalanobis distance from the healthy centroid  ("how far?")
Velocity      v_t = ||z̄_t - z̄_{t-1}|| on the SMOOTHED latent        ("how fast?")
Net displ.    D_t = ||z_t - z_ref||, z_ref = unit's first-10-cycle mean ("how much life?")
Direction     u_t = (z_t - z_ref)/D_t                               ("which fault mode?")

Design notes (validated in the pilot study):
- velocity on RAW latent is noise (AUROC 0.54 vs 0.81 smoothed)
- arc length sum||dz|| is an AGE PROXY under noise (corr 0.996 with cycle on
  healthy data); net displacement cancels noise (corr 0.568, AUROC 1.00)
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from .loaders import SENSORS, HEALTHY_RUL

class GeometricModel:
    def __init__(self, n_components=3, w=10, sensors=SENSORS):
        self.k, self.w, self.sensors = n_components, w, list(sensors)
        self.pca, self.mu_H, self.VI = None, None, None

    def fit(self, fit_df):
        """Fit PCA + healthy Gaussian on healthy cycles (RUL > 120) of fit engines."""
        H = fit_df[fit_df.rul > HEALTHY_RUL][self.sensors]
        self.pca = PCA(n_components=self.k).fit(H)
        Zh = self.pca.transform(H)
        self.mu_H = Zh.mean(axis=0)
        self.VI = np.linalg.inv(np.cov(Zh.T) + 1e-6 * np.eye(self.k))
        return self

    def latent(self, df):
        return self.pca.transform(df[self.sensors])

    def score(self, df):
        """Return a DataFrame (index-aligned to df) with the four scores."""
        rows = []
        for _, g in df.groupby('unit', sort=False):
            Z = self.latent(g)
            d = Z - self.mu_H
            pos = np.sqrt(np.einsum('ij,jk,ik->i', d, self.VI, d))
            Zs = pd.DataFrame(Z).rolling(self.w, min_periods=1).mean().values
            st = np.linalg.norm(np.diff(Zs, axis=0), axis=1)
            vel = np.concatenate([[st[0] if len(st) else 0.0], st])
            zref = Z[:min(10, len(Z))].mean(axis=0)
            disp = Z - zref
            D = np.linalg.norm(disp, axis=1)
            U = disp / (D[:, None] + 1e-12)
            r = pd.DataFrame({'pos': pos, 'vel': vel, 'net': D}, index=g.index)
            for j in range(self.k):
                r[f'dir{j}'] = U[:, j]
            rows.append(r)
        return pd.concat(rows).loc[df.index]

def unit_directions(model, df, late_rul=15, min_cycles=5):
    """Per-unit late-life mean direction from the healthy centroid (normalized).
    Used by the fault-mode clustering (cluster-then-angle; NEVER use the
    mean direction of a mixed-fault subset directly)."""
    dirs, units = [], []
    for u, g in df[df.rul < late_rul].groupby('unit'):
        if len(g) < min_cycles:
            continue
        v = model.latent(g).mean(axis=0) - model.mu_H
        dirs.append(v / np.linalg.norm(v))
        units.append(u)
    return np.array(dirs), np.array(units)
