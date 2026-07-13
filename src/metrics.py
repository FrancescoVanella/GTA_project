"""Evaluation metrics. Pre-registered protocol (notebook 02):
anomalous = RUL < 30; buffer protocol excludes 30 < RUL < 100;
primary comparison setting = FD004, regime-norm, NO buffer."""
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score

def labels(rul, theta=30, buffer=(30, 100), use_buffer=True):
    """1 = anomalous, 0 = normal, -1 = excluded (buffer band)."""
    rul = np.asarray(rul)
    if use_buffer:
        return np.where(rul < theta, 1, np.where(rul > buffer[1], 0, -1))
    return (rul < theta).astype(int)

def auroc(score, rul, **kw):
    y = labels(rul, **kw); m = y >= 0
    return roc_auc_score(y[m], np.asarray(score)[m])

def auprc(score, rul, **kw):
    y = labels(rul, **kw); m = y >= 0
    return average_precision_score(y[m], np.asarray(score)[m])

def cost_weighted(score, rul, thr, c_fn=50, c_fp=1, **kw):
    """Cost at an operating point; justify c_fn/c_fp in the report (A5)."""
    y = labels(rul, **kw); m = y >= 0
    pred = (np.asarray(score)[m] >= thr).astype(int)
    fn = int(((pred == 0) & (y[m] == 1)).sum())
    fp = int(((pred == 1) & (y[m] == 0)).sum())
    return c_fn * fn + c_fp * fp, fn, fp

def spearman_rul(score, rul):
    """Label-free sanity: score should correlate negatively with RUL."""
    return spearmanr(np.asarray(score), np.asarray(rul)).statistic

def detection_delays(df, score_col, thr, k=3):
    """Per-engine RUL at first sustained crossing (k consecutive) -> lead times."""
    leads = []
    for _, g in df.groupby('unit'):
        run, lead = 0, None
        s = g[score_col].values
        for t in range(len(s)):
            run = run + 1 if s[t] >= thr else 0
            if run >= k:
                lead = g['rul'].values[t - k + 1]; break
        leads.append(lead)
    return leads
