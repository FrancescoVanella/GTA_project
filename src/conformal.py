"""Split conformal anomaly detection on top of any anomaly score.

p-value of a new score s*: p* = (1 + #{cal scores >= s*}) / (n_cal + 1).
If the new cycle is exchangeable with the (healthy) calibration cycles,
P(p* <= alpha) <= alpha: distribution-free false-alarm control."""
import numpy as np

def p_values(s_cal, s_test):
    s_cal = np.sort(np.asarray(s_cal))
    n = len(s_cal)
    # #{cal >= s} = n - searchsorted_left(s)
    ge = n - np.searchsorted(s_cal, np.asarray(s_test), side='left')
    return (1.0 + ge) / (n + 1.0)

def sustained_alarm(pv, alpha=0.05, k=3):
    """Index of first cycle where p<=alpha holds k consecutive times, else None."""
    run = 0
    for t, hit in enumerate(np.asarray(pv) <= alpha):
        run = run + 1 if hit else 0
        if run >= k:
            return t - k + 1
    return None

def evaluate(pv, rul, alphas=(0.01, 0.05), healthy_rul=120, theta=30):
    """Empirical FPR on healthy cycles and detection rate on near-failure cycles."""
    out = {}
    pv, rul = np.asarray(pv), np.asarray(rul)
    for a in alphas:
        alarm = pv <= a
        out[a] = {'fpr_healthy': float(alarm[rul > healthy_rul].mean()),
                  'tpr_failing': float(alarm[rul < theta].mean())}
    return out
