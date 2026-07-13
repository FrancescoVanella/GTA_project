"""Data loading, splits, and normalization for NASA C-MAPSS."""
from pathlib import Path
import numpy as np
import pandas as pd

COLS = ['unit', 'cycle', 'op1', 'op2', 'op3'] + [f's{i}' for i in range(1, 22)]
SENSORS_ALL = [f's{i}' for i in range(1, 22)]
# 14 informative sensors (EDA, notebook 03): constant/near-constant sensors dropped
SENSORS = ['s2', 's3', 's4', 's7', 's8', 's9', 's11', 's12',
           's13', 's14', 's15', 's17', 's20', 's21']
SEEDS = [42, 123, 999, 7, 2026]
HEALTHY_RUL = 120   # healthy reference: RUL > 120 (pre-registered, notebook 02)
THETA = 30          # anomalous: RUL < 30 (pre-registered)
BUFFER = (30, 100)  # excluded band in buffer protocol

def data_dir():
    here = Path(__file__).resolve().parent.parent
    return here / 'data' / 'raw'

def load(fd, split='train'):
    """Load one subset. Adds per-cycle RUL for train (run-to-failure structure)."""
    df = pd.read_csv(data_dir() / f'{split}_{fd}.txt', sep=r'\s+',
                     header=None, names=COLS)
    df[SENSORS_ALL] = df[SENSORS_ALL].astype('float64')
    if split == 'train':
        df['rul'] = df.groupby('unit')['cycle'].transform('max') - df['cycle']
    return df

def load_rul(fd):
    """True RUL at last observed cycle for each test engine."""
    return pd.read_csv(data_dir() / f'RUL_{fd}.txt', header=None)[0].values

def split_units(units, seed, frac=(0.6, 0.2, 0.2)):
    """Engine-level split: fit / calibration / validation. Never split rows."""
    rng = np.random.RandomState(seed)
    u = np.array(sorted(units))
    rng.shuffle(u)
    n = len(u)
    a, b = int(frac[0] * n), int((frac[0] + frac[1]) * n)
    return u[:a], u[a:b], u[b:]

class RegimeAssigner:
    """KMeans(6) on op settings; regimes are trivially separable (notebook 01)."""
    def __init__(self, n_regimes=6, seed=0):
        from sklearn.cluster import KMeans
        self.km = KMeans(n_regimes, n_init=10, random_state=seed)
    def fit(self, df):
        self.km.fit(df[['op1', 'op2', 'op3']].values); return self
    def assign(self, df):
        df = df.copy()
        df['regime'] = self.km.predict(df[['op1', 'op2', 'op3']].values)
        return df

class Normalizer:
    """z-score per sensor, optionally per regime.
    DISCIPLINE: fit on healthy cycles of FIT engines only; never refit downstream."""
    def __init__(self, sensors=SENSORS, by_regime=False):
        self.sensors, self.by_regime, self.stats = list(sensors), by_regime, None
    def fit(self, healthy_df):
        if self.by_regime:
            g = healthy_df.groupby('regime')[self.sensors]
            self.stats = (g.mean(), g.std() + 1e-9)
        else:
            self.stats = (healthy_df[self.sensors].mean(),
                          healthy_df[self.sensors].std() + 1e-9)
        return self
    def transform(self, df):
        out = df.copy()
        if self.by_regime:
            mu, sd = self.stats
            m = mu.loc[df['regime']].values
            s = sd.loc[df['regime']].values
            out[self.sensors] = (df[self.sensors].values - m) / s
        else:
            mu, sd = self.stats
            out[self.sensors] = (df[self.sensors] - mu) / sd
        return out

def prepare(fd, seed=42, by_regime=None):
    """Full pipeline: load -> regime -> split -> normalize.
    Returns dict with fit/cal/val frames (normalized) and the fitted objects."""
    by_regime = by_regime if by_regime is not None else fd in ('FD002', 'FD004')
    df = load(fd)
    ra = None
    if by_regime:
        ra = RegimeAssigner().fit(df)
        df = ra.assign(df)
    else:
        df = df.copy(); df['regime'] = 0
    fit_u, cal_u, val_u = split_units(df.unit.unique(), seed)
    fit = df[df.unit.isin(fit_u)]
    norm = Normalizer(by_regime=by_regime).fit(fit[fit.rul > HEALTHY_RUL])
    out = {k: norm.transform(df[df.unit.isin(u)])
           for k, u in [('fit', fit_u), ('cal', cal_u), ('val', val_u)]}
    out.update(units=(fit_u, cal_u, val_u), normalizer=norm, regime_assigner=ra)
    return out
