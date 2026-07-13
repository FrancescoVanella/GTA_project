"""Shared plotting utilities for the C-MAPSS project notebooks."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent.parent / 'results'
FIGURES_DIR = RESULTS_DIR / 'figures'

COLORS = {'healthy': '#2ca02c', 'degraded': '#ff7f0e', 'anomalous': '#d62728'}
RUL_CMAP = 'RdYlGn'


def save_fig(name, dpi=120, tight=True):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if tight:
        plt.tight_layout()
    plt.savefig(FIGURES_DIR / name, dpi=dpi)


def rul_scatter(ax, x, y, rul, vmax=150, cmap=RUL_CMAP, s=3, **kw):
    sc = ax.scatter(x, y, c=rul, cmap=cmap, s=s, vmax=vmax, **kw)
    return sc


def regime_palette(n=6):
    return plt.cm.tab10(np.linspace(0, 1, n))


def score_heatmap(corr_df, ax=None, cmap='viridis', vmin=0, vmax=1):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr_df.values, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xticks(range(len(corr_df)))
    ax.set_xticklabels(corr_df.columns, rotation=45, ha='right')
    ax.set_yticks(range(len(corr_df)))
    ax.set_yticklabels(corr_df.columns)
    for i in range(len(corr_df)):
        for j in range(len(corr_df)):
            ax.text(j, i, f'{corr_df.iloc[i, j]:.2f}',
                    ha='center', va='center', c='w', fontsize=8)
    plt.colorbar(im, ax=ax)
    return ax
