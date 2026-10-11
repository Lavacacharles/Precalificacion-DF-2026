"""UMAP projections colored by the target label (objetivo 0/1), one figure per model design.

Same embeddings as umap_plots.py (Nov fold; cached in out/umap_cache/).
Per figure (rows: train sample Jan-Oct, test month Nov):
  both labels overlaid (random drawing order) | non-converters only | converters only
Plus figures/umap_labels_overview.png: every design, Nov, overlaid labels.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed

from common import FIG, load_train
from models import MODEL_NAMES
from umap_plots import embed_cached

C0, C1 = "#b8c4cc", "#d62728"


def overlay(a, E, y, s=2):
    """Points in random order, so neither label hides the other and the color mix shows the real proportion."""
    o = np.random.default_rng(0).permutation(len(y))
    a.scatter(E[o, 0], E[o, 1], c=np.where(y[o] == 1, C1, C0), s=s, rasterized=True)
    a.scatter([], [], c=C0, s=12, label=f"0: no conversion (n={(y == 0).sum():,})")
    a.scatter([], [], c=C1, s=12, label=f"1: conversion (n={(y == 1).sum():,})")


def draw(r):
    fig, ax = plt.subplots(2, 3, figsize=(16, 10.5), sharex=True, sharey=True)
    for row, (E, y, lab) in enumerate([(r["Es"], r["ys"], "train sample (Jan-Oct)"),
                                       (r["Et"], r["yt"], "test month (Nov)")]):
        overlay(ax[row, 0], E, y)
        ax[row, 0].legend(markerscale=1, fontsize=8, loc="best")
        ax[row, 0].set_title(f"{lab}: both labels  (rate {y.mean():.1%})")
        ax[row, 1].scatter(E[y == 0, 0], E[y == 0, 1], c=C0, s=2, rasterized=True)
        ax[row, 1].set_title(f"{lab}: label 0 only")
        ax[row, 2].scatter(E[y == 1, 0], E[y == 1, 1], c=C1, s=2, rasterized=True)
        ax[row, 2].set_title(f"{lab}: label 1 only")
    for a in ax.flat:
        a.set_xticks([]); a.set_yticks([])
    variant = "with weak candidates" if r["weak"] else "base (7 inputs)"
    fig.suptitle(f"{r['name']} - {variant} - UMAP colored by the target label", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG / f"umap_labels_{r['name']}_{'weak' if r['weak'] else 'base'}.png", dpi=110)
    plt.close(fig)


def main():
    d = load_train()
    jobs = [(n, w) for n in MODEL_NAMES for w in (False, True)]
    res = Parallel(n_jobs=8)(delayed(embed_cached)(n, w, d) for n, w in jobs)
    for r in res:
        draw(r)
    fig, ax = plt.subplots(2, len(MODEL_NAMES), figsize=(4 * len(MODEL_NAMES), 8.5))
    for r in res:
        a = ax[int(r["weak"]), MODEL_NAMES.index(r["name"])]
        overlay(a, r["Et"], r["yt"], s=1)
        a.set_title(f"{r['name']} ({'weak' if r['weak'] else 'base'})", fontsize=10)
        a.set_xticks([]); a.set_yticks([])
    ax[0, 0].legend(markerscale=1, fontsize=8, loc="lower left")
    fig.suptitle("Test month (Nov): UMAP of each model design, colored by the target label", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG / "umap_labels_overview.png", dpi=110)
    print("done")


if __name__ == "__main__":
    main()
