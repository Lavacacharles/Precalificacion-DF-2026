"""UMAP pictures of each model's structure, on the November fold (train Jan-Oct, test Nov).

Each model is embedded in its OWN representation of a client:
  M1_* : per-feature contributions to the linear predictor (coefficient x feature)
  M2   : one-hot of the cell (band, products, app, card, debt>0.55, recency bin, branch [, weak flags])
  M3   : SHAP contributions (log-odds)
  M4   : the raw 7 inputs, standardized (a blend has no space of its own)
UMAP is fitted on a train sample and the test month is projected into the same map,
so train and test panels are directly comparable.

Panels per figure (rows: train sample, test month):
  model score (percentile) | observed conversion rate per hexagon | risk band
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler

from common import FIG, load_train, rank01, gini
from models import make, MODEL_NAMES

TEST_MONTH = 202611
N_TRAIN = 12000
BAND_COLORS = np.array(["#2a9d8f", "#e9c46a", "#e76f51"])


CACHE = FIG.parent / "out" / "umap_cache"


def embed_cached(name, weak, d):
    """embed(), stored as out/umap_cache/<name>_<variant>.npz so other plots can reuse it."""
    f = CACHE / f"{name}_{'weak' if weak else 'base'}.npz"
    if f.exists():
        z = np.load(f)
        return dict(name=name, weak=weak, **{k: z[k] for k in z.files})
    r = embed(name, weak, d)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(f, **{k: v for k, v in r.items() if k not in ("name", "weak")})
    return r


def embed(name, weak, d):
    import umap
    tr, te = d[d.mes < TEST_MONTH], d[d.mes == TEST_MONTH]
    mdl = make(name, weak).fit(tr)
    smp = tr.sample(N_TRAIN, random_state=0)
    Rs, Rt = mdl.represent(smp), mdl.represent(te)
    if name == "M4_blend":
        sc = StandardScaler().fit(Rs)
        Rs, Rt = sc.transform(Rs), sc.transform(Rt)
    jitter = np.random.default_rng(0)
    Rs = Rs + jitter.normal(0, 1e-4, Rs.shape)  # many clients share an identical representation
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.15, random_state=0)
    Es = reducer.fit_transform(Rs)
    Et = reducer.transform(Rt)
    return dict(name=name, weak=weak, Es=Es, Et=Et,
                ps=mdl.predict(smp), pt=mdl.predict(te),
                ys=smp.objetivo.values, yt=te.objetivo.values,
                bs=smp.banda_riesgo.map({"low": 0, "medium": 1, "high": 2}).values,
                bt=te.banda_riesgo.map({"low": 0, "medium": 1, "high": 2}).values)


def draw(r):
    fig, ax = plt.subplots(2, 3, figsize=(16, 10.5))
    for row, (E, p, y, b, lab) in enumerate([(r["Es"], r["ps"], r["ys"], r["bs"], f"train sample (Jan-Oct, n={len(r['ys']):,})"),
                                             (r["Et"], r["pt"], r["yt"], r["bt"], f"test month (Nov, n={len(r['yt']):,})")]):
        a = ax[row, 0]
        sc = a.scatter(E[:, 0], E[:, 1], c=rank01(p), s=2, cmap="viridis", rasterized=True)
        fig.colorbar(sc, ax=a, label="model score (percentile)")
        a.set_title(f"{lab}\nmodel score" + (f"  |  Gini {gini(y, p):.3f}" if row == 1 else ""))
        a = ax[row, 1]
        hb = a.hexbin(E[:, 0], E[:, 1], C=y, reduce_C_function=np.mean, gridsize=35, mincnt=15,
                      cmap="magma", vmin=0, vmax=0.4)
        fig.colorbar(hb, ax=a, label="observed conversion rate")
        a.set_title("observed conversion rate (hexagons with 15+ rows)")
        a = ax[row, 2]
        a.scatter(E[:, 0], E[:, 1], c=BAND_COLORS[b], s=2, rasterized=True)
        for i, lb in enumerate(["low", "medium", "high"]):
            a.scatter([], [], c=BAND_COLORS[i], label=f"{lb} risk")
        a.legend(markerscale=4, loc="best", fontsize=8)
        a.set_title("risk band")
    for a in ax.flat:
        a.set_xticks([]); a.set_yticks([])
    variant = "with weak candidates" if r["weak"] else "base (7 inputs)"
    fig.suptitle(f"{r['name']} - {variant} - UMAP of the model's own representation", fontsize=14)
    fig.tight_layout()
    out = FIG / f"umap_{r['name']}_{'weak' if r['weak'] else 'base'}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print("wrote", out.name, flush=True)


def main():
    FIG.mkdir(exist_ok=True)
    d = load_train()
    names = sys.argv[1:] or MODEL_NAMES
    jobs = [(n, w) for n in names for w in (False, True)]
    res = Parallel(n_jobs=min(8, len(jobs)))(delayed(embed_cached)(n, w, d) for n, w in jobs)
    for r in res:
        draw(r)


if __name__ == "__main__":
    main()
