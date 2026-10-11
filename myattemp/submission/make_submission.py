"""December submission: rank blend of the rule logistic (M1, logit) and the monotone shallow LightGBM (M3), phase 2.

1. Dry run: train Jan-Oct, predict November with exactly the same procedure, report the Gini (sanity check).
2. Final: train Jan-Nov (all 11 months, no resampling), predict test.csv (December).
Inputs: the 7 rule variables (inside the phase-2 models). `dias_ultima_interaccion` never used.

Calibration: the blend gives an order. To output probabilities, the blended ranks are mapped onto the rule logistic's own
December probabilities (quantile mapping: the client ranked k-th by the blend receives the k-th smallest M1 probability).
The order is the blend's, the values are M1's probability scale at the most recent monthly level (November's month effect),
so the mean prediction stays near the recent conversion rate.
Output: submission.csv (id_cliente, prediccion), test.csv row order.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "phase2_ml"))
from models import RuleGLM, ShallowGBM  # noqa: E402


def rank01(p):
    return pd.Series(p).rank(pct=True).values


def m1_prob_latest(m1, d_new):
    """M1 probability at the level of the last training month (M1.predict uses the first month as reference)."""
    eta = m1._X(d_new, months=False) @ m1.w[:len(m1.feat_names)]
    last = m1.months[-1]
    shift = m1.w[len(m1.feat_names) + m1.months[1:].index(last)] if last != m1.months[0] else 0.0
    return expit(eta + shift)


def blend(train, new):
    m1 = RuleGLM("logit").fit(train)
    m3s = [ShallowGBM(seed=s).fit(train) for s in range(3)]  # 3 seeds averaged (subsampling noise)
    p1 = m1.predict(new)
    p3 = np.mean([m.predict(new) for m in m3s], axis=0)
    r = (rank01(p1) + rank01(p3)) / 2
    scale = np.sort(m1_prob_latest(m1, new))
    order = np.argsort(np.argsort(r, kind="mergesort"), kind="mergesort")  # 0..n-1 position of each row
    return scale[order], p1, p3


def gini(y, p):
    return 2 * roc_auc_score(y, p) - 1


def main():
    tr = pd.read_csv(ROOT / "data" / "train.csv")
    te = pd.read_csv(ROOT / "data" / "test.csv")

    # 1. dry run on November
    a, b = tr[tr.mes < 202611], tr[tr.mes == 202611]
    pb, p1, p3 = blend(a, b)
    print(f"dry run, Nov (train Jan-Oct): Gini blend {gini(b.objetivo, pb):.4f} | M1 {gini(b.objetivo, p1):.4f} | "
          f"M3 {gini(b.objetivo, p3):.4f} | mean prediction {pb.mean():.3f} vs observed {b.objetivo.mean():.3f}")

    # 2. final
    p, p1, p3 = blend(tr, te)
    sub = pd.DataFrame({"id_cliente": te["id_cliente"].values, "prediccion": p})
    ss = pd.read_csv(ROOT / "data" / "sample_submission.csv")
    assert len(sub) == len(te) == len(ss), "row count"
    assert (sub.id_cliente.values == ss.id_cliente.values).all(), "row order / ids differ from sample_submission"
    assert sub.prediccion.between(0, 1).all() and sub.prediccion.notna().all(), "range"
    assert list(sub.columns) == list(ss.columns), "columns"
    sub.to_csv(HERE / "submission.csv", index=False)

    survivor = te.id_cliente.isin(tr.id_cliente).values
    rho = pd.Series(p1).corr(pd.Series(p3), method="spearman")
    print(f"December: {len(sub):,} rows ({survivor.sum():,} still waiting since earlier months, {(~survivor).sum():,} new)")
    print(f"  prediccion: mean {p.mean():.3f}, min {p.min():.3f}, median {np.median(p):.3f}, max {p.max():.3f}")
    print(f"  mean prediction: waiting clients {p[survivor].mean():.3f}, new clients {p[~survivor].mean():.3f}")
    print(f"  rank agreement M1 vs M3 (Spearman): {rho:.3f}")
    print("wrote", HERE / "submission.csv")


if __name__ == "__main__":
    main()
