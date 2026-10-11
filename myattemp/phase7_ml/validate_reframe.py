"""Phase 7, step 0: can the problem be re-framed as "how many monthly attempts until a client converts"?

Checks on train.csv only:
  A. Panel bookkeeping per client: first/last month, attempts (months seen), converted or not, gaps, rows after a 1.
  B. Who never converts: share still waiting in November (censored = the story is cut by the end of the data).
  C. Does the chance of converting change with the attempt number? Monthly conversion rate by attempt k,
     raw and after controlling for the profile and the calendar month (logistic check).
  D. Geometric check: if every client has a fixed monthly probability p, attempts-until-conversion is geometric.
     Simulate each client's attempts from a fitted p (same entry month, same censoring at Nov) and compare.
Outputs: out/clients.csv, out/checks.txt, figures/attempts.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import expit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "phase2_ml"))
from models import RuleGLM  # noqa: E402

ROOT = HERE.parents[1]
OUT, FIG = HERE / "out", HERE / "figures"
LAST = 202611
log_lines = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    log_lines.append(s)


def month_index(mes):
    return (mes // 100 - 2026) * 12 + mes % 100  # Jan 2026 = 1


def main():
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    d = pd.read_csv(ROOT / "data" / "train.csv").sort_values(["id_cliente", "mes"]).reset_index(drop=True)
    d["t"] = month_index(d["mes"])
    d["attempt"] = d.groupby("id_cliente").cumcount() + 1

    # ---------------- A. bookkeeping
    g = d.groupby("id_cliente")
    c = pd.DataFrame({"first": g.t.min(), "last": g.t.max(), "attempts": g.size(), "converted": g.objetivo.max(),
                      "n_ones": g.objetivo.sum()})
    c["gaps"] = (c["last"] - c["first"] + 1) - c["attempts"]
    one_not_last = (d.objetivo == 1) & (d.t != d.groupby("id_cliente").t.transform("max"))
    left_without = (c.converted == 0) & (c["last"] < month_index(LAST))
    log("A. Panel bookkeeping")
    log(f"  clients: {len(c):,}   rows: {len(d):,}")
    log(f"  clients with more than one 1: {(c.n_ones > 1).sum()}")
    log(f"  rows with a 1 that are not the client's last row: {one_not_last.sum()}")
    log(f"  clients with gaps (missing months between first and last): {(c.gaps > 0).sum()}")
    log(f"  clients who disappear before Nov without converting: {left_without.sum()}")

    # ---------------- B. who never converts
    log("\nB. Converted vs still waiting at the end of November (censored)")
    log(f"  converted: {c.converted.sum():,} ({c.converted.mean():.1%})   "
        f"never converted (all still present in Nov): {(1 - c.converted).sum():,} ({1 - c.converted.mean():.1%})")
    coh = c.groupby("first").agg(clients=("converted", "size"), converted=("converted", "mean"),
                                 max_attempts=("attempts", "max"))
    coh.index = [f"entry month {i}" for i in coh.index]
    log(coh.round(3).to_string())
    log(f"  attempts among converters: mean {c[c.converted == 1].attempts.mean():.2f}, "
        f"median {c[c.converted == 1].attempts.median():.0f}, max {c.attempts.max()}")
    c.to_csv(OUT / "clients.csv")

    # ---------------- C. rate by attempt number
    log("\nC. Monthly conversion rate by attempt number")
    by_k = d.groupby("attempt").objetivo.agg(["mean", "size"])
    log(by_k.round(4).T.to_string())
    glm = RuleGLM("logit").fit(d)
    X = glm._X(d)  # rule features + month fixed effects (in-sample, diagnostic only)
    d["eta"] = X @ glm.w
    d["p_hat"] = expit(d["eta"])
    for label, extra in [("attempt as a number", d[["attempt"]].astype(float)),
                         ("attempt as categories (2..11 vs 1)",
                          pd.get_dummies(d.attempt, prefix="k", drop_first=True).astype(float))]:
        Z = sm.add_constant(pd.concat([d[["eta"]], extra], axis=1))
        r = sm.GLM(d.objetivo, Z, family=sm.families.Binomial()).fit()
        r0 = sm.GLM(d.objetivo, sm.add_constant(d[["eta"]]), family=sm.families.Binomial()).fit()
        lr = 2 * (r.llf - r0.llf)
        log(f"  profile + month known, adding {label}: likelihood-ratio = {lr:.1f} on {extra.shape[1]} df"
            + (f"; slope per attempt {r.params['attempt']:+.4f} (z = {r.tvalues['attempt']:.2f})" if 'attempt' in r.params else ""))
    # observed / expected by attempt, where expected uses profile + month
    oe = d.groupby("attempt").agg(observed=("objetivo", "sum"), expected=("p_hat", "sum"))
    oe["obs/exp"] = oe.observed / oe.expected
    log("  observed / expected conversions by attempt (expected from profile + month):")
    log(oe["obs/exp"].round(3).to_frame().T.to_string())

    # ---------------- D. geometric simulation with the same entry months and censoring
    log("\nD. Geometric check: simulate each client's attempts from its fitted monthly p")
    first_rows = d[d.attempt == 1].set_index("id_cliente")
    month_shift = {m: w for m, w in zip(glm.months[1:], glm.w[len(glm.feat_names):])}
    base_eta = first_rows["eta"] - first_rows["mes"].map(month_shift).fillna(0)  # profile part only
    rng = np.random.default_rng(0)
    months = sorted(d.mes.unique())
    sims = []
    for rep in range(20):
        att = np.zeros(len(first_rows), int)
        conv = np.zeros(len(first_rows), int)
        # walk forward month by month: each waiting client converts with its own p
        alive = np.ones(len(first_rows), bool)
        start = first_rows["mes"].values
        for m in months:
            act = alive & (start <= m)
            p = expit(base_eta.values + month_shift.get(m, 0.0))
            hit = act & (rng.random(len(p)) < p)
            att += act
            conv |= hit
            alive &= ~hit
        sims.append(pd.DataFrame({"attempts": att, "converted": conv, "rep": rep}))
    S = pd.concat(sims)
    log(f"  converted share: observed {c.converted.mean():.3f}, simulated {S.converted.mean():.3f} "
        f"(± {S.groupby('rep').converted.mean().std():.3f} between simulations)")
    obs_dist = c[c.converted == 1].attempts.value_counts(normalize=True).sort_index()
    sim_dist = S[S.converted == 1].attempts.value_counts(normalize=True).sort_index()
    log("  attempts-until-conversion distribution (converters), observed vs simulated:")
    log(pd.DataFrame({"observed": obs_dist, "simulated": sim_dist}).round(3).T.to_string())

    open(OUT / "checks.txt", "w", encoding="utf-8").write("\n".join(log_lines))
    draw(c, by_k, oe, obs_dist, sim_dist, coh)


def draw(c, by_k, oe, obs_dist, sim_dist, coh):
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    a = ax[0]
    ct = c.groupby("first").converted.agg(["sum", "size"])
    a.bar(ct.index, ct["sum"], color="#2a6fdb", label="converted by Nov")
    a.bar(ct.index, ct["size"] - ct["sum"], bottom=ct["sum"], color="#c9d3db", label="still waiting in Nov (censored)")
    a.set_xticks(ct.index)
    a.set_xlabel("entry month (2026)")
    a.set_ylabel("clients")
    a.set_title("Clients by entry month: converted or still waiting")
    a.legend(frameon=False)
    a = ax[1]
    a.plot(by_k.index, by_k["mean"] * 100, color="#8a99a6", marker="o", lw=2, label="raw rate")
    a.plot(oe.index, oe["obs/exp"] * by_k["mean"].iloc[0] * 100 / oe["obs/exp"].iloc[0], color="#2a6fdb",
           marker="o", lw=2, label="after profile + month (scaled to attempt 1)")
    a.set_xlabel("attempt number (months the client has been offered)")
    a.set_ylabel("monthly conversion rate (%)")
    a.set_title("Does the chance change with each new attempt?")
    a.set_xticks(by_k.index)
    a.legend(frameon=False)
    a = ax[2]
    k = obs_dist.index
    a.bar(k - 0.2, obs_dist.values, width=0.4, color="#2a6fdb", label="observed")
    a.bar(sim_dist.index + 0.2, sim_dist.values, width=0.4, color="#e8833a", label="simulated: fixed p per client")
    a.set_xticks(k)
    a.set_xlabel("attempts until conversion (converters only)")
    a.set_ylabel("share of converters")
    a.set_title("Fixed monthly chance per client reproduces the tries")
    a.legend(frameon=False)
    for a in ax:
        a.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(FIG / "attempts.png", dpi=110)


if __name__ == "__main__":
    main()
