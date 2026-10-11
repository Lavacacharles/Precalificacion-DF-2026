# Generator hypothesis and open questions

## 1. Best current reconstruction (pseudo-code)

Parts marked `??` are unknown and are the targets of the work.

```python
# ---------- population ----------
N_month = {1: 10400, 2: 9800, ..., 11: 9500, 12: 9900}   # round targets, total 120,000 (F3)

def new_client():                                          # every column independent (F5, F7, F8)
    edad        = randint(22, 71)
    ingresos    = max(18000, normal(~65e3, ~21.7e3))       # cap ~145.7k observed
    ratio_deuda = clip(beta(2.33, 4.77), 0.02, 0.95)
    antig_cuenta= randint(1, 179); antig_dir = randint(1, 239)
    productos   = choice([1..5], p=[.419,.290,.176,.077,.039])
    saldo       = clip(?? (mean 26.4k, sd 16.1k), 300, 90000)
    dias_trans  = randint(1, 364)
    visitas     = poisson(8)
    distancia   = min(80, lognormal(2.006, 0.791))
    dia_pago    = randint(1, 28)
    categoricals/bools ~ independent draws with the shares in F8
    return profile                                          # frozen for life (F6)

clients = [new_client() for _ in range(10400)]            # January stock
for t in 1..11:
    if t > 1: clients += [new_client() for _ in range(N_month[t] - len(clients))]   # fillers (F4)
    for c in clients:
        p = base_rule(c.profile) * month_factor(t)          # or on the logit scale: ?? (Q2)
        c.objetivo = bernoulli(p)
        # corrupted column (F12-F14)
        c.dias_ultima_interaccion = c.dias_trans if rand() > (t-1)/11 \
                                    else sample(dias_trans of this month's rows)
        emit_row(c, t)
    clients = [c for c in clients if c.objetivo == 0]      # converters leave (F1, F2)

# ---------- base_rule: current best description (F10, F11, F24) ----------
def base_rule(x):
    if x.risk == "high":
        r = 0.141 if x.dias_trans <= ~180 else 0.049        # soft transition ~172-190 (F11a) ??
        r += small products slope ??                         # F11f
        if x.canal == "branch": r -= ~0.04                   # F24
    elif x.risk == "low":
        r = f(productos, app)                               # F11b table; jump at 3 products; app only if >=2 products
        r += 0.036 * card                                    # F11c
        r -= slope * dias_trans                              # ~21.7% -> 16.2% over 1..364 (F11d)
        r -= ~0.026 * (ratio_deuda > ~0.54)                  # F11e
    else:  # medium
        r = ~0.14
        r -= slope * dias_trans                              # ~16.9% -> 12.7% (F11d)
        r -= ~0.027 * (ratio_deuda > ~0.58)                  # F11e
        r += small products slope ??                         # F11f
    return r
```

How well this already works: a logistic regression with these terms (risk band interacted with
products≥2, products≥3, transaction days, >180 days, card and debt>0.55, plus risk × products≥2 × app,
**without** branch × high) gives Gini 0.262 on Jan–Oct and **0.241 on a November holdout**. That matches the
tuned boosting models (`04_prior_results.md`).

## 2. Open questions, ranked by expected value for December ranking

| # | Question | Why it matters | Suggested test |
|---|---|---|---|
| Q1 | **Exact shape of each band's rule.** Steps or linear? Where exactly are the thresholds (180? products ≥ 3? debt 0.54 vs 0.55 vs 0.58)? Is the high-risk "soft step" a sigmoid, or a hard step blurred by another variable? | The main lever for ranking | Profile likelihood over the threshold position using `tables/recency_by_day_and_risk.csv` and `debt_by_bin_and_risk.csv`. Compare step, linear, piecewise and logistic shapes by log-likelihood / AIC. |
| Q2 | **Which scale combines the effects?** Additive in probability, multiplicative, or additive in logit? E.g. is the app lift the same in points, or in odds, across product levels? | Determines how to combine the rules | Fit additive-probability (identity-link binomial), log-link and logit models to `tables/cells_rule_vars.csv`, then compare deviance. Look for a clean structure, e.g. round coefficients. |
| Q3 | **Is anything missing?** Check the weak candidates in F26 (account age in low risk, ages 61–65, region × drivers p = 0.04). Look for other gated rules like F24: run every variable × every band, and every variable × products≥3 / app / card cells. | A hidden rule like branch × high was found late; more may exist | Per band, LR test of each variable with Bonferroni; replicate on Jan–May vs Jun–Nov halves before accepting. |
| Q4 | **Are the effects deterministic rules on a latent score?** E.g. the generator may compute a score s = Σ w·features, set p = sigmoid(s), and the risk band may itself be a bin of a hidden variable. | If p comes from a smooth score, smooth models fit better than step rules | Compare a GAM/monotone-spline fit against step-rule fits on held-out months. Check whether the rate inside a cell depends on the continuous variable that defines the cell. |
| Q5 | **Month factor** (F27): a fixed trend, re-centring to a monthly target rate (~15%), or random shocks? | Irrelevant to the December Gini (a constant within the month). Matters only for calibration. | Fit "target rate per month" vs trend vs free effects; compare the implied conversions per month with round numbers. |
| Q6 | How the corrupted column's replacements are drawn: from the whole month's transaction values, or only from the overwritten rows' own values? | Low priority; the column is dropped either way | Compare the multiset of replacements against the month's transaction multiset. |
| Q7 | The `saldo_promedio` distribution family (looks like normal-ish with clipping, skew 0.29) and the income cap of ~145.7k (clip or tail?). | Cosmetic; these columns don't affect the target | Distribution fits. |

## 3. What would *not* be productive

- Hyper-parameter tuning of boosting. The three model families tried by the team agree within one standard deviation.
- Social, seasonal or Peru-specific features: none exist in the data (F19, F28).
- Duration features (F17) and `dias_ultima_interaccion` (F16).
- Interactions between independent noise columns (F5).

## 4. Useful facts for scoring

- Expected ceiling: even the true `p(profile)` cannot rank much better than ~0.26–0.27, because most clients sit in
  cells with 13–19% rates and the outcome is a coin flip. Gains from a perfect reconstruction are probably
  **+0.005 to +0.02 Gini**, not more. State this to the user up front.
- December contains November survivors, selected toward low rates, plus fresh entrants drawn from the same
  population (F4). Rank both with the same `base_rule`. The month factor cancels.
- Month-to-month Gini is noisy: per-month folds vary ±0.01–0.02 (Elmer's November holdout 0.230 vs his fold mean 0.258).
  Judge reconstructions by **log-likelihood on all held-out months together**, not by one month's Gini.

## 5. Tables provided (`tables/`)

All tables hold counts: `rows` = client-months exposed, `conv` = conversions, `rate` = conv/rows.

| File | Grain | Use |
|---|---|---|
| `cells_rule_vars.csv` | risk × products(1–5) × `rec30` (30-day bin start) × app × card × debt>0.55 × branch (2,107 non-empty cells) | fit and compare rule structures (Q2) |
| `recency_by_day_and_risk.csv` | risk × each day 1–364 | threshold and shape search (Q1) |
| `debt_by_bin_and_risk.csv` | risk × debt in 0.01 bins | debt threshold (Q1) |
| `month_by_k.csv` | calendar month × months-in-panel | month factor, duration (Q5) |
| `myattemp/viz/data/clients.csv` | one row per client: static profile + `t0`, `t1`, `n_months`, `converted` | survival-style likelihoods |

Regenerate with `python myattemp/context_for_reverse_engineering/tables/make_tables.py`.
