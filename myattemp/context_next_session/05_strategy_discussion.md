# Strategy discussion

The user raised two doubts at the end of the EDA phase:
1. Could the clients be a **selected target group** rather than an independent sample, which would explain the odd behavior?
2. Is **feature engineering** worth it, given that the data doesn't behave like reality, and that "fitting a function to random noise" seems pointless?

The previous session's answers follow, then the options for the next phase.

## 1. Selected target group vs independent sample

### What selection *can* explain

If the panel is a campaign list (BCP selects a segment, e.g. pre-approved clients, and follows them monthly until they accept, with a fixed contact capacity), these anomalies become **design choices**:

| Observation | Explained by a targeted program? |
|---|---|
| Clients leave only by converting | yes: follow-up until response |
| Frozen profiles | yes: attributes snapshotted at list entry (ac's audit supports an *entry* snapshot) |
| Round monthly sizes with fillers | yes: fixed monthly capacity |
| ~15% conversion every month | plausible for a pre-approved offer |
| No gratificación / CTS effect | plausible if the offer doesn't depend on cash |
| Level anomalies (iOS 43%, narrow ages) | plausible for an affluent segment |

### What selection *cannot* explain

- **Selection creates dependence; it doesn't remove it.** Selecting on some variables makes them *more* related in the kept group (Berkson's paradox). A real population with linked age, income, occupation, debt and products cannot be filtered into one where 210 pairwise tests look like pure chance without an absurdly engineered filter.
- Textbook marginals (uniform age, Poisson visits, symmetric income with a hard floor), the exact (t−1)/11 overwrite schedule, and clean rules that switch off completely outside one band.
- **Direction of generation:** if a privacy synthesizer had drawn features *from* the target (naive-Bayes style), effects would be additive on the logit scale. The observed true interactions (app only with ≥2 products, card only in low risk, debt only without the app) imply **features first, target computed by rules**.

### Synthesis

A real-looking **program design** (snapshot at entry, follow-up until conversion, monthly capacity) + **independently simulated clients** + **a hand-written conversion rule**. The design may mirror BCP practice; the population does not.

Optional test, not yet run: selection on risk band would induce dependence *within* bands. Check pairwise dependence inside each `banda_riesgo` (and inside rule cells). Finding none strengthens "independent sample".

## 2. Is feature engineering worth it?

### Against (the user's view, mostly right)

- Columns are independent; ratios or products of noise columns are noise.
- Real-world-motivated features assume dependencies that don't exist here.
- No feature can carry information the generator didn't use. The ceiling is the true probability per profile (≈0.26 Gini).
- Month-to-month Gini noise (SE ≈ 0.009, swings ±0.01–0.02) exceeds any plausible gain; many features make it easy to be fooled by a lucky validation month.
- Empirical confirmation: ac tested income/age, device × income and entry-snapshot clocks: ΔAUC −0.001 (CI includes 0); Elmer's 54 candidates reduced to 7; Rick's 70+ experiments: nothing significant.

### For, in a different sense: generator-shaped features

- The target is not pure noise: it is a **coin flip with a real probability behind it**, which depends on 7 variables. The goal is to recover that function, not the coin flips.
- Threshold indicators (>180 days, ≥3 products, debt > 0.55) and gates (band × products × app, app × debt, band × branch) let a simple model express the true rule in ~20 parameters.
- They add no information a tree couldn't find; their value is **lower variance and more stable rankings**. Evidence: Jorge's logistic with threshold rules ≥ boosting; Rick: depth-3 trees suffice.

### The bigger lever: removing features

Dropping the 14 flat variables and `dias_ultima_interaccion` likely helps as much as anything added: fewer ways to overfit, and protection against a column that becomes pure noise in December.

### Verdict

Domain feature engineering: **not worth it**. Generator-shaped features: a small, bounded effort, for variance reduction. Total plausible upside: **+0.005 to +0.02 Gini**, comparable to the measurement noise.

## 3. Other strategies, roughly ordered by value

1. **Model only what's there, then stop.** Fit a small rule-shaped model on all 11 months (the rules don't drift; include a month fixed effect). Check the leftovers per cell: if they look like coin-flip noise, there is nothing more to find, and that conclusion is defensible.
2. **Cell lookup with shrinkage.** Estimate the rate per rule cell, pulling small cells toward their parent (band × products), e.g. empirical Bayes. Transparent and low-variance.
3. **Blend** the rule model with a regularized boosting model (Jorge's best, 0.269). Different errors, lower variance.
4. **Choose for stability, not peak.** Select by average log-loss across all held-out months; use paired bootstrap tests (same rows) for any comparison.
5. **No history features** (D8), **no seasonal features** (D9).
6. **Explanation as an asset.** If the competition also judges methodology or the pitch: "the data is generated; here are the rules; here is proof the rest is noise; here is why more features can't help" beats another 0.003 of Gini.

## 4. Options for the next session (the user will choose)

### Option A: the ML / submission part

1. Build features: the 7 inputs + explicit gates (see 04). Drop everything else.
2. Models: (a) rule logistic regression, (b) cell lookup with shrinkage, (c) shallow regularized LightGBM/CatBoost (depth ≤ 3) on the 7 inputs, (d) rank-average blend.
3. Validation: walk-forward Jul, Aug, Sep, Oct, Nov (train on all earlier months); report the mean and per-month Gini; log-loss; paired bootstrap vs the best teammate model (Jorge's ensemble ≈ 0.269 on Jul–Oct).
4. Final fit on Jan–Nov with month fixed effects (December uses any month level; the ranking is unaffected); predict `test.csv`; write `id_cliente,prediccion` in test order.
5. Sanity checks: 9,900 rows, no NaN, predictions in [0,1], order matches `sample_submission.csv`; compare the ranking with teammates' submissions (rank correlation).

### Option B: continue exploring / reverse engineering

1. Run the within-band dependence test (§1) to close the selection question.
2. Fit the combination scale (additive vs multiplicative vs logit) on `context_for_reverse_engineering/tables/cells_rule_vars.csv`. The table already holds `activo_movil` and `debt_gt_055` as columns, so the app-gated debt rule can be fitted directly.
3. Locate thresholds by profile likelihood (`recency_by_day_and_risk.csv`, `debt_by_bin_and_risk.csv`); hard vs soft steps.
4. Systematic search for further gated rules: each variable × each band × app cell, Bonferroni, replicated on Jan–May vs Jun–Nov.
5. Residual test: a calibration chi-square per cell for the final rule model. If consistent with binomial noise, stop.

Either option ends with a short report in plain language for the user.
