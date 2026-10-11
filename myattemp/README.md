# myattemp: understanding the variables before predicting

The goal here is to understand what the panel describes and whether each variable behaves the
way its name promises, before any model is fitted. Source: `data/train.csv` only.

> **Status (2026-10-07):** EDA (§1–§4) and seven modeling phases are done. Every approach ends at Gini ≈ 0.25–0.26,
> the ceiling set by the data. **§5 summarizes all the tries; §6 is the submission recommendation.**

> **Start here:** [`02_interaccion_y_objetivo.md`](02_interaccion_y_objetivo.md) has a plain-language glossary,
> the full `dias_ultima_interaccion` mechanism, what `objetivo` is, and the synthetic-data evidence.
> [`03_grupos_y_realidad.md`](03_grupos_y_realidad.md) contrasts groups, the Peruvian pay calendar and social effects:
> the verdict is that the data is fully synthetic at the attribute level, and one more hidden rule was found (branch × high risk).
> `python myattemp/validate.py` re-derives every claim (29/29 PASS).
> In this README "hazard" = **monthly conversion rate** (conversions ÷ rows of a group).

## How to run

```bash
python myattemp/eda.py                 # regenerates viz/data/clients.csv + viz/data/eda.json
cd myattemp/viz && python -m http.server 8000
# open http://localhost:8000   (tabs are linkable: #panel #audit #integrity #explore #personas)
```

The explorer is plain HTML + d3 v7 from cdnjs, with no build step.

| Tab | What it answers |
|---|---|
| Guide | Glossary, what a row is, and a suggested reading order. |
| 1 · Panel | Who enters and leaves the panel, and when. Survivor selection. |
| 2 · Variable audit | Per variable: what a banker expects vs. what the data shows, distribution and hazard curve. |
| 3 · Integrity | Do variables relate to each other as they should? Frozen values, clipping, the corrupted column. |
| 4 · Segment explorer | Any variable × any split, filtered by risk band and product count. This is where the hidden rules show. |
| 5 · Client types | Personas built only from the variables that carry signal, with hazard and expected months to convert. |
| 6 · dias_ultima_interaccion | The overwritten column, client by client, against its exact `(12 − month)/11` schedule. |
| 7 · Groups vs reality | Independence fingerprint, Peru reality checks, seasonality per group, branch × high-risk rule. |

All rates are **monthly hazards** (conversions ÷ client-months), which is what a row-level
model predicts and what the Gini on December measures.

---

## 1. What the panel really is

| Fact | Evidence |
|---|---|
| 24,628 clients, 110,100 client-months, 15.0% monthly conversion | |
| A client leaves the panel **only by converting**. All 8,061 never-converters are still present in November. | `last month == 11` for every non-converter |
| The conversion is always the client's last row, and histories have no gaps. | |
| 10,400 clients form a January stock. The rest enter later (921–2,043 per month), with ids assigned in entry order. | |
| **21 of 22 features never change for a client across months.** | Tab 3, "Frozen profiles" |

The last row is the important one. `edad`, `antiguedad_cuenta_meses` and
`dias_ultima_transaccion` should move as the months pass: age by up to 1 year, tenure by +1 per
month, and "days since last transaction" by +30 per month for a client who did not transact.
They do not move. Each client is a **static profile copied into every month** they remain
unconverted. The panel is therefore a discrete-time survival process,
`P(convert in month m | still here) = h(profile)`, and not a sequence of evolving behavior.

So:
- `dias_ultima_transaccion` is not a recency. It is a fixed **activity-level trait** drawn once.
- "Months observed" (`k`) lowers the rate only mildly (16% → 13%). Once the profile **and the calendar
  month** are in the model, `k` has no effect at all (likelihood-ratio χ² = 5.8 on 10 df). An earlier
  version of this note claimed obs/pred ≈ 1 without the calendar control. That check failed on re-run
  (up to 1.13 at k = 9) because high k only exists in late months. **Duration features add nothing.**
- The December test population is mostly November survivors, so it is selected toward
  low-hazard profiles. Expect lower base rates and compressed scores.

## 2. Variables that do not mean what their names say

The features were generated **independently of each other**: the maximum |Spearman| between any
two features is 0.021. In real bank data these pairs are strongly linked:

| Check | A real bank would show | Here |
|---|---|---|
| Loan holders' debt-to-income | clearly higher | 0.326 vs 0.328 |
| High vs low risk debt ratio | high risk has more debt | 0.329 vs 0.328 |
| Manager vs manual income | large gap | 65,162 vs 65,224 |
| `es_nuevo_cliente` among 10+ year accounts | ~0% | **18.6%** |
| `es_nuevo_cliente` among accounts ≤ 6 months | ~100% | **17.6%** |
| Account opened before age 18 | ~0% | 8.3% |
| `numero_productos` ≥ card+loan+insurance flags | 100% | 73.8% |
| App-active share on `web`/`otro` device | lower | 68% vs 67% |
| Web visits for `web`-device users | higher | 7.99 vs 7.97 |

Consequences:
- **`banda_riesgo` is not a credit score built from debt or income.** It is an exogenous label, and
  it is the label the outcome was built around (§3).
- Ratios and combinations such as `deuda_por_producto`, `ingreso/edad` or `saldo/ingreso` cannot create
  information. Each component is independent noise, so the combination is noise too unless one of
  the components is itself a signal.
- Distribution shapes are textbook simulations: uniform `edad`, tenure, address, payment day and transaction
  days; normal `ingresos` with a hard floor at 18,000 (1.6% of clients); `saldo` floored at 300 (6.7%)
  and capped at 90,000; Poisson(8) web visits; log-normal distance capped at 80 km.

### The one time-varying column is a corrupted copy

`dias_ultima_interaccion` equals `dias_ultima_transaccion` in **100% of January rows,
falling linearly to 9% in November**: each row is overwritten with probability `(t−1)/11`. When the two
differ, the value is a random draw from that month's transaction values (r = 0.003)
and the hazard is flat across it. Its row-level Gini (−0.058) is borrowed entirely from the
transaction variable, and that borrowed signal fades month by month (Tab 3). The decay pattern
indicates that in December the column will be almost pure noise.
**Drop it, or at most use it as a duplicate of the transaction variable.** A model that learns from it
trains on a relationship that has disappeared by test time.

## 3. Where the signal is: risk-band-gated rules

Only 5–6 variables move the hazard. They act **inside risk bands**, not additively:

| Band (share of clients) | Base hazard | What moves it | What does not |
|---|---|---|---|
| **low** (53%) | 18.6% | products ≥ 3: about +10 pts. App active: +8–10 pts, **only with ≥ 2 products**. Credit card: 16.0% → 19.6%. Linear recency slope (21.7% → 16.2%). Debt > 0.55: −3 pts. | — |
| **medium** (31%) | 14.0% | mild recency slope (16.9% → 12.7%), debt > 0.6: −2 to −3 pts | card (0 effect), app (small) |
| **high** (17%) | 8.7% | **hard step at 180 days**: ≤180d → 14%, >180d → 5% | card (0), app (≈0), products (small) |

Read as client types (Tab 5):

- **Engaged low-risk multi-product clients** (low · 3+ products): 29.5% per month, about 3.4 months to convert.
  With the app active they reach about 33–36%. This is the best type.
- **Low-risk single-product clients**: 13.8%, which is *below the base rate*. Low risk alone does not
  make a good prospect. The low-risk advantage comes from product depth and app use.
- **Medium-risk clients**: nearly flat, between 13% and 15%. Little separates them.
- **Dormant high-risk clients** (no transaction in > 180 days): about 5%, around 20 months to convert. The worst type, and
  separable by one threshold.
- **Recently active high-risk clients** convert at about 14%, as well as an average medium-risk client.

All other variables, including occupation, region, channel, device, income, balance, age, tenure, web
visits, distance, loan, insurance and the "new client" flag, are flat within the confidence intervals,
**within every risk band**.

## 4. Why everyone plateaus at Gini ≈ 0.26–0.27

- With static profiles and a constant hazard per profile, the best possible score for a row is `h(profile)`.
- The strongest single segment still has only a ~30% hazard. Most conversions come from the large middle
  of 13–19% types, which no feature can order.
- An additive logistic regression on the signal variables gets 0.237 in-sample. A logistic with only the
  band-gated rules above, about 20 parameters, gets **0.262 on Jan–Oct and 0.241 on a November holdout**. That is
  the same level as Elmer's tuned LightGBM, which scored 0.230 on November, and Jorge's walk-forward 0.26–0.27.
  The rules carry essentially everything the boosting models found. The remaining gap to 1 is noise by
  construction (a Bernoulli draw each month), not missing feature engineering.

What this means for the next step, before tuning anything:
1. Model **per band, or with explicit band interactions**: `high × (trans > 180)`,
   `low × products≥3`, `low × app × products≥2`, `low × card`, `(low|medium) × debt > 0.55`.
2. Remove `dias_ultima_interaccion` and the 15 flat variables. They can only add variance.
3. Expect duration features (`k`, months seen) to add almost nothing. Their PSI to December is large
   because of calendar effects, not information.
4. December is a survivor-heavy population. Check calibration and ordering on the *November survivors*
   subgroup, which looks like the test, not on all of November.

> Corrections made after §3 was written (details in `context_next_session/02_decisions_log.md`):
> the debt penalty applies **only without the app** (low 16.4 → 10.6%, medium 14.1 → 9.0%), the high-risk drop is a **soft step around
> 172–190 days** (best split 180–182), and **branch-acquired high-risk clients** convert about 4 points less on both sides of it.

## 5. Modeling tries (phases 2–7)

Common rules for every phase: train only; **walk-forward by calendar month** (each test month Jul–Nov is predicted by a model trained on
earlier months only); `dias_ultima_interaccion` never used; differences judged with a **paired bootstrap** (same rows, 95% CI).
A gap below ~0.01 is noise unless its CI excludes zero (one month's Gini has a standard error of ≈ 0.009).
Each folder has its own README with code, numbers and figures.

| Phase | Folder | What was tried | Best mean Gini Jul–Nov | Verdict |
|---|---|---|---|---|
| 2 | `phase2_ml/` | Rule logistic (gates inside each risk band; logit / log / additive links), cell table with shrinkage, shallow monotone LightGBM, rank blend; ± 2 weak candidates | 0.2550 (rule logistic) · 0.2561 (LightGBM) | Rule logistic, LightGBM and blend tied; cell table worse (−0.013). Link choice irrelevant. Weak candidates +0.002, partly selection bias |
| 2b | `phase2_ml/difficulty.py` | How hard is the target? | — | No client reaches 50%; inputs remove ~4% of the outcome's uncertainty; a perfect model would score 0.27 ± 0.015 in a month; learning curve flat after ~20k rows |
| 3 | `phase3_ml/` | 20 bank-practice features (capacity, engagement, product gaps, life cycle, months in panel); XGBoost tuned with Optuna (60 trials per feature set) | 0.2612 tuned CV → 0.2355 Nov holdout (rule7) | Tuning +0.002–0.006 in CV, gone on the holdout (winner's curse). Bank features add nothing. Good region: depth 3–4, learning rate 0.005–0.02, 500–1,200 trees |
| 4 | `phase4_ml/` | Seasonal features: Peru calendar (gratificación, CTS, holidays, business days), cash windfall for salaried clients, payday alignment, anniversary | 0.2550 (vs 0.2560 without) | No gain; client × season features hurt (−0.003). One year = one example per season: July's model learned May's noise |
| 5 | `phase5_ml/` | Latent-space learning with small networks (300–1,600 weights): autoencoder, SCARF (self-supervised contrastive), SupCon (supervised contrastive), each + classification head (frozen or fine-tuned) | 0.249 (plain small network) | No representation beats the plain network; autoencoder keeps age/income and drops the risk band. All networks below the classic models |
| 6 | `phase6_ml/` | All classic models with saved settings (no search) × resampling: none, undersample 1:1 and 1:3, oversample 1:1 | **0.2577 (blend rule logistic + LightGBM + XGBoost)** · 0.2573 (rule logistic + LightGBM) | Resampling never helps; 1:1 undersampling −0.002; oversampling hurts XGBoost with noise columns; probabilities become miscalibrated (14.8% observed → 22–45% predicted) |
| 7 | `phase7_ml/` | Re-framing as "attempts until conversion": data checks + survival models (Cox, Weibull/LogNormal/LogLogistic, XGBoost survival:cox/aft), scored the December way | 0.2552 (Weibull) vs 0.2550 (monthly model) | 32.7% never convert (censored); chance per attempt is flat once the profile is known; a fixed chance per client reproduces the attempt counts exactly. Survival models tie; the hard "0 until predicted attempt" mapping collapses to 0.05 |

What all phases agree on:
* **The signal is 7 inputs** (risk band, products, days since last transaction, app, card, debt-to-income, branch channel) combined as gated rules.
  Every other column, every constructed or seasonal feature and every learned representation adds nothing or adds noise.
* **The ceiling is in the data, not the models.** Each client has one fixed monthly chance (5–45%); the rest is luck.
  Very different models tie within ±0.003, below the measurement noise.
* **Stable beats tuned.** The models that hold up best across months are the simple, shape-constrained ones (rule logistic, monotone LightGBM)
  and their rank blends. Heavily tuned or noise-column models swing more and did worst on the November holdout.

## 6. Recommendation for the submission

* **Model:** rank average of the **rule logistic (M1, logit link)** and the **monotone shallow LightGBM (M3)**, both in `phase2_ml/models.py`,
  optionally adding **XGBoost on the 7 rule inputs** with the phase-3 settings (`phase3_ml/out/best_params.json`).
  Jul–Nov: 0.2573 (two models, steadiest: month-to-month SD 0.006) or 0.2577 (three models, SD 0.008).
* **Inputs:** the 7 rule inputs only. No `dias_ultima_interaccion`, no weak candidates, no calendar/seasonal or bank-practice features.
* **Training:** all 11 months (Jan–Nov), no resampling.
* **Output:** `id_cliente,prediccion` in `test.csv` row order, `prediccion` in [0, 1]. The Gini only uses the order;
  a calibrated probability scale is optional.
* **Expected score:** about 0.24–0.26 on December (November-like months score 0.23–0.25).
* **Built:** `submission/submission.csv` with the two-model blend (M1 + M3) and probabilities on M1's scale at November's level
  (`submission/make_submission.py`, details and checks in `submission/README.md`). Dry run on November: Gini 0.2517.

## Open questions (they change the interpretation, not the facts)

1. What business event is `objetivo` ("first conversion")? Is it taking a product, accepting an offer, or a credit
   approval? The risk gating fits an eligibility or approval process.
2. Is the dataset known to be synthetic? If so, recovering the generator's rules (§3) is the
   highest-value path. If it is supposed to be real, the integrity findings in §2 should be raised
   with the organizers.

## Files

- `eda.py`: all computations, reproducible, writes the viz data.
- `validate.py`: re-checks every claim from scratch and prints PASS/FAIL.
- `02_interaccion_y_objetivo.md`: the corrupted column, the target, and whether the data is synthetic.
- `groups.py` + `03_grupos_y_realidad.md`: group contrasts, seasonality, Peru benchmarks.
- `viz/index.html`, `viz/app.js`, `viz/style.css`: the d3 explorer.
- `viz/data/clients.csv`: one row per client (static profile + `t0`, `t1`, `n_months`, `converted`).
- `viz/data/eda.json`: precomputed panel, integrity, audit and segment tables.
- `context_next_session/`: hand-off notes (decisions, corrections, current model of the data).
- `phase2_ml/` … `phase7_ml/`: modeling phases, each with its own `README.md`, `out/` and `figures/` (§5).
- `submission/`: the December submission (§6).
