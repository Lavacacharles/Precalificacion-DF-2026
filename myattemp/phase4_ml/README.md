# Phase 4: seasonal features (2026-10-07, a try)

Train only. No new tuning: each base set uses its phase-3 best XGBoost settings (`../phase3_ml/out/best_params.json`).
Walk-forward Jul–Nov, 3 seeds averaged per month. Paired bootstrap (same rows) against the variant's own base set.

| File | What it does |
|---|---|
| `seasonal.py` | seasonal features on top of phase 3's sets (`rule7`, `built`) |
| `run_seasonal.py` | comparison + descriptive check (~15 min). Outputs in `out/`, figures in `figures/` |

## Features

* **Calendar** (same for every client in a month): month, sin/cos of month, quarter, gratificación month (Jul, Dec), CTS month (May, Nov),
  Christmas, Fiestas Patrias, school season (Feb–Mar), tax season (Mar–Apr), months to next gratificación,
  holidays / weekend days / business days (Peru 2026 calendar), last month's observed conversion rate.
* **Client × season**: salaried worker (receives gratificación and CTS), gratificación × salaried, CTS × salaried,
  expected cash windfall (one monthly salary for the gratificación, half for CTS), windfall ÷ balance, distance of the preferred
  payment day to the quincena (15th / 30th), pays in the first half, entry month, current tenure (frozen tenure + months in panel),
  account anniversary month, branch × business days.

## Results (Gini)

| Variant | Jul | Aug | Sep | Oct | Nov | Mean | vs base (95% CI) |
|---|---|---|---|---|---|---|---|
| rule7 | 0.2633 | 0.2604 | 0.2553 | 0.2655 | 0.2355 | **0.2560** | — |
| rule7 + calendar | 0.2653 | 0.2582 | 0.2505 | 0.2646 | 0.2363 | 0.2550 | −0.0010 (−0.0025, +0.0006) |
| rule7 + client-season | 0.2481 | 0.2641 | 0.2504 | 0.2670 | 0.2335 | 0.2526 | −0.0032 (−0.0079, +0.0019) |
| rule7 + both | 0.2473 | 0.2626 | 0.2500 | 0.2680 | 0.2343 | 0.2525 | −0.0033 (−0.0083, +0.0015) |
| built | 0.2728 | 0.2584 | 0.2504 | 0.2637 | 0.2316 | 0.2554 | — |
| built + both | 0.2673 | 0.2604 | 0.2499 | 0.2619 | 0.2347 | 0.2548 | −0.0005 (−0.0024, +0.0016) |

* **No seasonal variant beats its base.** Every difference is negative or zero; no CI excludes zero.
* **Calendar features can't reorder clients by themselves.** They are constant inside a month, and the Gini is computed inside a month.
  They only act through interactions with client features, and the trees found none worth keeping (≤ 0.8% of the gain each).
* **The July drop (−0.015 with client-season features) shows the main trap: one year = one example of each season.**
  The July model is trained on Jan–Jun: it has never seen a gratificación month. It has seen one CTS month (May),
  so the cash-windfall feature learned whatever May's noise was and applied it to July's larger windfall.
  Checked: removing only the two windfall features brings July from 0.2481 back to 0.2574; the remaining −0.006 against rule7 (0.2633) is
  the usual cost of adding noise columns. The same thing would
  happen in December, which is both a gratificación month and Christmas, and absent from training except via July.
* **Descriptive check** (`figures/monthly_rates_by_group.png`): salaried clients stay within ±2% of each month's overall rate.
  Self-employed clients (~950 rows a month) swing ±15%, but with a sampling error of about ±8% per month, and
  in the wrong direction for a gratificación story (they rise in July, when only salaried clients get the bonus).
  Income thirds wander ±8% with no CTS / gratificación pattern. This matches the earlier finding:
  one common monthly level for everyone, no group-specific seasonality.

## Conclusion

Seasonal features add nothing and can hurt when a season appears only once in training. For December, keep the 7 rule inputs and no calendar features.
