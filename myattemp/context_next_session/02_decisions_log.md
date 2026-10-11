# Decisions log

Each decision has its reason and evidence. "User" = decided or requested by the user; "Analysis" = taken by the session on evidence.

## Scope and working style

| ID | Decision | By | Reason |
|---|---|---|---|
| D1 | Start with understanding (what kind of clients, do variables behave as expected) **before** prediction | User | Teammates plateaued at ~0.27 by fitting a "bank story" without questioning the variables |
| D2 | Use only `data/train.csv` for analysis | User | Explicit instruction; test is for submission |
| D3 | Visualization: d3, static files, runnable with `python -m http.server` | User | Simplest setup |
| D4 | Plain-language wording (e.g. "monthly conversion rate", not "hazard") | User | Got lost with the terminology |
| D5 | Don't modify the explorer or code while writing context documents | User | Explicit instruction |

## Modeling-relevant decisions

| ID | Decision | By | Reason / evidence |
|---|---|---|---|
| D6 | **Drop `dias_ultima_interaccion`** | Analysis | It is a copy of `dias_ultima_transaccion` overwritten with probability (t−1)/11 per row (fits within 0.12 pts every month); 0% intact in December; replacements are random same-month transaction values; LightGBM is never worse without it (Nov 0.2322 vs 0.2306). Teammates ac and Rick reached the same conclusion independently. |
| D7 | **Validate walk-forward by calendar month**; judge on several months, not one | Analysis | Random row splits leak (future rows reveal survival); random client splits overrate flexible models (Rick: 0.227 vs 0.103 for a memorizer under the two schemes). One-month Gini SE ≈ 0.009. |
| D8 | **No duration / history features** (months seen, cohort) | Analysis | Once profile and calendar month are controlled, months-in-panel adds nothing (χ² = 5.8 on 10 df). Rick and ac confirm. Remaining months (`meses_restantes`) is a leak (AUC = 1.0). |
| D9 | **No seasonal or Peru-calendar features** | Analysis | No gratificación effect (July coef −0.009, p = 0.78); CTS borderline (p ≈ 0.05); every group shares the same monthly level. Within December's single month the monthly level is a constant. |
| D10 | **No domain feature engineering** (ratios, combinations of columns, "bank logic" features) | Analysis | All columns are mutually independent, so combinations of noise columns are noise. ac tested income/age, device × income and entry-snapshot clocks: no gain (ΔAUC −0.001, CI includes 0). |
| D11 | Generator-shaped features (thresholds, band/app gates) are acceptable, **for variance reduction only** | Analysis | Jorge: logistic + threshold rules ties or beats boosting; Rick: depth-3 trees suffice. See 05. |
| D12 | Treat the client population as **synthetic** | Analysis | 210 dependence tests between attributes ≈ pure chance (5.7% p < 0.05, uniform p-values); every strong Peru reality check fails; see 03. The *program design* (follow-up until conversion, monthly capacity) may still mirror reality; see 05 §1. |

## Corrections made along the way (be aware of them)

| ID | What was claimed | Correction | How found |
|---|---|---|---|
| C1 | "Months in panel adds nothing (obs/pred ≈ 1 for every k)" | True **only after controlling for the calendar month**; without it, obs/pred reaches 1.13 at k = 9, because high k exists only in late months | `validate.py` check failed, then fixed |
| C2 | "`canal_adquisicion` is noise" (overall Gini 0.019) | Hidden rule: **high-risk clients acquired at a branch** convert 5.6% vs 9.3% (both sides of the 180-day drop) | group × driver tests |
| C3 | "Debt penalty applies in low and medium risk bands" | **The debt penalty is gated by the APP**: debt > 0.55 lowers conversion only without the app (low 16.4 → 10.6%, medium 14.1 → 9.0%); with the app no effect. AIC 89,457 (app-gated) vs 89,506 (band-gated) | Rick's notebook; re-tested in this session |
| C4 | "High-risk step at 180 days" | Best split 180–182 (Rick: 182, 13.7 → 4.6%); the transition looks **soft** over ~172–190 days with 6-day bins | finer bins |
| C5 | "Replacement values in the interaction column are not a permutation" | True for Feb–Nov in train; in **December the interaction vector is an exact permutation** of the transaction values (same values and frequencies, r = −0.003, 26 coincidences in 9,900). Found by ac on test.csv | ac's origin report |
| C6 | "Synthetic signals" list included frozen profiles, no attrition, round month sizes | These could also come from a **real program design** (snapshot at entry, follow-up until response, monthly capacity). They are weaker evidence than the attribute independence | strategy discussion (05) |
