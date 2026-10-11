# Phase 7: re-framing as "attempts until conversion" (2026-10-07) — step 0, validation only

Question: since a client's features never change between months, should we predict *how many monthly attempts a client needs*
instead of a 0/1 per month? First check the data. `validate_reframe.py` (train only, ~1 min) → `out/checks.txt`, `out/clients.csv`, `figures/attempts.png`.

## What the data shows

**A. Panel bookkeeping (all clean):** 24,628 clients, 110,100 rows. No client converts twice, no row after a 1,
no missing months between a client's first and last row, and **nobody leaves without converting**.
Each client is offered every month from entry until they convert or the data ends.

**B. Not everyone converts.** 16,567 clients (67.3%) converted by November; **8,061 (32.7%) never did and were still waiting in November.**
Their count of attempts is unknown (censored): we only know it is larger than what we saw. The share of converters depends on how long a client was watched:
81% of January entrants converted within 11 attempts, 20% of November entrants in their single attempt.
Among converters: mean 3.7 attempts, median 3.

**C. The chance does not change with the attempt number.** The raw monthly rate falls from 16.0% (attempt 1) to ~13% (attempts 8–11), but only because
the easy clients leave early and the hard ones stay. Once the profile and calendar month are known, observed/expected is 0.98–1.04 at every attempt;
adding the attempt number to the model: likelihood-ratio 0.1 on 1 df (slope +0.0008 per attempt, z = 0.24); 4.3 on 10 df as categories.
No fatigue, no warming up, no hidden "stubbornness" left over beyond the profile.

**D. A fixed monthly chance per client reproduces everything.** Simulating each client with its own constant p (from the profile + month) and the same
entry month and November cut-off gives 67.3% converters (observed 67.3%) and the same distribution of attempts within 0.002 at every k.

## Interpretation

* The 0/1 rows are not a modeling mistake: each row is one independent attempt with the same chance p(profile) × month level.
  Stacking a client's 0,0,0,1 rows is exactly the right way to estimate that chance (a discrete-time survival model in "person-month" form).
* "Attempts until conversion" is then a geometric count with mean 1/p. Predicting it is predicting p in another unit:
  ordering clients by expected attempts is the same as ordering them by p, reversed. Same information, same Gini.
* The overlap of 0s and 1s does not come from the short time window; it comes from p being small (5–45%). Even a client with p = 0.4 says no 60% of the time.
* What the re-framing could add, and what the data rules out: a client's past refusals would carry information only if clients with the same profile
  differed in a hidden way (check C), or if the chance changed with each attempt (check C). Neither happens.

## Step 1: survival models, scored the December way (`survival_models.py`, ~5 min → `out/survival_summary.csv`)

For each test month m: one row per client from the months before m (attempts seen, converted or still waiting = censored), fit, then score every row of m
(waiting clients on their k-th attempt + new clients on attempt 1). **November (trained on Jan–Oct) is the December stand-in.**
Score = chance of converting on this attempt given no earlier conversion. "vs M1" = paired bootstrap gap to the monthly rule logistic.

| Model | Nov | vs M1, Nov (95% CI) | Mean Jul–Nov |
|---|---|---|---|
| monthly 0/1: M1 rule logistic (reference) | **0.2500** | — | 0.2550 |
| Weibull AFT, ranked by predicted median attempts | 0.2499 | −0.000 (−0.002, +0.001) | 0.2552 |
| Cox proportional hazards (penalizer 0.01) | 0.2494 | −0.001 (−0.003, +0.001) | 0.2548 |
| Weibull AFT, interval-censored, hazard at attempt k (one-off check, Nov only) | 0.2474 | — | — |
| Cox (penalizer 0.1) | 0.2415 | −0.009 (−0.018, +0.001) | 0.2513 |
| monthly 0/1: XGBoost rule7 | 0.2368 | −0.013 | 0.2566 |
| XGBoost survival:aft (best of 6: normal/logistic/extreme × σ 0.8/1.5) | 0.2352 | −0.015 | 0.2550 |
| XGBoost survival:cox (best of 4: depth 2/3 × 300/800 trees) | 0.2350 | −0.015 | 0.2541 |
| Weibull / LogNormal / LogLogistic AFT, exact-time, hazard at attempt k | 0.18–0.20 | −0.045 to −0.070 | 0.19 |
| **hard 0/1: "1 if predicted median attempts ≤ current attempt, else 0"** | **0.0487** | −0.201 | 0.049 |

* **Survival models tie the monthly 0/1 model; none beats it.** Weibull-by-median and Cox land within 0.001 of M1 in November and on average.
  The tree versions (XGBoost survival) tie the tree version of the monthly model (XGBoost rule7). Same information, same ceiling (step 0).
* **The hard 0/1 mapping collapses (Gini 0.05).** It turns an ordered score into two groups and ties thousands of clients.
  A survival model must be used as a score: the chance of accepting now, or equivalently "fewer predicted attempts = more likely".
* **Pitfall found: continuous-time models on monthly data.** Treated as exact times, the AFT fits an increasing chance per attempt
  (Weibull shape 1.38: 8% at attempt 1 vs 20% observed), so it ranks new clients far too low (Gini 0.19).
  Declaring the event as "sometime within month k" (interval censoring) fixes it: shape 1.03, i.e. flat, as step 0 found, and Gini 0.247.
  The monthly 0/1 model never had this problem: it is the discrete-time survival model.

## Conclusion

The re-framing is a correct reading of the problem (a survival process with censoring), but in this data it is mathematically the same model
we already fit: one fixed monthly chance per profile. It cannot raise the Gini. For December keep the monthly 0/1 models (or their blend).
