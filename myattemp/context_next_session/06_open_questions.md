# Open questions

## For the user (ask early; the answers change the plan)

| # | Question | Why it matters | Status |
|---|---|---|---|
| U1 | **Next phase: A (ML / submission) or B (keep exploring)?** | Determines the whole session | **open: the user will decide** |
| U2 | Is the score the only judging criterion, or are methodology, explanation or a pitch also judged? | If explanation counts, the synthetic-data story and rule recovery are the strongest assets | open |
| U3 | How many submissions are allowed, and is there a public leaderboard? | Teammates noted there is no leaderboard or feedback, so local validation is the only guide | partly known (no feedback, per Elmer) |
| U4 | Should the final answer be the user's own model, or combined with teammates' (e.g. a blend with Jorge's ensemble)? | Team strategy | open |
| U5 | May `test.csv` be analysed beyond producing the submission (e.g. new vs returning clients in December)? | The user scoped analysis to train | open (default: no) |
| U6 | What does "conversion" mean for the organizers? Is the dataset officially synthetic? | Interpretation only; ranking unaffected | unanswerable from the data, could be asked of the organizers |

## For the data

| # | Question | Suggested test | Priority |
|---|---|---|---|
| Q1 | Exact rule shapes: thresholds (180 vs 182 days; debt 0.55; products 3), hard vs soft steps | profile likelihood on the count tables | high (B), medium (A) |
| Q2 | Combination scale: additive in probability, multiplicative, or logit? | compare GLM links on `cells_rule_vars.csv` | high (B) |
| Q3 | Any further gated rule (like branch × high, app × debt)? | each variable × band × app, Bonferroni, half-sample replication | medium |
| Q4 | Selected group or independent sample? | dependence tests within each risk band / rule cell (selection would create dependence) | medium (interpretation) |
| Q5 | Weak candidates: account age 150–180 in low risk; age 61–65 | focused tests with replication | low |
| Q6 | What drives the common monthly level (trend, target rate, shocks)? | compare month-level models | low (irrelevant for the December ranking) |
| Q7 | Is the remaining variation pure coin-flip noise? | per-cell calibration test of the final model | high, as the stopping rule |

## Already answered (don't reopen without new evidence)

- Is `dias_ultima_interaccion` usable? **No** (D6).
- Do months in panel or history matter? **No**, after the calendar month is controlled (D8, C1).
- Is there Peruvian seasonality or a group-specific calendar? **No** (D9).
- Do social groups (age, income, region, occupation, device) change conversion? **No** main effect after correction, no interaction with the drivers.
- Does domain feature engineering help? **No** (ac, Elmer, Rick experiments; D10).
