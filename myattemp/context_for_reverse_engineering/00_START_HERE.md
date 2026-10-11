# Context: reverse-engineering the data generator

You are picking up an investigation. The previous session (EDA only, no submission) concluded that the
competition dataset is **synthetic**, produced by a fairly simple generator. **Your mission: reconstruct
that generator, above all the rule that produces the target, as precisely as the data allows**, then use
it to rank December clients.

## Read in this order

| File | What it gives you | Read time |
|---|---|---|
| `00_START_HERE.md` | mission, guardrails, definition of done | 3 min |
| `01_dataset_and_task.md` | files, columns, metric, how the panel works | 3 min |
| `02_established_facts.md` | **verified** facts (F1–F28) with numbers and how each was checked | 10 min |
| `03_generator_hypothesis.md` | best current reconstruction as pseudo-code, measured parameters, **ranked open questions** | 10 min |
| `04_prior_results.md` | what models scored, what failed, the expected ceiling | 3 min |
| `tables/` | count tables (exposure + conversions per cell) for fitting candidate rules quickly | as needed |

## Ground rules (from the user)

- **Work from `data/train.csv`.** The user scoped the EDA to train only. Before using `data/test.csv` for
  anything beyond producing the submission (e.g. checking which December clients are new), **ask the user**.
- **Do not modify** `myattemp/eda.py`, `groups.py`, `validate.py` or `myattemp/viz/`. Write new work in a new folder,
  e.g. `myattemp/reverse_engineering/`.
- The user wants understanding, not blind tuning. They are skeptical of "fit a model to bank context"
  approaches. Explain *what* the generator does and back it with evidence.
- The user found the earlier terminology confusing. Prefer plain words: "monthly conversion rate"
  (conversions ÷ rows), not "hazard".

## Verify before you trust

```bash
python myattemp/validate.py          # ~3 min, 29 checks, all PASS as of 2026-10-07
python myattemp/context_for_reverse_engineering/tables/make_tables.py   # regenerates tables/
```

If a check fails, treat the related fact as unverified. One earlier claim was wrong and was corrected:
"months in panel has no effect" holds **only after controlling for calendar month** (see F17).

## Traps already found (don't re-learn them)

1. `dias_ultima_interaccion` **looks** like the 4th-best variable but is a corrupted copy that will be
   **pure noise in December** (F12–F16). Exclude it.
2. **Validate per month, not with random K-fold.** Clients repeat across months, and a random split leaks
   clients between fit and measure.
3. Any "months-in-panel" feature leaks the future if built as remaining months: `meses_restantes`
   gives AUC = 1.0, because the panel stops at conversion. Count only *previous* months, and even
   then it adds nothing (F17).
4. Column ratios and combinations (debt per product, balance/income…) cannot create signal: the columns are
   mutually independent (F5).
5. The overall Gini of a variable can hide a gated rule. `canal_adquisicion` has an overall Gini of only
   0.019 but contains a clear rule (F24). **Always look inside each `banda_riesgo`.**
6. With ~110k rows and many cells, multiple-testing false positives are easy. Require p < 0.001 or
   replication across month halves before accepting a new rule.

## Definition of done

1. A written generator spec, `P(objetivo=1 | profile, month)`, with each component justified by a test.
   Compare its likelihood against flexible baselines (e.g. LightGBM) on held-out months. If the spec
   matches the baseline's log-loss and Gini, nothing big is missing.
2. A list of the remaining unexplained variance, with tests showing it is consistent with pure
   coin-flip noise (e.g. a calibration test per cell).
3. Optionally, a December submission ranked by the reconstructed rule (`id_cliente,prediccion`, in test row order).

## Existing analysis (for reference, not required reading)

- `myattemp/README.md`, `02_interaccion_y_objetivo.md`, `03_grupos_y_realidad.md`: the human-facing reports.
- `myattemp/viz/` d3 explorer (`cd myattemp/viz && python -m http.server`). Its tab 4 "Segment explorer"
  shows any variable split by risk band, which helps for eyeballing rule shapes.
- `myattemp/viz/data/clients.csv`: one row per client (static profile + `t0`, `t1`, `n_months`, `converted`). Booleans are 0/1.
- Teammates' attempts: `team_attempts/elmer/resultado_final.md`, `team_attempts/jorge/reports/ml.md`.
