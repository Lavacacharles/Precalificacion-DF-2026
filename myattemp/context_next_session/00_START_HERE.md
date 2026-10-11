# Session hand-off: Datafest 2026 conversion challenge

**Date written:** 2026-10-07. **Written by:** the previous Claude session (EDA, visualization, data-origin analysis).
**Status:** the EDA phase is complete. **The user has not yet decided** whether the next phase is
**(A) the ML/submission part** or **(B) further exploration** (reverse engineering the generator).
Your first job is to help them decide, or to execute the option they pick (see `05_strategy_discussion.md` §4).

## Read in this order

| # | File | Contents |
|---|---|---|
| 0 | `00_START_HERE.md` | this file: who the user is, ground rules, state of play |
| 1 | `01_project_structure.md` | annotated map of every folder and file, what to read and what to skip, teammates' scoreboard |
| 2 | `02_decisions_log.md` | decisions already taken, with reasons, and the corrections made along the way |
| 3 | `03_expected_vs_observed.md` | for every variable and behavior: what reality predicts, what the data shows, the interpretation |
| 4 | `04_current_model_of_the_data.md` | compact current description of how the data was generated (corrected rule set) |
| 5 | `05_strategy_discussion.md` | selected vs sampled population, feature engineering pros/cons, strategy options for A and B |
| 6 | `06_open_questions.md` | questions for the user and for the data, with status |

The deeper, numbered fact base (F1–F28, with verification status) is in
`myattemp/context_for_reverse_engineering/02_established_facts.md`. Its F11e has an erratum, superseded here.

## About the user (how to work with them)

- Competing in a hackathon (Datafest 2026, UTEC); the teammates are Elmer, Jorge, "ac" and Rick. AI agents are allowed by the rules.
- **Wants critical understanding over blind modeling.** They were unhappy that teammates' agents "fit the model to
  the bank context rather than analysing the variables critically". They care whether variables behave as
  expected and, if not, what they actually describe.
- Got lost with jargon earlier. **Use plain words**: "monthly conversion rate" (conversions ÷ rows), not "hazard".
  Define any statistical term on first use.
- Likes to be asked about undefined objectives, but expects you to proceed on sensible defaults otherwise.
- Prefers visual exploration: there is a d3 explorer (`myattemp/viz/`, run `cd myattemp/viz && python -m http.server`).
- Writes in English and reads Spanish; the challenge documents and teammates' reports are in Spanish.

## Ground rules

1. **Do not modify existing code or the explorer** (`myattemp/eda.py`, `groups.py`, `validate.py`, `myattemp/viz/`)
   unless the user asks. Put new work in a new folder, e.g. `myattemp/phase2_ml/` or `myattemp/phase2_explore/`.
2. **Data scope:** the user asked to work from `data/train.csv`. `test.csv` is needed only to produce a
   submission. Ask before analysing it otherwise. (Teammate "ac" did analyse it; their findings are summarized in 01 and 03.)
3. **Validate per calendar month** (walk-forward), never with a random row or client split (see 02, D7).
4. **Never use `dias_ultima_interaccion`** in a model (see 02, D3).
5. Treat any Gini difference below ~0.01 as noise unless a paired test (same rows, bootstrap) says otherwise.
   The standard error of a one-month Gini is ≈0.009 (Rick).
6. Re-verify before building on a fact: `python myattemp/validate.py` (29 checks, all PASS, ~3 min).

## State of play in one paragraph

The panel is a survival design: each client appears monthly until they convert, then leaves; nobody leaves
otherwise. Client attributes are frozen snapshots, mutually independent, and drawn from textbook distributions.
None of the social, regional or seasonal structure of real Peruvian bank clients appears, so the client
population is synthetic. Conversion follows a small rule set gated by `banda_riesgo`, with risk band, products,
transaction recency, app, card, debt (only without the app) and branch × high risk, plus a common monthly
level. One column (`dias_ultima_interaccion`) is deliberately overwritten on a (month−1)/11 schedule and is
pure noise in December. Every team attempt (boosting, logistic regression with rules, CatBoost, neural nets, survival models,
pairwise ranking, feature engineering) lands at **Gini ≈ 0.23–0.27**. The estimated ceiling, if the true
probabilities were known, is ≈ **0.26** (Rick: 0.262). Remaining gains are probably below the
measurement noise.
