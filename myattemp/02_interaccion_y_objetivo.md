# `dias_ultima_interaccion`, what `objetivo` is, and whether the data is synthetic

Every number below is recomputed by `python myattemp/validate.py` (24/24 checks pass).

## 0. Plain-language glossary

| Term used in the explorer | Meaning |
|---|---|
| **Row / client-month** | One line of `train.csv`: one client in one calendar month. |
| **Monthly conversion rate** (earlier called *hazard*) | Of the clients present in a month (or a group), the share that converts that month. The overall value is 15.0%. This is what a row-level model predicts. |
| **Calendar month `t`** | 1 = January 2026 … 11 = November. December, the test month, is `t = 12`. |
| **Months already in the panel `k`** | How many earlier rows the client has. `k = 0` is the client's first appearance. |
| **January stock** | The 10,400 clients present in January. Everyone else **enters later**. |
| **Survivor** | A client still present because they have not converted yet. |
| **Frozen** | The value is identical in every month for that client. |
| **Gini** | The competition metric, 2·AUC − 1. 0 = random ordering, 1 = perfect. Computed per variable, it says how well that variable alone orders rows. |
| **95% CI (whiskers)** | The range where the true rate likely lies given the sample size. Overlapping whiskers mean the difference may be noise. |

---

## 1. The `dias_ultima_interaccion` problem, step by step

### 1.1 What the column should be

"Days since the last interaction registered with the client." For a real bank:
- A transaction is a kind of interaction, so it should be **≤ `dias_ultima_transaccion`**.
- From one month to the next it should either **reset** (the client interacted) or **grow by about 30**.
- It should therefore be strongly related to its own value in the previous month.

### 1.2 What it actually is

Three real clients (all 11 months shown, values copied from `train.csv`):

| month | client 2 transaction | client 2 interaction | client 3 transaction | client 3 interaction |
|---|---|---|---|---|
| Jan | 304 | **304** | 220 | **220** |
| Feb | 304 | **304** | 220 | 7 |
| Mar | 304 | 236 | 220 | **220** |
| Apr | 304 | 128 | 220 | **220** |
| May | 304 | **304** | 220 | 285 |
| Jun | 304 | **304** | 220 | **220** |
| Jul | 304 | 137 | 220 | 27 |
| Aug | 304 | 70 | 220 | 257 |
| Sep | 304 | 302 | 220 | 237 |
| Oct | 304 | 279 | 220 | 30 |
| Nov | 304 | **304** | 220 | 131 |

Bold = identical to the transaction value. The pattern:

1. **The real value is just a copy of `dias_ultima_transaccion`.** It is the transaction
   value, which itself never changes (frozen).
2. **Each row, independently, is overwritten with probability `(t − 1) / 11`.** That is 0% in
   January, 9% in February, … 91% in November. The observed share of intact copies follows
   `(12 − t)/11` within **0.0012** in every month. A rule this exact is a deliberate schedule, not a
   measurement artifact.
3. **The replacement is a random draw from that month's transaction values**, the values of
   *other clients* (KS test vs that month's transaction distribution: p = 1.00; vs uniform: rejected).
   The replacement has no relation to the client's own value (r = 0.003) or to the client's
   previous month (r = 0.006).
4. **It has no memory.** A row that was overwritten last month is no more or less likely to be
   overwritten this month. Client 2 goes back to the true 304 in May and November.
5. **It ignores the target.** Converting rows are overwritten at the same rate (p = 0.63).
   Given the transaction value, the conversion rate is the same whether the copy is intact
   or replaced (max gap 0.7 pts).
6. **It breaks the semantics.** Only 50% of replaced values are ≤ the transaction value (a coin
   flip). Month-to-month changes are ±150 days at random, never "+30 or reset".

### 1.3 Why it looks predictive, and why that is an illusion

All of its apparent signal is **borrowed** from `dias_ultima_transaccion`, in the share of rows where it
is still an intact copy:

| month | intact copies | Gini of interaction | Gini of transaction |
|---|---|---|---|
| Jan | 100% | 0.076 | 0.076 |
| Nov | 9% | 0.017 | ~0.10 |
| **Dec (test)** | **0% by the formula** | **≈ 0** | unchanged |

On the whole train set it shows a Gini of −0.058, the 4th-best variable. That makes it look like
a real recency signal that a model should use. In December the formula gives 0% intact, so it
will be pure noise.

### 1.4 How much does it hurt a model? (measured)

LightGBM trained on Jan–Oct, scored on November. "December-like" means the November
column was fully replaced by random same-month transaction values, as the formula predicts:

| Features | Nov as-is | Nov, December-like | Gain share of the column |
|---|---|---|---|
| all 22 | 0.2306 | 0.2293 | 4.7% |
| all except `dias_ultima_interaccion` | **0.2322** | **0.2322** | — |
| 5 signal vars + interaction | 0.2273 | 0.2239 | 12.5% |
| 5 signal vars + debt, no interaction | 0.2313 | 0.2313 | — |

Removing it is never worse and is immune to the December change. The harm is **small (0.1–0.3 Gini
points)** because the model also has the real transaction variable. The bigger danger is in
**validation**: folds built on early months reward the column, so feature selection and tuning get
misled. **Decision: drop it.** Do not try to "denoise" it either. The intact rows carry
nothing beyond `dias_ultima_transaccion`.

---

## 2. What is `objetivo`?

### 2.1 What the definition and the data agree on

- `objetivo = 1` in month m means the client's **first** conversion happened in month m.
- After converting, the client **leaves the panel for good** (verified: every converter's
  conversion is its last row). "First" is therefore also "only": nobody converts twice.
- A client leaves **only** by converting. All 8,061 non-converters are still present in November.
  No churn, no closed accounts, no gaps.

So the task is a **time-to-event (survival) problem**: every month each remaining client
either converts and exits, or stays. The submission asks for the probability that a client
present in December converts in December.

### 2.2 What the data says the event is not

| Hypothesis for "conversion" | What we would expect | What we see | Verdict |
|---|---|---|---|
| Accepting a cross-sell offer or a new product | ~1–5% per month; seasonality around campaigns; product holders behave differently | 15% **every** month (14.1–15.7%); 78% of the January stock converts within 11 months; loan or insurance holders no different | unlikely |
| Credit approval | income and debt ratio dominate | income is flat; debt only matters above 0.55; the risk band is not built from debt | partially (risk gate fits) |
| Digital activation | web visits, device and app drive it | visits and device are flat; the app matters only for low-risk clients with 2+ products | unlikely |
| A newly opened account converting | late entrants have short tenure and are flagged new | entrants have the same tenure as the stock (89.8 vs 89.3 months) and 19% "new", same as the stock | no |

No real banking event fits all the facts. The event has **no recoverable business meaning beyond
the rule that generates it** (§3).

### 2.3 What the event is, statistically (verified)

The data behaves exactly like this process:

```
each month m, for every client still present:
    p = h(profile) × month adjustment
    flip a coin with probability p  -> 1: convert and leave; 0: stay
```

Evidence:
- **A constant coin per profile reproduces survival.** Taking January's 10,400 clients and a
  20-parameter rule model, the predicted share still present each month matches the observed
  share within 2.2 points through November (21.7% observed vs 21.5% predicted).
- **Time in the panel adds nothing** once profile and calendar month are known (likelihood-ratio
  χ² = 5.8 on 10 df, p ≈ 0.83). An earlier note said obs/pred ≈ 1 for every k without controlling
  for the calendar. That check failed when re-run (up to 1.13) because high k only happens in
  late months. With month intercepts it passes (0.98–1.04).
- **The monthly rate is held near 15% on purpose.** Survivors get worse every month, since the good
  profiles have already left, so the predicted rate should fall. The observed rate does not fall:
  observed/predicted goes from 0.88 in January to 1.08 in November. Each month's rate appears to be
  re-centered to ~15%. **This does not affect the Gini**, which is computed inside the single month
  of December. It would only matter for calibrated probabilities.

## 3. Is the dataset synthetic? The evidence

| # | Evidence | Strength |
|---|---|---|
| 1 | All 21 profile columns are **frozen**. Age, tenure and days-since-transaction never advance over 11 months. | strong |
| 2 | Features are **mutually independent** (max \|ρ\| = 0.021). Loans don't raise debt, occupation doesn't set income, risk isn't built from debt. | very strong |
| 3 | **Textbook distributions**: uniform age, tenure, address, payment day and transaction days; Poisson(8) visits; normal income; log-normal distance. | strong |
| 4 | **Round clipping bounds**: income ≥ 18,000; balance 300–90,000; debt 0.02–0.95; distance ≤ 80. | strong |
| 5 | **Every month has a round size** (multiple of 50: 10,400, 9,800, …). Train + test = 110,100 + 9,900 = **120,000** exactly. Late "entrants" have the same profile as the stock, so they behave like **fillers that top each month up to its target size**. | very strong |
| 6 | `dias_ultima_interaccion` overwritten on an **exact linear schedule** `(t−1)/11`. | very strong (deliberate noise / drift injection) |
| 7 | The outcome follows **clean rules gated by one label** (a 180-day step for high risk, 3+ products for low, card only for low). Ten variables have exactly no effect. | strong |
| 8 | Monthly conversion rate held at ~15% regardless of who is left. | moderate |

**Conclusion:** it is almost certainly synthetic. It looks like a simulator built for a competition,
with decoy variables, an injected-drift column and a hidden rule set. The useful question is therefore
not "what would a bank customer do?" but **"what rules did the generator use?"**

## 4. Consequences, and where to look next

1. The ceiling is set by the coin flip. Even the true `h(profile)` cannot order rows much better,
   because most rows sit in the 13–19% band. A Gini of about 0.26–0.27 on validation months (0.23–0.24 on
   November) is close to that ceiling. Big jumps from tuning are not available.
2. Gains, if any, come from **recovering the rules precisely**: exact thresholds (is it 180 or 182
   days? 0.55 or 0.6 debt?), whether recency is linear or stepped for low and medium risk, and whether
   other variables have effects hidden *inside* a band × product cell.
3. Things to drop on evidence, not taste: `dias_ultima_interaccion`, months-in-panel features,
   and the ten flat variables.
4. One open risk: if the December rate is re-centered like the other months, probabilities would be
   mis-calibrated, but ordering (Gini) is unaffected.
