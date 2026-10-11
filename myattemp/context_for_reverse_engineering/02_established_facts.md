# Established facts

Each fact has an ID, the numbers, and how it was verified. **[V]** = asserted by a PASS check in
`python myattemp/validate.py` (29/29 as of 2026-10-07). **[M]** = measured in the EDA sessions but not part of
`validate.py`. Re-run the measurement before building on an [M] fact.

## A. Panel mechanics

| ID | Fact | Evidence |
|---|---|---|
| F1 | A client converts at most once; the conversion is always their last row; histories have no gaps. | [V] |
| F2 | Leaving the panel happens **only** through conversion: all 8,061 never-converters are present in November. | [V] |
| F3 | Every month's row count is a multiple of 50: 10,400 / 9,800 / 10,300 / 9,600 / 10,100 / 10,350 / 9,700 / 10,050 / 9,900 / 10,400 / 9,500. Train + test = 110,100 + 9,900 = **120,000**. | [V] (train part) |
| F4 | Clients entering after January have the same profile distribution as the January stock (mean account age 89.8 vs 89.3 months; 19% "new client" in both). Entry ≠ account opening. New-entrant counts per month (t=2…11): 921, 2,043, 827, 1,973, 1,834, 950, 1,760, 1,299, 1,893, 728. They look like **fillers that top each month up to a round target size**. | [V] tenure; [M] counts |

## B. Attributes (client profile)

| ID | Fact | Evidence |
|---|---|---|
| F5 | **All profile columns are mutually independent.** 210 pairwise tests (Kruskal-Wallis / chi-square / Spearman): 5.7% p<0.05, p-values uniform (KS p = 0.46); max \|ρ\| = 0.021. | [V] |
| F6 | **21 of 22 columns are frozen per client**: identical in every month, including age, account age and days since last transaction. Only `dias_ultima_interaccion` changes. | [V] |
| F7 | Marginal distributions look like textbook draws: `edad` uniform integers 22–71 (chi-square vs uniform p = 0.56); `antiguedad_cuenta_meses` ~U{1..179}; `antiguedad_direccion_meses` ~U{1..239}; `dias_ultima_transaccion` ~U{1..364}; `dia_preferido_pago` ~U{1..28}; `visitas_web_ultimos_90_dias` ≈ Poisson(8) (mean 8.000, var 7.988); `ingresos` ≈ Normal(≈65k, ≈21.7k) clipped at 18,000 (1.6% at the floor), skew 0.07; `ratio_deuda_ingresos` ≈ Beta(2.33, 4.77) clipped to [0.02, 0.95]; `saldo_promedio` mean 26.4k, sd 16.1k, clipped to [300, 90,000] (6.7% at 300); `distancia_sucursal_km` ≈ LogNormal(μ = 2.006, σ = 0.791) capped at 80. | [M] |
| F8 | Category shares (clients): products 1–5 = 41.9 / 29.0 / 17.6 / 7.7 / 3.9%; risk low/med/high = 52.6 / 30.6 / 16.8%; channel web/mobile/branch/call_center/partner = 33.5 / 26.6 / 15.0 / 14.7 / 10.1%; device android/ios/web/otro = 42.1 / 31.7 / 20.4 / 5.8%; occupation clerical/manual/professional/manager/self_employed = 30.2 / 27.8 / 22.9 / 10.0 / 9.2%; region west/north/south/east/central = 25.2 / 21.7 / 20.1 / 17.9 / 15.0%; card 75.2%, app 67.3%, new 19.0%, loan 44.4%, insurance 38.3%. | [M] |
| F9 | Semantic contradictions: 18.6% of 10+-year accounts flagged "new client"; 8.3% opened the account before age 18; `numero_productos` < card+loan+insurance count in 26% of clients; loan holders have the same debt ratio as non-holders. | [V] new-flag; [M] rest |

## C. Target

| ID | Fact | Evidence |
|---|---|---|
| F10 | **Gating by `banda_riesgo`.** Rates: low 18.6%, medium 14.0%, high 8.7%. Most other effects exist only inside one band (see F11, F24). | [M] |
| F11 | Measured rules (rates are monthly conversion rates): | |
| F11a | **High risk: drop around 180 days since last transaction.** ≤180d 14.1% vs >180d 4.9%. With 6-day bins the transition is **soft, from ~172 to ~190 days** (15.5, 14.5, 12.3, 12.8, 11.8, 11.6, 10.3, 8.4, 6.4, 6.7, 5.6% for windows ending at 170, 172, …, 190). Best single split = 180. Inside each side the rate is roughly flat. | [V] split; [M] shape |
| F11b | **Low risk: products × app.** Rates by products 1/2/3/4/5, app off vs on: 13.3/14.0, 13.7/21.6, 22.9/33.0, 22.3/34.7, 24.7/35.8%. App lift ≈ 0 with 1 product, +8 pts with 2, +10–12 pts with 3+. A jump at 3 products even without the app. | [V] app part; [M] table |
| F11c | **Credit card matters only in low risk**: lift +3.6 pts in low, −0.2 in medium, 0.0 in high. | [V] |
| F11d | **Recency slope in low and medium** (30-day bins, first → last): low 21.7 → 16.2%, medium 16.9 → 12.7%. It looks linear and gentle, unlike the high-risk step. | [M] |
| F11e | ~~Debt ratio threshold in low and medium only~~ **ERRATUM (2026-10-07): the debt penalty is gated by the APP, not the band.** Debt > 0.55 lowers conversion only for clients WITHOUT the app: low 16.4 → 10.6%, medium 14.1 → 9.0%, high 8.5 → 6.8%. With the app there is no effect (20.3 → 19.6%, 14.2 → 14.4%, 8.8 → 9.4%). AIC: app-gated 89,457 vs band-gated 89,506. First found by teammate Rick (`team_attempts/rick/auditoria_gini_2026.ipynb`). See `myattemp/context_next_session/`. | [M] |
| F11f | Products in medium/high: weak, roughly monotone (medium 13.6, 13.6, 14.8, 15.8, 15.1%; high 8.5, 8.2, 9.1, 9.9, 10.7%). | [M] |
| F24 | **Branch × high risk** (found last): high-risk clients acquired at a branch convert 5.6% vs 9.3% via other channels. The gap appears on both sides of the 180-day drop: 10.7 vs 14.8% (≤180d), **1.6 vs 5.5%** (>180d). No branch effect in low (18.2 vs 18.7%) or medium (13.8 vs 14.1%). 572 clients. | [V] |
| F25 | Ten variables are flat (\|Gini\| < 0.015): age, income, balance, address months, web visits, distance, payment day, new-client flag, loan, insurance. Occupation, region and device are also flat: no main effect survives multiple-testing correction, and none interacts with the drivers. | [V] flat list; [M] groups |
| F26 | Weak, **unconfirmed** candidates: account age — low risk 150–180 months 19.7% vs ~18.4% elsewhere; overall top decile 16.1%. Age 61–65: 16.1% vs ~15%. Age group and income quartile main effects p = 0.02–0.03 (not significant after correction). Worth testing properly; could be noise. | [M] |

## D. Time

| ID | Fact | Evidence |
|---|---|---|
| F17 | **No duration effect.** Once the profile and the calendar month are in the model, months-in-panel `k` adds nothing (likelihood ratio χ² = 5.8 on 10 df; obs/pred by k 0.98–1.04). Each client behaves like a coin flipped every month with a fixed probability. | [V] |
| F18 | A constant per-profile coin reproduces survival: the January stock's share still present each month matches the prediction within 2.2 pts (Nov 21.7% observed vs 21.5% predicted). | [V] |
| F27 | **There is a common month factor.** Observed ÷ profile-predicted rate by month: 0.883, 0.983, 0.944, 1.007, 1.039, 1.038, 0.997, 0.992, 0.980, 1.087, 1.077. It splits into an upward trend (+0.0166 log-odds/month) plus a wobble (χ² = 30.2 on 9 df beyond the trend). Raw monthly rates stay at 14.1–15.7% while survivors get worse, which looks like the **monthly rate being re-centred toward ~15%**. | [V] Jan/Nov ratio; [M] decomposition |
| F28 | The month factor is **identical for every group** (0 of 10 group × month tests significant: worker type, occupation, region, age, income, channel, device, risk, loan, new flag). No Peruvian pay-calendar effect: July (gratificación) −0.009 (p = 0.78), CTS months +0.047 (p = 0.05), school season +0.020 (p = 0.44). For December ranking the month factor is a constant and cancels out. | [V] |

## E. The corrupted column

| ID | Fact | Evidence |
|---|---|---|
| F12 | `dias_ultima_interaccion` = `dias_ultima_transaccion` in the uncorrupted rows. | [V] |
| F13 | **Each row is independently overwritten with probability (t − 1)/11.** The share of intact copies follows (12 − t)/11 within 0.0012 in every month (Jan 100%, Nov 9.4%). Formula for December: 0% intact. | [V] |
| F14 | Replacement values are drawn from **that month's** `dias_ultima_transaccion` values: KS p = 1.00, not uniform. This is **not** an exact permutation (the multiset test fails for t ≥ 2), so probably sampling with replacement. Unrelated to the client's own value (r = 0.003) or to the client's previous month (r = 0.006). No memory across months. | [V] |
| F15 | Overwriting ignores the target (p = 0.63). The conversion rate is the same for intact and overwritten rows given the transaction value (max gap 0.7 pts). | [V] |
| F16 | Its Gini falls from 0.076 (Jan) to 0.017 (Nov). LightGBM on November: 0.2322 without the column vs 0.2306 with it; 0.2322 vs 0.2293 when November's column is fully overwritten to imitate December. | [V] decay; [M] LightGBM |

## F. Realism (why the data is considered synthetic)

| ID | Fact | Evidence |
|---|---|---|
| F19 | Strong within-data contrasts that real Peruvian bank data would show are all flat: iOS share by income quartile 43/42/43/43% of smartphone users; income by region, age and occupation (all ≈65k); balance, products and card by income; app use by age (66 → 68%); 60% digital acquisition even for 10–15-year-old accounts; branch distance by region; high-risk share by occupation. | [V] iOS; [M] rest |
| F20 | Against external Peru benchmarks: iOS ≈ 18–19% of mobile (StatCounter, mid-2026) vs 43% here; INEI 2024 rural income about half of urban vs no regional gradient here; no gratificación or CTS seasonality. | [M] |
| F21 | Round clip bounds (18,000; 300; 90,000; 0.02; 0.95; 80) and round month sizes (F3). | [M] |
| F22 | Zero attrition (F2) and conversion rate held near 15% each month (F27). | [V] |
| F23 | **Conclusion: the rows are fully synthetic at the attribute level.** Columns are drawn independently; the target comes from a small rule set gated by risk band, times a common month factor; one column is deliberately corrupted. Whether the *rule set* was inspired by a real BCP model cannot be determined from the data. | — |
