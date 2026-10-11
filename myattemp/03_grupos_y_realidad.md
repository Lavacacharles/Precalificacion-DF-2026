# Groups, seasonality and Peru: is the data fully synthetic?

**Claim tested:** if this panel came from Banco de Crédito del Perú clients, separated groups (age,
income, region, occupation, formal vs independent workers, channel, device) would (a) differ from each
other in their attributes, (b) react to the Peruvian pay calendar, and (c) react to it *differently*
from each other. If every group behaves the same and none reacts to the calendar, that is strong
evidence the data is generated.

Reproduce: `python myattemp/groups.py` (writes the data for explorer tab **7 · Groups vs reality**).
The key claims are also re-checked in `python myattemp/validate.py`.

---

## 1. Fingerprint: client attributes have no relationships at all

I ran **every** pairwise dependence test between the 21 client attributes, 210 tests in total:
- Kruskal-Wallis for a group vs a number (e.g. income by region),
- chi-square for a group vs a group (e.g. risk band by occupation),
- Spearman for a number vs a number (e.g. balance vs income).

| | Real bank data | This data |
|---|---|---|
| Share of tests with p < 0.05 | most of them (income, age, occupation, debt, products, digital use are all linked) | **5.7%**, the false-positive rate expected from pure chance |
| Distribution of p-values | piled up near 0 | **uniform** (KS test vs uniform: p = 0.46) |
| Strongest relationship | \|ρ\| 0.3–0.7 is common | \|ρ\| = 0.021 |

A uniform p-value distribution means **the columns were drawn independently of each other**. That is the
exact statistical signature of an independent column-by-column generator. No real population of bank
clients has zero relationship between age, income, occupation, region, balance and products.

## 2. Peru reality checks

Benchmarks: StatCounter (iOS ≈ 18–19% of mobile in Peru, Jun–Jul 2026); INEI 2024 (average monthly labor
income S/1,766 nationally, S/2,035 in Lima, S/915 rural); Ley 27735 (gratificación in July and December);
CTS deposits in May and November. The currency of `ingresos` is unknown, so only **contrasts inside the
data** are judged, not levels.

**Strong:** no plausible real-world mechanism produces these.

| Check | Expected | Observed |
|---|---|---|
| iOS vs income | iOS concentrated in high incomes | 32% / 31% / 32% / 32% by income quartile |
| Income by region | rural about half of urban; Lima highest | 64.7k–66.1k in all 5 regions |
| Income by age | rises to a 40–55 peak, falls after retirement | 65.1k / 65.8k / 64.8k / 64.8k |
| Income by occupation | manager/professional ≫ manual | 64.8k–65.6k for all five |
| Balance vs income | grows with income | 25.4k / 25.6k / 26.2k / 26.1k |
| Products and credit card vs income | richer clients hold more | 2.04 / 2.02 / 2.03 / 2.02 products; card 74–76% |
| App use by age | young far more app-active | 66% / 68% / 68% / 68% (slightly *rising* with age) |
| Digital acquisition by account age | ~0% web/mobile for 10–15-year-old accounts (opened 2011–2016, before mass mobile banking) | **60% in every tenure band** |
| Distance to branch by region | much larger in Amazon or rural regions | 7.3–7.7 km median everywhere |
| Risk band vs occupation | independent workers riskier | 16–17% high risk in every occupation |
| Attrition | some clients close accounts or go inactive | 0 exits other than conversion in 11 months |
| Group-specific seasonality | segments react differently to calendar events | 0 of 10 groups (§3) |

**Moderate:** the observation is anomalous but has a possible innocent explanation.

| Check | Observed | Innocent explanation |
|---|---|---|
| iOS share level | 43% of smartphone clients vs ~19% nationally | a wealthy clientele, but then iOS should rise with income, and it doesn't |
| Income shape | symmetric bell (skew 0.07) | a bank truncates the bottom, but the top should still skew right |
| Income heaping | no round amounts except the 18,000 floor | income could be modelled rather than declared |
| Preferred payment day | uniform over 1–28, day 15 = 3.5% | depends on which dates the bank offers |
| Age structure | flat per year of age from 22 to 71 | product targeting |
| Pay-calendar effect | none (§3) | depends on what "conversion" is |

Each moderate item alone is weak. The strong items need no assumption about BCP's clientele: they
compare groups *within* the data, and every one of them is flat.

## 3. Seasonality

### 3.1 There is a monthly effect, but it is not Peruvian

After controlling for client profiles, the calendar month matters (χ² = 67.4 on 10 df, p ≈ 10⁻¹⁰).
Its shape:
- **A smooth upward drift** of +1.7% in odds per month (χ² = 37.2 of the 67.4). This is the
  "re-centering" from note 02. Survivors get worse each month, yet the raw rate stays near 15%,
  so the expected-vs-observed ratio climbs from 0.88 in January to 1.08 in November.
- **A residual month-to-month wobble** (χ² = 30.2 on 9 df, p = 0.0004), with peaks in October/November and a dip in January.

Tested against the Peruvian pay calendar on top of the drift:

| Event | Months | Effect (log-odds) | p |
|---|---|---|---|
| Gratificación + Fiestas Patrias | July | −0.009 | 0.78 |
| CTS deposit | May, Nov | +0.047 | 0.05 (borderline) |
| School season | Feb–Mar | +0.020 | 0.44 |

July, the largest cash injection of the year for formal workers, shows **nothing**. The weak CTS hit
is driven by November, which is also the end of the upward drift.

### 3.2 Every group moves in lockstep

The decisive test asks whether any group's monthly pattern differs from the common monthly level.

| Group | χ² | df | p |
|---|---|---|---|
| formal (dependiente) vs independent | 17.0 | 10 | 0.074 |
| occupation | 39.3 | 40 | 0.50 |
| region | 35.4 | 40 | 0.68 |
| age group | 20.2 | 30 | 0.91 |
| income quartile | 36.7 | 30 | 0.19 |
| acquisition channel | 33.2 | 40 | 0.77 |
| device | 34.8 | 30 | 0.25 |
| risk band | 17.9 | 20 | 0.59 |
| has a loan | 13.0 | 10 | 0.22 |
| new-client flag | 9.4 | 10 | 0.49 |

**0 of 10 significant.** The closest is formal vs independent (p = 0.074), and it points the wrong way:
in the pay months (May/Jul/Nov), formal workers convert exactly as expected (ratio 1.001 vs 0.998
in other months). Independent workers, who receive no gratificación or CTS, show the larger swings, and
those swings are noise from a group 9× smaller. In real data, formal workers' behavior would move
around gratificación and CTS months.

A single monthly shock shared by every segment is what a generator produces when it sets one
month-level rate and applies it to everybody.

## 4. Do conversion drivers differ between social groups?

For each social group: does it shift conversion at all ("main"), and do the drivers (risk band, products,
recency, app, card) act differently inside it ("× drivers")?

| Group | Main effect p | × drivers p |
|---|---|---|
| age group | 0.020 | 0.67 |
| income quartile | 0.030 | 0.18 |
| region | 0.14 | 0.041 |
| occupation | 0.068 | 0.54 |
| **acquisition channel** | **< 0.001** | **< 0.001** |
| device | 0.78 | 0.18 |

With 12 tests, only p < 0.004 survives multiple-testing correction. Age, income and region do not.
Age, income, region and occupation **do not change who converts or why**. In real banking these are
the first variables that matter.

### The exception: a hidden rule, not a social effect

The channel effect comes entirely from **one cell: high-risk clients acquired at a branch** (572 clients).

| High-risk clients | Last transaction ≤ 180 days | > 180 days |
|---|---|---|
| acquired at a branch | 10.7% | **1.6%** |
| any other channel | 14.8% | 5.5% |

That is about −4 points on both sides of the 180-day step (p = 2·10⁻⁵ and 10⁻¹³). Low and medium risk show no
branch effect (18.2% vs 18.7%, 13.8% vs 14.1%). It has the same shape as the other rules: **one band, one
condition, one fixed shift**. That is consistent with a generator rule, and I had wrongly labelled the channel as
"noise" because its overall Gini is only 0.019. Add it to the rule set:
`high × branch` → lower.

## 5. Verdict

| Question | Answer |
|---|---|
| Do attributes relate to each other like a real population? | No. 210 tests are indistinguishable from pure chance. |
| Do groups differ (income by region/age/occupation, iOS by income, app by age)? | No. Every strong reality check fails. |
| Is there Peruvian seasonality (gratificación, CTS, school season)? | No. There is a monthly drift and wobble, but no July effect and at most a borderline CTS one. |
| Do groups have their own calendars? | No. All 10 move in lockstep. |
| Do social groups change who converts? | No. Only exact rules on risk band, products, recency, app, card, debt, and now branch × high risk. |

**Conclusion: the dataset is fully synthetic at the attribute level.** Each column was drawn
independently from a textbook distribution. Conversion was generated by a small rule set gated by
`banda_riesgo`, multiplied by a common month factor, with round monthly sizes and one deliberately
corrupted column. Nothing in the client attributes carries a Peruvian, social or behavioral pattern. The
BCP label may describe the *intended context* of the challenge, but the rows are not real client records,
or at least not with their real attributes.

What this does not prove: the *rule set itself* (low risk + products + app, the 180-day step) might
have been designed from a real BCP model. The data cannot tell us whether the rules are realistic, only that
the population they are applied to is not.

### For modeling

1. Add `high × branch` to the band rules. It is the first new signal found outside the earlier set.
2. Nothing seasonal to model: December's Gini is computed within one month, so the common month factor
   cancels out.
3. Social variables (age, income, region, occupation, device) can be dropped with evidence: no main effect
   survives correction, and no interaction with the drivers.

Sources: [StatCounter — mobile OS share, Peru](https://gs.statcounter.com/os-market-share/mobile/peru) ·
[BBVA Perú — gratificación (Jul/Dec)](https://www.bbva.pe/blog/mis-finanzas/ahorro/gratificacion-que-es-como-se-calcula.html) ·
[Trámites Perú — gratificación](https://tramitesperu.com/mtpe/gratificacion/) ·
[Perú Retail — INEI ingreso Lima 2024](https://www.peru-retail.com/ingreso-mensual-en-lima-aumento-un-6-6-en-2024-cual-es-el-salario-promedio/) ·
[Bloomberg Línea — ingreso promedio y población ocupada](https://www.bloomberglinea.com/latinoamerica/peru/poblacion-ocupada-en-peru-llega-a-173-millones-este-es-su-ingreso-promedio-mensual/)
