# Expected vs observed behavior

"Expected" = what a real bank (Banco de Crédito del Perú, per the challenge) or simple logic predicts.
"Observed" = `train.csv`. Rates are monthly conversion rates (conversions ÷ rows); overall 15.0%.

## 1. Panel and time

| Behavior | Expected | Observed | Reading |
|---|---|---|---|
| Who leaves the panel | churners, closed accounts, converters | **only converters**; all 8,061 non-converters present in Nov | survival design: follow each client until conversion |
| Attributes over months | age, account age, days-since-transaction move every month | **all 21 frozen** per client | snapshot at entry (ac: final-snapshot hypothesis contradicted by 416 negative implied tenures) |
| Monthly sizes | irregular | multiples of 50 (10,400 … 9,500); train + test = 120,000 | fixed targets; new clients are fillers (same profile distribution as the January stock) |
| Conversion over time in panel | "cooling down" or engagement effects | none once profile and month are known | constant per-profile monthly probability; survivors are just lower-probability profiles |
| Conversion rate by month | campaign or seasonal swings | 14.1–15.7%; drift relative to profiles from 0.88 (Jan) to 1.08 (Nov) | one common monthly level, rate seemingly held near 15% |
| Peru pay calendar | rises around gratificación (Jul), CTS (May/Nov), school season (Feb–Mar), mostly for formal workers | July −0.009 (p 0.78), CTS +0.047 (p ≈ 0.05), school n.s.; formal = independent | not encoded |
| Group-specific seasonality | segments react differently | 0 of 10 groups differ | single shock applied to everyone |

## 2. Variables one by one

| Variable | Expected effect / behavior | Observed | Verdict |
|---|---|---|---|
| `banda_riesgo` | low risk → more offers / approvals | low 18.6%, medium 14.0%, high 8.7%; **acts as a gate** for the other effects | **signal (master switch)** |
| `numero_productos` | more products → more cross-sell | low risk: jump at 3+ (≈13–14% → 23–25% without app, 33–36% with app); weak in medium/high | **signal (gated)** |
| `dias_ultima_transaccion` | recent activity → more conversion; should grow ~30/month while a client stays | frozen; high risk: drop from ~14% to ~5% around 180–182 days (soft, ~172–190); low/medium: gentle linear slope (21.7 → 16.2%, 16.9 → 12.7%) | **signal (fixed trait, not a recency)** |
| `activo_movil` | app users see more offers | low risk: +8–12 pts only with 2+ products; also **cancels the debt penalty** | **signal (gated)** |
| `tiene_tarjeta_credito` | card holders reachable for cross-sell | +3.6 pts in low risk only; 0 in medium/high | **signal (gated)** |
| `ratio_deuda_ingresos` | higher debt → lower conversion | flat up to ~0.55; above it, lower **only without the app** (low 16.4 → 10.6%, medium 14.1 → 9.0%) | **signal (tail, app-gated)** |
| `canal_adquisicion` | digital vs branch differ | flat overall; **high risk + branch: 5.6% vs 9.3%** (1.6% vs 5.5% for dormant clients) | **hidden rule** |
| `antiguedad_cuenta_meses` | loyalty or onboarding effect; must be ≤ age − 18 | uniform 1–179, frozen; 8.3% opened before age 18; possible weak bump in low risk at 150–180 months (19.7% vs ~18.4%, unconfirmed) | noise (one candidate to test) |
| `es_nuevo_cliente` | strongly tied to short account age | 18.6% of 10+-year accounts flagged new; no effect | **contradictory, noise** |
| `edad` | life-cycle effect; income peaks mid-age | uniform 22–71, no aging over months; no effect (61–65 bump 16.1%, unconfirmed) | noise |
| `ingresos` | more income → more capacity; varies by region/age/occupation | Normal(~65k, ~21.7k) floored at 18,000; **identical across regions, ages, occupations**; no effect | noise |
| `saldo_promedio` | grows with income | clipped [300, 90,000]; unrelated to income; no effect | noise |
| `tiene_prestamo` | loan holders have higher debt | same debt ratio; no effect | contradictory, noise |
| `tiene_seguro` | engaged clients | no effect | noise |
| `ocupacion` | managers/professionals earn more; independents riskier | same income, risk and conversion | noise |
| `region` | rural income ~half of urban (INEI); far branches in the Amazon | same income and distance everywhere; no effect | noise |
| `dispositivo_principal` | iOS ~18–19% in Peru (StatCounter), more among the rich; tied to app use | iOS 43% of smartphones, flat by income; unrelated to app use or web visits | noise, unrealistic |
| `visitas_web_ultimos_90_dias` | digital intent → conversion; more for web users | Poisson(8) for everyone; no effect | noise |
| `distancia_sucursal_km` | branch channel and region related | LogNormal capped at 80; no relation | noise |
| `dia_preferido_pago` | heaps at paydays (15, month end) | uniform 1–28 | noise (placebo) |
| `antiguedad_direccion_meses` | stability proxy | uniform; no effect | noise |
| `dias_ultima_interaccion` | ≤ transaction days; +30/month or reset | copy of transaction days overwritten with probability (t−1)/11; apparent Gini −0.058 borrowed and fading (Jan 0.076 → Nov 0.017) | **corrupted: drop** |

## 3. Relationships between attributes

| Expected in real data | Observed |
|---|---|
| dense correlations (income–occupation, debt–risk, loan–debt, age–tenure, app–age, device–income) | 210 pairwise tests: 5.7% with p < 0.05 (chance level), uniform p-values (KS p = 0.46), max \|ρ\| = 0.021 |
| digital acquisition rare for accounts opened 10–15 years ago | 60% web/mobile in every account-age band |
| right-skewed income with heaping at round numbers | symmetric (skew 0.07), no heaping except the floor |

## 4. Interpretation summary

- **Strong evidence of a synthetic population:** attribute independence, flat within-data contrasts, textbook marginals, exact corruption schedule, clean gated rules.
- **Possibly real program design:** follow-up until conversion, snapshot at entry, fixed monthly capacity, ~15% monthly response.
- What the data **cannot** tell: whether the conversion rules were modeled on a real BCP scoring process.
