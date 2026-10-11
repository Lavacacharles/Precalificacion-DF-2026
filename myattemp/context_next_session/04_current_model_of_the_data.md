# Current model of how the data was generated (corrected 2026-10-07)

This is the best current description, combining this session with Rick's and ac's findings. Unknown values are marked `??`.

## Population and panel

```
for each month t = 1..11 (train), 12 (test):
    survivors = clients present last month who did not convert
    add new clients until the month reaches its target size (multiple of 50)
    each new client: every attribute drawn independently, frozen for life
        edad ~ U{22..71}; antig_cuenta ~ U{1..179}; antig_dir ~ U{1..239}; dias_trans ~ U{1..364}; dia_pago ~ U{1..28}
        ingresos ~ max(18000, Normal(~65k, ~21.7k)); ratio_deuda ~ clip(Beta(2.33, 4.77), 0.02, 0.95)
        saldo ~ clip(?? mean 26.4k sd 16.1k, 300, 90000); visitas ~ Poisson(8); distancia ~ min(80, LogNormal(2.006, 0.791))
        productos 1..5 = 41.9/29.0/17.6/7.7/3.9%; risk low/med/high = 52.6/30.6/16.8%; app 67.3%; card 75.2%; branch 15.0%; …
    for each client present: objetivo ~ Bernoulli( rule(profile) × month_level(t) )      # month level: same for everyone
        dias_ultima_interaccion = dias_trans with prob 1 − (t−1)/11, else a random same-month transaction value
    converters leave
```

## Conversion rule (rates in %, measured; additive vs multiplicative structure still unknown)

| Band | Main structure | Modifiers |
|---|---|---|
| **High risk** (8.7% overall) | ≤ ~180–182 days since last transaction: ~14%; beyond: ~5% (soft transition ~172–190) | branch-acquired: about −4 pts on both sides (10.7% / 1.6%); products: slight positive slope; debt > 0.55 without app: 8.5 → 6.8% (weak); card and app: no effect |
| **Low risk** (18.6%) | products: 1–2 ≈ 13–14% without app; 3+ ≈ 22–25% without app | app: +0 with 1 product, +8 with 2, +10–12 with 3+ (up to ~36%); card: +3.6; recency: linear decline ~21.7 → 16.2% across 1–364 days; **debt > 0.55 without app: 16.4 → 10.6%** (none with app); Rick: low + 3+ products + app + card + ≤90 days ≈ 41% |
| **Medium risk** (14.0%) | nearly flat 13–16% | recency: mild decline 16.9 → 12.7%; **debt > 0.55 without app: 14.1 → 9.0%**; products: slight slope; card: no effect |

Not used by the rule (flat within every band): age, income, balance, occupation, region, device, web visits,
distance, payment day, address months, loan, insurance, new-client flag. Weak unconfirmed candidates: account age
150–180 months in low risk; age 61–65.

## What a model needs, therefore

- Inputs: `banda_riesgo`, `numero_productos`, `dias_ultima_transaccion`, `activo_movil`, `tiene_tarjeta_credito`,
  `ratio_deuda_ingresos`, `canal_adquisicion == branch` (7 variables). Optionally the calendar month as a fixed effect when training on all months.
- Interactions up to order 3 (band × products × app; band × recency step; app × debt; band × branch). Depth-3 trees or an explicit rule logistic both capture them.
- Expected ceiling: Gini ≈ 0.26 (Rick's self-consistency estimate 0.262); November-like months 0.23–0.25.

## Quantities still unknown

1. Combination scale: additive in probability, multiplicative, or logit?
2. Exact thresholds (180 vs 182 days; 0.55 debt; products 3) and whether steps are hard or smooth.
3. The month level: trend, re-centring to a target rate, or random shocks? (Irrelevant for the December ranking.)
4. Whether any further gated rule exists (search each variable inside each band × app cell, with strict p and replication).
