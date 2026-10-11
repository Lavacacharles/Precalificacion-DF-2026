# Phase 5: latent-space learning (2026-10-07, a try)

Train only. Walk-forward Jul–Nov. For test month m, networks train on months < m−1 and stop early on month m−1;
pre-training uses the same training rows. 3 seeds, predictions averaged. Small networks on purpose (300–1,600 parameters).
Inputs: `rule7` (7 rule variables, 13 columns after one-hot) and `raw` (22 usable columns, 38 after one-hot).

| File | What it does |
|---|---|
| `common.py` | input encoding, folds, small networks, training loop with early stopping, paired bootstrap |
| `step1_autoencoder.py` | autoencoder (d → 16 → k → 16 → d; k = 4 rule7, 8 raw), then heads on the latent (~45 min on GPU) |
| `step2_contrastive.py` | SCARF (self-supervised) and SupCon (supervised) contrastive encoders, same heads |

Reference: XGBoost on rule7 (phase 4) = 0.2560. Phase-2 rule logistic = 0.2550.

## Step 1: autoencoder (Gini, mean Jul–Nov; vs = paired gap to `rule7 | mlp_direct`, 95% CI)

| Variant | Parameters | rule7 | raw |
|---|---|---|---|
| logreg (no hidden layer, no interactions) | 14 / 39 | 0.2311 | 0.2340 |
| mlp_direct (d → 16 → 8 → 1) | 369 / 769 | **0.2491** | **0.2495** |
| AE frozen + linear head | 297 / 769 | 0.2024 | 0.1356 |
| AE frozen + small MLP head | 341 / 841 | 0.2330 | 0.1361 |
| AE fine-tuned + head | 341 / 841 | 0.2469 (−0.002, CI −0.006 to +0.002) | 0.2477 (−0.001, CI −0.008 to +0.005) |

* **The autoencoder's latent space is worse than the raw inputs.** Fine-tuning only climbs back to the direct network's level.
* **Why: an autoencoder keeps what varies most, not what predicts.** Share of each column rebuilt from the latent (raw, k = 8):
  age 0.92, income 0.96, address months 0.87 … but risk band 0.00, app 0.00, card 0.00, channel 0.00.
  The bottleneck spends itself on the noise columns with the most spread and throws away the risk band, the strongest input.
  On rule7 (k = 4) it keeps band and the numeric inputs but drops app and card (0.00), so the frozen heads lose the gates.
* The columns are mutually independent, so there is no shared structure to compress: any bottleneck must drop whole columns.
* All networks stay below the rule logistic and XGBoost (≈ 0.255). The plain linear model (no interactions) loses 0.018, so the gates matter.

## Step 2: contrastive pre-training (Gini, mean Jul–Nov; vs = paired gap to `rule7 | mlp_direct` 0.2491)

| Variant | rule7 | raw |
|---|---|---|
| SCARF (self-supervised) frozen + linear head | 0.1770 | 0.1681 |
| SCARF frozen + small MLP head | 0.2002 | 0.1770 |
| SCARF fine-tuned + head | 0.2482 (−0.001, CI −0.005 to +0.003) | 0.2473 (−0.001, CI −0.008 to +0.005) |
| SupCon (supervised) frozen + linear head | 0.2404 | 0.2424 |
| SupCon frozen + small MLP head | 0.2415 | 0.2465 |
| SupCon fine-tuned + head | 0.2479 (−0.001, CI −0.005 to +0.002) | 0.2479 (−0.001, CI −0.007 to +0.005) |

Encoder + projection: 368 (rule7) / 904 (raw) parameters.

* **Self-supervised contrastive learning (SCARF) learns nothing useful here**: frozen, it is the worst of all (0.17–0.20).
  It teaches the encoder to recognize a row from a corrupted copy of itself, which rewards remembering whichever columns are most distinctive,
  not the ones that predict. With independent columns there is no shared structure for it to discover (as with the autoencoder).
* **Supervised contrastive (SupCon) is much better frozen** (0.24), because the labels tell it what to keep, but it is still below the direct network.
* **Fine-tuning brings every pre-trained encoder back to the direct network's level (≈ 0.248), never above.** No CI excludes zero.

## Conclusion of phase 5

Latent-space learning does not help: no pre-trained representation (autoencoder, SCARF, SupCon) beats a small network trained directly on the label,
and every network stays below the rule logistic / XGBoost / blends (0.255–0.258). The data has no low-dimensional structure to discover:
the columns are independent and the signal is a small set of gated rules on 7 inputs.
