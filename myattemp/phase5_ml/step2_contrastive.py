"""Step 2: contrastive latent space + classification head.

Two pre-training objectives, same small encoder as step 1 (d -> 16 -> k; rule7 k=4, raw k=8) + projection k -> 8 -> k:
  scarf    self-supervised (Bahri et al. 2022, SCARF): a corrupted view replaces ~60% of the source columns with
           values from other random rows; InfoNCE pulls each row towards its own corrupted view, away from the others.
  supcon   supervised contrastive (Khosla et al. 2020): SCARF views, but every row with the same label counts as a positive.
Heads on the encoder (projection discarded): frozen + linear, frozen + (k -> 8 -> 1), fine-tuned encoder + (k -> 8 -> 1).
Baselines (logreg, mlp_direct) come from step 1 (out/ae_preds.npy), same folds and seeds.
Outputs: out/cl_summary.csv, out/cl_params.csv, figures/cl_gini.png
"""
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import (OUT, FIG, DEV, MONTHS, SEEDS, load, Encoder, folds, mlp, n_params, train, clf_loss, predict,
                    summarize)
from step1_autoencoder import Head, LATENT, HID, draw

CORRUPT = 0.6
TAU = 0.5


class Contrastive(nn.Module):
    def __init__(self, d, k, groups, supervised):
        super().__init__()
        self.enc, self.proj = mlp([d, HID, k]), mlp([k, 8, k])
        self.groups = torch.as_tensor(groups, device=DEV)
        self.n_src = int(max(groups)) + 1
        self.supervised = supervised

    def corrupt(self, x):
        """Replace a random ~60% of source columns (whole one-hot blocks) by another row's values."""
        n = len(x)
        mask_src = torch.rand(n, self.n_src, device=DEV) < CORRUPT
        mask = mask_src[:, self.groups]
        donor = x[torch.randint(0, n, (n,), device=DEV)]
        return torch.where(mask, donor, x)

    def loss(self, x, y):
        z1 = F.normalize(self.proj(self.enc(x)), dim=1)
        z2 = F.normalize(self.proj(self.enc(self.corrupt(x))), dim=1)
        z = torch.cat([z1, z2])
        sim = z @ z.T / TAU
        n = len(x)
        eye = torch.eye(2 * n, device=DEV, dtype=torch.bool)
        sim = sim.masked_fill(eye, -1e9)
        logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        idx = torch.arange(n, device=DEV)
        if not self.supervised:  # each row's positive is its own other view
            return -(logp[idx, idx + n].mean() + logp[idx + n, idx].mean()) / 2
        yy = torch.cat([y, y])
        pos = (yy[:, None] == yy[None, :]) & ~eye
        return -((logp * pos).sum(1) / pos.sum(1)).mean()


def cl_loss(model, xb, yb):
    if len(xb) <= 2048:
        return model.loss(xb, yb)
    # validation month: same batch size as training (the loss depends on it), averaged over chunks
    return torch.stack([model.loss(xb[i:i + 1024], yb[i:i + 1024]) for i in range(0, len(xb) - 1023, 1024)]).mean()


def main():
    torch.set_num_threads(8)
    d = load()
    y = d["objetivo"].values.astype(np.float32)
    y_by_m = {m: y[d.mes.values == m] for m in MONTHS}
    base = np.load(OUT / "ae_preds.npy", allow_pickle=True).item()
    preds = {k: v for k, v in base.items() if k.endswith(("logreg", "mlp_direct"))}
    prow = []
    for which in ["rule7", "raw"]:
        k = LATENT[which]
        for m, tr, va, te in folds(d):
            enc = Encoder(which).fit(d.iloc[tr])
            Xtr, Xva, Xte = (enc.transform(d.iloc[i]) for i in (tr, va, te))
            dim = Xtr.shape[1]
            t0 = time.time()
            per = {}
            for s in SEEDS:
                for obj in ("scarf", "supcon"):
                    torch.manual_seed(s)
                    cl = Contrastive(dim, k, enc.groups, supervised=(obj == "supcon")).to(DEV)
                    cl = train(cl, cl_loss, Xtr, Xva, y[tr], y[va], seed=s, batch=1024, max_epochs=60)[0]
                    for name, freeze, head, lr in [("lin", True, mlp([k, 1]), 3e-3),
                                                    ("mlp", True, mlp([k, 8, 1]), 3e-3),
                                                    ("finetune", False, mlp([k, 8, 1]), 1e-3)]:
                        e = mlp([dim, HID, k]).to(DEV)
                        e.load_state_dict(cl.enc.state_dict())
                        mdl = train(Head(e, head.to(DEV), freeze), clf_loss, Xtr, Xva, y[tr], y[va], lr=lr, seed=s)[0]
                        per.setdefault(f"{obj}_{name}", []).append(predict(mdl, Xte))
                    if m == MONTHS[0] and s == 0:
                        prow += [{"set": which, "model": f"{obj} pre-training (encoder + projection)", "params": n_params(cl)},
                                 {"set": which, "model": f"{obj} encoder + (k->8->1) head", "params": n_params(cl.enc, mlp([k, 8, 1]))}]
            for name, ps in per.items():
                preds.setdefault(f"{which} | {name}", {})[m] = np.mean(ps, 0)
            print(f"{which} {m} done in {time.time() - t0:.0f}s", flush=True)

    S = summarize(preds, y_by_m, "rule7 | mlp_direct", OUT / "cl_summary.csv")
    pd.DataFrame(prow).to_csv(OUT / "cl_params.csv", index=False)
    draw(S, name="cl")
    with pd.option_context("display.width", 220, "display.float_format", "{:.4f}".format):
        print(S.to_string(index=False))
        print(pd.DataFrame(prow).to_string(index=False))


if __name__ == "__main__":
    main()
