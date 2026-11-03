"""Neuron scan: benchmark every MLP neuron of a model under two conditions and rank the ones that over- or
under-activate, in one pass and without a frontier LLM.

Input: token sequences with a per-token label, 1 = condition A (e.g. "the model is repeating itself"), 0 = condition B
(e.g. "normal reasoning"), -1 = ignore. One forward pass per sequence; hooks on every MLP down_proj accumulate running
sums, so memory stays flat no matter how many neurons or tokens are scanned.

Per neuron j of layer l it returns
  mean_a, mean_b, d        activation means and Cohen's d (pooled-SD effect size): the over/under-activation benchmark
  write                    (mean_a - mean_b) * ||W_down[:, j]||: how much the neuron's extra activity moves the residual
  push                     (mean_a - mean_b) * (u_l . W_down[:, j]): how much it pushes along u_l, the unit
                           difference-of-means direction of the residual stream after layer l (A minus B)
and per layer the size of the A-vs-B residual gap. Works for any model whose MLPs end in a `down_proj` Linear.
"""
from __future__ import annotations

import time

import numpy as np
import torch


class _Acc:
    def __init__(self, width):
        z = lambda: torch.zeros(width, dtype=torch.float64)
        self.sa, self.sa2, self.sb, self.sb2 = z(), z(), z(), z()
        self.na = self.nb = 0


@torch.no_grad()
def scan(model, layers, examples, log_every=4):
    """model: HF causal LM; layers: list of decoder layers (each with .mlp.down_proj);
    examples: list of (ids, labels) with labels in {1, 0, -1}, same length as ids."""
    accs = [_Acc(L.mlp.down_proj.in_features) for L in layers]
    width = model.config.get_text_config().hidden_size if hasattr(model.config, "get_text_config") else model.config.hidden_size
    res = [{"sa": torch.zeros(width, dtype=torch.float64), "sb": torch.zeros(width, dtype=torch.float64)} for _ in layers]
    state = {}

    def make_hook(i):
        def hook(mod, inp):
            a = inp[0][0].float()                       # (T, I)
            ma, mb = state["ma"], state["mb"]
            acc = accs[i]
            xa, xb = a[ma].double(), a[mb].double()
            acc.sa += xa.sum(0); acc.sa2 += (xa * xa).sum(0); acc.na += int(ma.sum())
            acc.sb += xb.sum(0); acc.sb2 += (xb * xb).sum(0); acc.nb += int(mb.sum())
        return hook

    hooks = [L.mlp.down_proj.register_forward_pre_hook(make_hook(i)) for i, L in enumerate(layers)]
    t0 = time.time()
    try:
        for k, (ids, labels) in enumerate(examples):
            lab = torch.tensor(labels)
            state["ma"], state["mb"] = lab == 1, lab == 0
            out = model.model(torch.tensor([ids]), output_hidden_states=True)
            for i in range(len(layers)):
                h = out.hidden_states[i + 1][0].double()   # residual after layer i
                res[i]["sa"] += h[state["ma"]].sum(0)
                res[i]["sb"] += h[state["mb"]].sum(0)
            if log_every and (k + 1) % log_every == 0:
                print(f"    scanned {k + 1}/{len(examples)} sequences [{time.time() - t0:.0f}s]", flush=True)
    finally:
        for h in hooks:
            h.remove()

    result = []
    for i, L in enumerate(layers):
        acc = accs[i]
        ma, mb = acc.sa / acc.na, acc.sb / acc.nb
        va = (acc.sa2 / acc.na - ma ** 2).clamp(min=0)
        vb = (acc.sb2 / acc.nb - mb ** 2).clamp(min=0)
        pooled = ((acc.na * va + acc.nb * vb) / (acc.na + acc.nb)).sqrt() + 1e-8
        delta = ma - mb
        W = L.mlp.down_proj.weight.detach().double()      # (hidden, I)
        ra, rb = res[i]["sa"] / acc.na, res[i]["sb"] / acc.nb
        gap = ra - rb
        u = gap / gap.norm()
        result.append({
            "mean_a": ma.numpy(), "mean_b": mb.numpy(), "d": (delta / pooled).numpy(),
            "write": (delta * W.norm(dim=0)).numpy(), "push": (delta * (u @ W)).numpy(),
            "u": u.float(), "residual_gap": float(gap.norm()), "residual_gap_rel": float(gap.norm() / rb.norm()),
            "n_a": acc.na, "n_b": acc.nb,
        })
    return result, time.time() - t0


def layer_scores(result, k=64):
    """Per layer: how strongly its top-k neurons separate the conditions, under each neuron score."""
    out = {}
    for name in ("d", "write", "push"):
        out[name] = np.array([np.sort(np.abs(r[name]))[::-1][:k].sum() for r in result])
    out["residual_gap_rel"] = np.array([r["residual_gap_rel"] for r in result])
    return out


def top_neurons(result, layer, score="push", k=64, sign=+1):
    """The k neurons of `layer` with the largest score in the direction `sign` (+1: more active / pushing toward
    condition A; -1: the opposite)."""
    v = sign * result[layer][score]
    return np.argsort(-v)[:k]
