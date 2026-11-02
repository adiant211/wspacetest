"""H2 tools for Qwen2.5-0.5B-Instruct: ordinary, LLM-free ways to decide WHERE to fine-tune.

  grad_scores      task-specific. One backward pass of the task loss on a few dozen examples; for every linear module
                   the squared Frobenius norm of its gradient (the first-order loss drop a freshly initialised LoRA
                   there can deliver per unit step), summed per layer; and per MLP neuron j the squared norm of the
                   gradient of down_proj[:, j] (the column that neuron writes with).
  spectrum_scores  task-agnostic, weights only. A Spectrum-style signal-to-noise ratio (Hartford et al., 2024): the
                   singular values above the Marchenko-Pastur edge (noise level from the IQR of the spectrum) divided
                   by those below, normalised by the largest singular value; z-scored per module type, averaged per
                   layer.
  instruct_delta   weights only: how far instruction tuning moved each layer from Qwen2.5-0.5B (base).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .common import LINEAR_MODULES, core, get_linear, lm_head, load_model
from .train import batches, tokenize


def grad_scores(task_items, tok, n=64, bs=16, model=None):
    m = model or load_model(dtype=torch.float32)
    n_layers = len(core(m).layers)
    for l in range(n_layers):
        for mod in LINEAR_MODULES:
            get_linear(m, l, mod).weight.requires_grad_(True)
    data = tokenize(tok, task_items[:n])
    m.zero_grad(set_to_none=True)
    for ids, labels in batches(data, bs, shuffle=False):
        h = core(m)(input_ids=ids).last_hidden_state
        sel = labels[:, 1:] != -100
        logits = lm_head(m)(h[:, :-1][sel]).float()
        loss = F.cross_entropy(logits, labels[:, 1:][sel], reduction="sum") / sel.sum().clamp(min=1)
        (loss * ids.shape[0] / n).backward()
    layer, module, neuron = np.zeros(n_layers), {}, []
    for l in range(n_layers):
        for mod in LINEAR_MODULES:
            g = get_linear(m, l, mod).weight.grad.float()
            module[(l, mod)] = float((g ** 2).sum())
            layer[l] += module[(l, mod)]
        neuron.append((get_linear(m, l, "down_proj").weight.grad.float() ** 2).sum(0).numpy())
        for mod in LINEAR_MODULES:
            get_linear(m, l, mod).weight.requires_grad_(False)
            get_linear(m, l, mod).weight.grad = None
    return {"layer": layer, "module": module, "neuron": np.stack(neuron)}


def _snr(W: torch.Tensor) -> float:
    S = torch.linalg.svdvals(W.float())
    q75, q25 = torch.quantile(S, 0.75), torch.quantile(S, 0.25)
    sigma = (q75 - q25) / 1.349
    n, m_ = W.shape
    beta = min(n, m_) / max(n, m_)
    thr = sigma * (1 + np.sqrt(beta))
    signal, noise = S[S > thr].sum(), S[S <= thr].sum()
    return float((signal / noise) / S[0]) if noise > 0 else float("inf")


def spectrum_scores(model=None):
    m = model or load_model()
    n_layers = len(core(m).layers)
    snr = np.zeros((n_layers, len(LINEAR_MODULES)))
    for l in range(n_layers):
        for j, mod in enumerate(LINEAR_MODULES):
            snr[l, j] = _snr(get_linear(m, l, mod).weight.detach())
    z = (snr - snr.mean(0)) / (snr.std(0) + 1e-12)
    return {"layer": z.mean(1), "module_snr": snr}


def instruct_delta(base=None, inst=None):
    from .common import BASE
    b = base or load_model(BASE)
    i = inst or load_model()
    n_layers = len(core(i).layers)
    rel = np.zeros((n_layers, len(LINEAR_MODULES)))
    for l in range(n_layers):
        for j, mod in enumerate(LINEAR_MODULES):
            wb, wi = get_linear(b, l, mod).weight.float(), get_linear(i, l, mod).weight.float()
            rel[l, j] = float((wi - wb).norm() / wb.norm())
    return {"layer": rel.mean(1), "module": rel}
