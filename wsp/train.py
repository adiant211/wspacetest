"""H1 training: LoRA on chosen layers (PEFT), or neuron-sparse fine-tuning of chosen MLP columns, on a frozen
bfloat16 Qwen2.5-0.5B-Instruct, with cost accounting.

Two cost savers come for free once the trainable part is confined to layers >= L_min:
  * backward stops at L_min (autograd never visits frozen layers below it), and
  * frozen-prefix caching: the hidden states entering layer L_min never change, so they are computed once per
    training example instead of once per epoch (`cache_prefix=True`).
"""
from __future__ import annotations

import math
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import LINEAR_MODULES, chat, core, embed, get_linear, lm_head, module_name, peak_rss_mb, run_layers


# --------------------------------------------------------------------------------------------- data
def tokenize(tok, items):
    out = []
    for it in items:
        p = tok(chat(tok, it["user"]), add_special_tokens=False)["input_ids"]
        a = tok(it["answer"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
        out.append((p + a, [-100] * len(p) + a))
    return out


def batches(data, bs, shuffle=True, seed=0):
    idx = list(range(len(data)))
    if shuffle:
        random.Random(seed).shuffle(idx)
    for b in range(0, len(idx), bs):
        chunk = [data[i] for i in idx[b:b + bs]]
        T = max(len(x) for x, _ in chunk)
        ids = torch.full((len(chunk), T), 151643, dtype=torch.long)      # right padding
        lab = torch.full((len(chunk), T), -100, dtype=torch.long)
        for i, (x, y) in enumerate(chunk):
            ids[i, :len(x)] = torch.tensor(x)
            lab[i, :len(y)] = torch.tensor(y)
        yield ids, lab


# ------------------------------------------------------------------------------------- adapters
def add_lora(model, layers, r=8, alpha=16, modules=LINEAR_MODULES):
    from peft import LoraConfig, get_peft_model
    targets = [module_name(l, m) for l in layers for m in modules]
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.0, target_modules=targets, bias="none")
    return get_peft_model(model, cfg)


class ColumnDelta(nn.Module):
    """y = W x + D x[cols]: trains only the output columns of the chosen MLP neurons (down_proj[:, cols])."""

    def __init__(self, base: nn.Linear, cols):
        super().__init__()
        self.base = base
        self.register_buffer("cols", torch.as_tensor(sorted(cols), dtype=torch.long))
        self.delta = nn.Parameter(torch.zeros(base.out_features, len(cols), dtype=torch.float32))

    def forward(self, x):
        y = self.base(x)
        return y + (x[..., self.cols].float() @ self.delta.T).to(y.dtype)


def add_neuron_deltas(model, picks: dict[int, list[int]]):
    for l, cols in picks.items():
        mlp = core(model).layers[l].mlp
        mlp.down_proj = ColumnDelta(mlp.down_proj, cols)
    return model


def trainable(model):
    return [p for p in model.parameters() if p.requires_grad]


# ------------------------------------------------------------------------------------------ train
def lowest_trainable_layer(model) -> int:
    for i, layer in enumerate(core(model).layers):
        if any(p.requires_grad for p in layer.parameters()):
            return i
    return len(core(model).layers)


def loss_on(model, ids, labels, h0=None, start=0):
    h = embed(model, ids) if h0 is None else h0
    h = run_layers(model, h, start)
    h = core(model).norm(h)
    sel = labels[:, 1:] != -100
    logits = lm_head(model)(h[:, :-1][sel]).float()
    return F.cross_entropy(logits, labels[:, 1:][sel])


def train(model, tok, items, steps, bs, lr, cache_prefix=True, seed=0, warmup=10, log_every=25):
    data = tokenize(tok, items)
    params = trainable(model)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup) * max(0.0, 1 - s / steps))
    start = lowest_trainable_layer(model) if cache_prefix else 0
    stats = {"trainable_params": sum(p.numel() for p in params), "lowest_trainable_layer": lowest_trainable_layer(model),
             "prefix_start": start, "steps": steps, "bs": bs, "lr": lr}
    t0 = time.perf_counter()
    cache = None
    if start > 0:   # frozen-prefix cache: one no-grad pass through layers [0, start) per training example
        cache = {}
        with torch.no_grad():
            for k, (ids, lab) in enumerate(batches(data, 32, shuffle=False)):
                h = run_layers(model, embed(model, ids), 0, start)
                for i in range(ids.shape[0]):
                    n = int((lab[i] != -100).nonzero().max()) + 1
                    cache[k * 32 + i] = h[i, :n].clone()
    stats["prefix_cache_s"] = time.perf_counter() - t0
    model.train()
    order, step, losses = [], 0, []
    t1 = time.perf_counter()
    epoch = 0
    while step < steps:
        idx = list(range(len(data)))
        random.Random(seed * 1000 + epoch).shuffle(idx)
        for b in range(0, len(idx), bs):
            if step >= steps:
                break
            chunk = idx[b:b + bs]
            T = max(len(data[i][0]) for i in chunk)
            ids = torch.full((len(chunk), T), 151643, dtype=torch.long)
            lab = torch.full((len(chunk), T), -100, dtype=torch.long)
            for j, i in enumerate(chunk):
                x, y = data[i]
                ids[j, :len(x)] = torch.tensor(x)
                lab[j, :len(y)] = torch.tensor(y)
            h0 = None
            if cache is not None:
                d = model.config.hidden_size if hasattr(model, "config") else cache[chunk[0]].shape[-1]
                h0 = torch.zeros(len(chunk), T, cache[chunk[0]].shape[-1], dtype=cache[chunk[0]].dtype)
                for j, i in enumerate(chunk):
                    h0[j, :cache[i].shape[0]] = cache[i]
            loss = loss_on(model, ids, lab, h0=h0, start=start)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            losses.append(loss.item())
            step += 1
            if log_every and step % log_every == 0:
                print(f"    step {step}/{steps} loss {sum(losses[-log_every:]) / log_every:.4f}", flush=True)
        epoch += 1
    model.eval()
    stats.update({"train_loop_s": time.perf_counter() - t1, "total_train_s": time.perf_counter() - t0,
                  "s_per_step": (time.perf_counter() - t1) / steps, "final_loss": sum(losses[-20:]) / min(20, len(losses)),
                  "loss_curve": losses[::5], "peak_rss_mb": peak_rss_mb(), "epochs": epoch})
    return stats
