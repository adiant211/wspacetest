"""Neuron-gain edits for the hybrid LM: W_down[:, j] <- s_j * W_down[:, j] for a set of (layer, neuron, scale).

Scaling neuron j's w_down column by s is exactly scaling that neuron's output (its SwiGLU activation) by s, so the
edit is a plain weight change that any loader (patch.py on GPU included) picks up. s = 0 ablates the neuron.

An edit is stored as {"layers": {layer: {"idx": [...], "scale": [...]}}}; `apply` returns an undo closure.
`write_checkpoint` saves a model.safetensors with the edit baked in, under the checkpoint's own tensor names.
"""
from __future__ import annotations

import json
import os

import torch


def make_edit(pairs):
    """pairs: iterable of (layer, neuron, scale) -> edit dict."""
    lay = {}
    for l, j, s in pairs:
        d = lay.setdefault(int(l), {"idx": [], "scale": []})
        d["idx"].append(int(j)); d["scale"].append(float(s))
    return {"layers": lay}


def n_neurons(edit):
    return sum(len(d["idx"]) for d in edit["layers"].values())


def n_weights(edit, d_model=768):
    return n_neurons(edit) * d_model


@torch.no_grad()
def apply(model, edit):
    """Per layer: W[:, idx] *= scale (if "scale" given) and/or W[:, idx] += delta (a [d_model, len(idx)] tensor, e.g. a
    merged LoRA update restricted to those columns)."""
    saved = []
    for l, d in edit["layers"].items():
        W = model.w_down(int(l))
        idx = torch.as_tensor(d["idx"], dtype=torch.long)
        saved.append((W, idx, W[:, idx].clone()))
        if "scale" in d:
            W[:, idx] *= torch.as_tensor(d["scale"], dtype=W.dtype)[None]
        if "delta" in d:
            W[:, idx] += torch.as_tensor(d["delta"], dtype=W.dtype)

    def undo():
        for W, idx, old in saved:
            W[:, idx] = old
    return undo


class edited:
    """with edited(model, edit): ..."""

    def __init__(self, model, edit):
        self.model, self.edit = model, edit

    def __enter__(self):
        self.undo = apply(self.model, self.edit) if self.edit else (lambda: None)
        return self.model

    def __exit__(self, *exc):
        self.undo()


def save_edit(edit, path, **meta):
    json.dump({**meta, "n_neurons": n_neurons(edit), "n_weights": n_weights(edit), **edit},
              open(path, "w"), indent=1)


def load_edit(path):
    if path.endswith(".pt"):
        d = torch.load(path)
    else:
        d = json.load(open(path))
    return {"layers": {int(k): v for k, v in d["layers"].items()}}


@torch.no_grad()
def write_checkpoint(src_dir, edit, out_dir):
    """Copy a checkpoint directory's small files and write model.safetensors with the edit applied (the n-gram table
    is unchanged; it is symlinked)."""
    from safetensors.torch import load_file, save_file
    os.makedirs(out_dir, exist_ok=True)
    sd = load_file(os.path.join(src_dir, "model.safetensors"))
    for l, d in edit["layers"].items():
        k = f"blocks.{int(l)}.mlp.w_down.weight"
        W = sd[k].clone()
        idx = torch.as_tensor(d["idx"], dtype=torch.long)
        if "scale" in d:
            W[:, idx] *= torch.as_tensor(d["scale"], dtype=W.dtype)[None]
        if "delta" in d:
            W[:, idx] += torch.as_tensor(d["delta"], dtype=W.dtype)
        sd[k] = W
    save_file(sd, os.path.join(out_dir, "model.safetensors"), metadata={"format": "pt"})
    for f in os.listdir(src_dir):
        if f in ("model.safetensors",) or f.startswith("."):
            continue
        dst = os.path.join(out_dir, f)
        if os.path.exists(dst):
            continue
        if f == "ngram.safetensors":
            os.symlink(os.path.join(src_dir, f), dst)
        else:
            with open(os.path.join(src_dir, f), "rb") as a, open(dst, "wb") as b:
                b.write(a.read())
