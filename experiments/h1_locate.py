"""Step 1 of H1: decide where to fine-tune, with LLM-free tools only (wsp/locate.py), before any training.

Writes results/h1/picks_<purpose>.json with, per purpose:
  grad_top4 / grad_bottom4   the 4 layers with the largest / smallest task-gradient energy (task-specific tool)
  grad_top4_upper            the 4 upper-half layers (12-23) with the largest task-gradient energy: a cost-aware pick,
                             since backprop and the frozen-prefix cache then only cover the top of the network
  spectrum_top4              the 4 layers with the highest Spectrum-style SNR (weights only, task-agnostic)
  rand4                      two fixed random draws of 4 layers (controls)
  neurons_top                the 64 MLP neurons with the largest down_proj-column gradient energy in each of the
                             2 layers whose MLPs carry the most gradient (the size of Astra's Exp 8 edit)
  neurons_rand               64 random neurons in the same 2 layers (control)
and how long each tool took (part of the cost of hand-picking).

Usage: python experiments/h1_locate.py --purpose add|french
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wsp.common import RESULTS, load_model, load_tokenizer, seed_all  # noqa: E402
from wsp.locate import grad_scores, spectrum_scores  # noqa: E402
from wsp.tasks import add_items, french_data  # noqa: E402

OUT = os.path.join(RESULTS, "h1")
os.makedirs(OUT, exist_ok=True)


def purpose_items(purpose, tok):
    if purpose == "add":
        return add_items(3000, seed=1)
    return french_data(load_model(), tok)["train"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--purpose", required=True, choices=["add", "french"])
    a = ap.parse_args()
    seed_all(0)
    tok = load_tokenizer()
    items = purpose_items(a.purpose, tok)
    t = time.perf_counter()
    g = grad_scores(items, tok, n=64)
    t_grad = time.perf_counter() - t
    sp_path = os.path.join(OUT, "spectrum.json")
    if os.path.exists(sp_path):
        sp = json.load(open(sp_path))
    else:
        t = time.perf_counter()
        s = spectrum_scores()
        sp = {"layer": s["layer"].tolist(), "module_snr": s["module_snr"].tolist(), "seconds": time.perf_counter() - t}
        json.dump(sp, open(sp_path, "w"), indent=1)
    L = len(g["layer"])
    order = np.argsort(-g["layer"])
    mlp_energy = np.array([sum(g["module"][(l, m)] for m in ("gate_proj", "up_proj", "down_proj")) for l in range(L)])
    top_mlp = sorted(np.argsort(-mlp_energy)[:2].tolist())
    rng = random.Random(1234)
    neurons_top = {int(l): np.argsort(-g["neuron"][l])[:64].tolist() for l in top_mlp}
    neurons_rand = {int(l): sorted(rng.sample(range(g["neuron"].shape[1]), 64)) for l in top_mlp}
    rng2 = random.Random(7)
    picks = {
        "purpose": a.purpose,
        "grad_layer_energy": g["layer"].tolist(),
        "grad_share_by_layer": (g["layer"] / g["layer"].sum()).tolist(),
        "grad_top4": sorted(order[:4].tolist()),
        "grad_bottom4": sorted(order[-4:].tolist()),
        "grad_top4_upper": sorted([int(l) for l in order if l >= L // 2][:4]),   # cost-aware: upper half only
        "spectrum_top4": sorted(np.argsort(-np.array(sp["layer"]))[:4].tolist()),
        "rand4": [sorted(rng2.sample(range(L), 4)) for _ in range(2)],
        "top_mlp_layers": top_mlp,
        "neurons_top": neurons_top,
        "neurons_rand": neurons_rand,
        "neuron_grad_share_top64": {int(l): float(np.sort(g["neuron"][l])[::-1][:64].sum() / g["neuron"][l].sum()) for l in top_mlp},
        "tool_seconds": {"grad": t_grad, "spectrum": sp.get("seconds")},
    }
    json.dump(picks, open(os.path.join(OUT, f"picks_{a.purpose}.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in picks.items() if k not in ("grad_layer_energy", "neurons_top", "neurons_rand")}, indent=1))


if __name__ == "__main__":
    main()
