"""Math-neuron surgery on Ba2han/step-17k: build candidate edits from the scan, screen them on held-in data, and test
the chosen ones on held-out data against random edits of the same size.

Every edit scales the w_down columns (= the outputs) of a set of MLP neurons: W_down[:, j] <- a_j W_down[:, j].

Arms (k = number of neurons, out of 129,024):
  ablate_d<k>          top-k math neurons by activation selectivity (Cohen's d, GSM8K vs Wikipedia, mean of EN and
                       TR), a_j = 0. The causal check: if these are math neurons, math should suffer far more than text.
  ablate_rand<k>       k random neurons, a_j = 0
  amp_d<k>x<a>         the same top-k by d, a_j = a  ("turn the math neurons up")
  mathneuro<p>x<a>     neuron-level MathNeuro (Christ et al. 2024): per layer, the top p% by ||write|| x RMS activation
                       on math text that are not in the top p% on Wikipedia; a_j = a
  gain<k>x<s>          projected gradient gains: v = -(g_math - proj_{g_gen} g_math), where g_* are the scan's
                       d(mean NLL)/d(gain) (math = GSM8K calculation results + arithmetic answers, gen = Wikipedia EN+TR,
                       each normalized by its spread over neurons). Keep the k largest |v|; a_j = 1 + s * sign(v_j).
                       To first order this lowers math loss and leaves Wikipedia loss unchanged.
  gainrand<k>x<s>      k random neurons, a_j = 1 +- s with random signs

Usage:
  python experiments/math_surgery.py --stage screen --arms base ablate_d1024 ...
  python experiments/math_surgery.py --stage test --arms base gain1024x0.25 gainrand1024x0.25
Each arm's result goes to results/math/<stage>/<arm>.json; existing files are skipped (resumable).
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "experiments"))
from math_bench import bench, load_split  # noqa: E402
from math_neuron_analyze import load_scores, mathneuro_specific, topk_flat  # noqa: E402

from wsp.hybrid_cpu import load  # noqa: E402
from wsp.neuron_edit import edited, load_edit, make_edit, n_neurons, save_edit  # noqa: E402

OUT = os.path.join(REPO, "results", "math")


def z(x):
    return x / (x.std() + 1e-12)


def gain_direction(sc, target=("calc", "arith"), general=("gen_en", "gen_tr")):
    gt = sum(z(sc[f"G_{c}"]) for c in target)
    gg = sum(z(sc[f"G_{c}"]) for c in general)
    v = -(gt - (gt * gg).sum() / (gg * gg).sum() * gg)
    return v


def build_arm(name, sc, seed=0):
    L, I = sc["w"].shape
    rng = np.random.default_rng(seed)
    if name == "base":
        return None
    if name.startswith("tune_"):                       # tune_<set>_step<k>: a learned-gain checkpoint
        st, step = name[5:].rsplit("_", 1)
        return load_edit(os.path.join(OUT, "tune", st, f"{step}.edit.json"))
    m = re.fullmatch(r"ablate_d(\d+)", name)
    if m:
        return make_edit((l, j, 0.0) for l, j in topk_flat(sc["d"], int(m[1])))
    m = re.fullmatch(r"ablate_rand(\d+)", name)
    if m:
        flat = rng.choice(L * I, int(m[1]), replace=False)
        return make_edit((f // I, f % I, 0.0) for f in flat)
    m = re.fullmatch(r"amp_d(\d+)x([\d.]+)", name)
    if m:
        return make_edit((l, j, float(m[2])) for l, j in topk_flat(sc["d"], int(m[1])))
    m = re.fullmatch(r"mathneuro([\d.]+)x([\d.]+)", name)
    if m:
        mask = mathneuro_specific(sc, frac=float(m[1]) / 100)
        return make_edit((l, j, float(m[2])) for l, j in zip(*np.nonzero(mask)))
    m = re.fullmatch(r"gain(\d+)x([\d.]+)", name)
    if m:
        v = gain_direction(sc)
        top = topk_flat(np.abs(v), int(m[1]))
        return make_edit((l, j, 1 + float(m[2]) * np.sign(v[l, j])) for l, j in top)
    m = re.fullmatch(r"gainrand(\d+)x([\d.]+)", name)
    if m:
        flat = rng.choice(L * I, int(m[1]), replace=False)
        sg = rng.choice([-1.0, 1.0], len(flat))
        return make_edit((f // I, f % I, 1 + float(m[2]) * s) for f, s in zip(flat, sg))
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["screen", "test"], required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    od = os.path.join(OUT, a.stage)
    os.makedirs(od, exist_ok=True)
    sc, _ = load_scores()
    model, tok = load(threads=a.threads)
    data = load_split(a.stage)
    for arm in a.arms:
        path = os.path.join(od, f"{arm}.json")
        if os.path.exists(path):
            print(f"skip {arm} (exists)")
            continue
        edit = build_arm(arm, sc)
        t0 = time.time()
        with edited(model, edit):
            res, rows = bench(model, tok, data)
        res["arm"], res["n_neurons"] = arm, n_neurons(edit) if edit else 0
        json.dump({"summary": res, "rows": rows}, open(path, "w"))
        if edit:
            save_edit(edit, os.path.join(od, f"{arm}.edit.json"), arm=arm)
        keys = ["arith/acc", "arith/sym/acc", "arith/en/acc", "arith/tr/acc", "gsm/en/result_acc",
                "gsm/tr/result_acc", "gsm/en/solution_nll", "gsm/tr/solution_nll", "wiki/en/nll", "wiki/tr/nll"]
        print(f"{arm:>22} n={res['n_neurons']:6d} " + " ".join(f"{k.replace('/acc', '').replace('arith/', 'ar/')}="
                                                           f"{res[k]:.4f}" for k in keys) + f" [{time.time() - t0:.0f}s]",
              flush=True)


if __name__ == "__main__":
    main()
