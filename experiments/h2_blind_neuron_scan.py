"""H2, blind version: find the loop neurons of Qwen3.5-0.8B with a neuron scan, *without* being told where to look.

experiments/h2_rediscover_loop_edit.py was handed Astra's layers (15 and 19). Here nothing is handed over:
  1. scan      wsp/neuron_scan.py records all 24 x 3,584 = 86,016 MLP neurons on 16 of the base model's looping
               MATH-500 answers. Each token is labelled "repeating" (the 12-gram ending there already occurred earlier
               in the answer) or "not repeating", over the 768 tokens before and after the loop starts.
  2. pattern   rank layers and neurons by how much they over- or under-activate while the model repeats itself
               (Cohen's d), how much that moves the residual stream (write), and how much it pushes along the
               residual's repeat-vs-not direction (push).
  3. compare   where do Astra's 128 neurons fall in those rankings? Do the scan's top layers include 15 and 19?
  4. test      edit the scan's own top-2 layers x top-64 neurons with Astra's formula (W' = W - 6 u (mask * u^T W)),
               then measure teacher-forced loop selectivity on the 12 held-out answers used before, against a random
               edit of the same size in the same layers. An activation-only variant (layers and neurons by Cohen's d)
               tests the simplest version of the idea.

Usage: python experiments/h2_blind_neuron_scan.py [--dtype float32]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "verify"))
sys.path.insert(0, os.path.join(REPO, "experiments"))
from exp8_qwen35_edit import MODEL, REV, load_edit, token_logprobs  # noqa: E402
from h2_rediscover_loop_edit import candidates, random_edit, with_edit  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from wsp.neuron_scan import layer_scores, scan, top_neurons  # noqa: E402

OUT = os.path.join(REPO, "results", "h2")


def repeat_labels(ids, plen, s0, e, n=12):
    """1 if the n-gram ending at token t already occurred earlier in the answer, 0 if not; -1 outside [s0, e)."""
    seen, lab = set(), [-1] * e
    for t in range(plen + n - 1, e):
        g = tuple(ids[t - n + 1:t + 1])
        if t >= s0:
            lab[t] = 1 if g in seen else 0
        seen.add(g)
    return lab


def make_edit(model, scan_res, picks, alpha=6.0):
    sd = dict(model.named_parameters())
    edit = {}
    for l, cols in picks.items():
        name = f"model.layers.{l}.mlp.down_proj.weight"
        W = sd[name].detach().float()
        u = scan_res[l]["u"]
        mask = torch.zeros(W.shape[1])
        mask[torch.as_tensor(cols)] = 1
        edit[name] = (W - alpha * torch.outer(u, mask * (u @ W))).to(torch.bfloat16)
    return edit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-scan", type=int, default=16)
    ap.add_argument("--n-eval", type=int, default=12)
    ap.add_argument("--seg", type=int, default=768)
    ap.add_argument("--dtype", default="float32", choices=["bfloat16", "float32"])
    a = ap.parse_args()
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REV, dtype=getattr(torch, a.dtype)).eval()
    layers = list(model.model.layers)
    cands = candidates(tok, a.seg)
    evals, scans = cands[:a.n_eval], cands[24:24 + a.n_scan]          # same split as h2_rediscover_loop_edit.py
    examples = []
    for c in scans:
        s0, e = c["onset"] - a.seg, c["onset"] + a.seg
        examples.append((c["ids"][:e], repeat_labels(c["ids"], c["plen"], s0, e)))
    n_rep = sum(sum(x == 1 for x in lab) for _, lab in examples)
    n_non = sum(sum(x == 0 for x in lab) for _, lab in examples)
    print(f"scan set: {len(examples)} answers, {n_rep:,} repeating and {n_non:,} non-repeating tokens; "
          f"{len(layers)} layers x {layers[0].mlp.down_proj.in_features} neurons", flush=True)

    # 1-2. scan + pattern
    res, scan_s = scan(model, layers, examples)
    ls = layer_scores(res)
    print(f"scan done in {scan_s:.0f}s", flush=True)
    for name in ("d", "write", "push", "residual_gap_rel"):
        order = np.argsort(-ls[name])
        print(f"  layers ranked by {name:16s}: {order[:6].tolist()}", flush=True)

    # 3. Astra's neurons in the scan's rankings
    _, astra = load_edit("p50c")
    sd = dict(model.named_parameters())
    astra_cols = {}
    for l in (15, 19):
        name = f"model.layers.{l}.mlp.down_proj.weight"
        dA = astra[name].float() - sd[name].detach().float()
        astra_cols[l] = sorted(torch.nonzero(dA.abs().sum(0) > 0).flatten().tolist())
    allv = {s: np.concatenate([r[s] for r in res]) for s in ("d", "write", "push")}
    compare = {}
    for s in ("d", "write", "push"):
        glob_rank = (-allv[s]).argsort().argsort()                    # 0 = most over-active / strongest push
        per = {}
        for l in (15, 19):
            idx = np.array(astra_cols[l]) + l * res[0][s].shape[0]
            within = (-res[l][s]).argsort().argsort()[astra_cols[l]]
            top64 = set(top_neurons(res, l, s).tolist())
            per[l] = {"median_global_percentile": float(100 * np.median(glob_rank[idx]) / len(glob_rank)),
                      "share_in_global_top1pct": float(np.mean(glob_rank[idx] < 0.01 * len(glob_rank))),
                      "median_rank_in_layer": float(np.median(within)),
                      "overlap_with_scan_top64": len(top64 & set(astra_cols[l])),
                      "share_positive": float(np.mean(res[l][s][astra_cols[l]] > 0))}
        compare[s] = per
        print(f"  Astra's neurons under '{s}': " + "; ".join(
            f"L{l}: median global percentile {v['median_global_percentile']:.2f}%, {v['share_in_global_top1pct']:.0%} in "
            f"global top 1%, overlap with scan top-64 {v['overlap_with_scan_top64']}/64" for l, v in per.items()), flush=True)

    # 4. causal test: the blind pipeline's own edits
    blind_layers = sorted(np.argsort(-ls["push"])[:2].tolist())
    act_layers = sorted(np.argsort(-ls["d"])[:2].tolist())
    picks_push = {l: top_neurons(res, l, "push").tolist() for l in blind_layers}
    picks_d = {l: top_neurons(res, l, "d").tolist() for l in act_layers}
    edits = {"blind_push": make_edit(model, res, picks_push), "blind_activation": make_edit(model, res, picks_d)}
    edits["random"] = random_edit(model, edits["blind_push"])
    print(f"blind picks: push -> layers {blind_layers}; activation-only -> layers {act_layers}", flush=True)

    prev = {r["id"]: r for r in json.load(open(os.path.join(OUT, "rediscover_loop_edit.json")))["per_sample"]}
    names = set().union(*[set(e) for e in edits.values()])
    base_sd = {k: sd[k].detach().clone() for k in names}
    rows = []
    for c in evals:
        s0, e = c["onset"] - a.seg, c["onset"] + a.seg
        p = prev[c["id"]]                  # base and Astra, scored in float32 by h2_rediscover_loop_edit.py
        r = {"id": c["id"], "base_prod": p["base_prod"], "base_loop": p["base_loop"],
             "astra_prod": p["astra_prod"], "astra_loop": p["astra_loop"]}
        for nm, ed in edits.items():
            lp = token_logprobs(with_edit(base_sd, model, ed), c["ids"], s0, e)
            r[f"{nm}_prod"], r[f"{nm}_loop"] = float(lp[:a.seg].mean()), float(lp[a.seg:].mean())
        rows.append(r)
        print("  " + c["id"].ljust(34) + "  ".join(
            f"{nm}: {r[nm + '_prod'] - r['base_prod']:+.4f}/{r[nm + '_loop'] - r['base_loop']:+.4f}"
            for nm in ("astra", *edits)), flush=True)
    with_edit(base_sd, model, {})
    summ = {}
    for nm in ("astra", *edits):
        dp = np.array([r[f"{nm}_prod"] - r["base_prod"] for r in rows])
        dl = np.array([r[f"{nm}_loop"] - r["base_loop"] for r in rows])
        summ[nm] = {"productive_delta_mean": float(dp.mean()), "loop_delta_mean": float(dl.mean()),
                    "selectivity": float((dl - dp).mean()), "selectivity_se": float((dl - dp).std(ddof=1) / np.sqrt(len(rows))),
                    "loop_hit_harder_in": int((dl < dp).sum())}
    print(json.dumps(summ, indent=1), flush=True)
    json.dump({
        "dtype": a.dtype, "scan_seconds": scan_s, "n_scan": len(examples), "tokens_repeating": n_rep, "tokens_not": n_non,
        "layer_scores": {k: v.tolist() for k, v in ls.items()}, "astra_cols": astra_cols, "astra_in_scan": compare,
        "blind_layers_push": blind_layers, "blind_layers_activation": act_layers,
        "picks_push": picks_push, "picks_activation": picks_d, "summary": summ, "per_sample": rows,
        "per_layer_neuron_scores": {s: [r[s].round(5).tolist() for r in res] for s in ("d", "push")},
    }, open(os.path.join(OUT, "blind_neuron_scan.json"), "w"))


if __name__ == "__main__":
    main()
