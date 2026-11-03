"""H2, blind version + a small causal screen: scan, shortlist layers, test each shortlisted layer once, keep the best.

experiments/h2_blind_neuron_scan.py showed that a one-pass neuron scan ranks Astra's layer 19 first and layer 15 fourth,
but its own top-2 pick (layers 19 + 22) gives under half of Astra's selectivity. The scan score says which neurons
push the residual toward "repeating"; it does not say whether removing that push changes the next-token
distribution. So we add one cheap causal step, still without hints and still never touching the held-out answers:

  1. scan      same as before: 16 looping answers (cands[24:40]), every MLP neuron, labels = repeating / not.
  2. shortlist the top-6 layers by the push score.
  3. screen    for each shortlisted layer alone, apply the scan's edit (top-64 push neurons, W' = W - 6 u (mask * u^T W))
               and measure teacher-forced selectivity (loop delta minus pre-loop delta) on 4 of the *scan* answers.
               6 candidate edits x 4 answers, instead of a search over 86,016 neurons.
  4. keep      the 2 layers with the most negative selectivity; build the 2-layer edit.
  5. test      on the 12 held-out answers (cands[:12]) used for every other loop result, against a random edit of
               the same size in the same layers. Base and Astra numbers come from results/h2/rediscover_loop_edit.json.

Also saves the scan's per-layer directions u (results/h2/scan_u.pt) so later runs can skip step 1.

Usage: python experiments/h2_scan_then_screen.py [--shortlist 6] [--n-screen 4] [--keep 2]
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
from exp8_qwen35_edit import MODEL, REV, token_logprobs  # noqa: E402
from h2_blind_neuron_scan import make_edit, repeat_labels  # noqa: E402
from h2_rediscover_loop_edit import candidates, random_edit, with_edit  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from wsp.neuron_scan import layer_scores, scan, top_neurons  # noqa: E402

OUT = os.path.join(REPO, "results", "h2")


def window_logprobs(model, base_sd, edit, cands, seg):
    """Per answer: (mean token log-prob over the 768 tokens before the loop starts, over the 768 after)."""
    rows = []
    for c in cands:
        s0, e = c["onset"] - seg, c["onset"] + seg
        lp = token_logprobs(with_edit(base_sd, model, edit), c["ids"], s0, e)
        rows.append((float(lp[:seg].mean()), float(lp[seg:].mean())))
    return rows


def summarize(dp, dl):
    dp, dl = np.asarray(dp), np.asarray(dl)
    s = dl - dp
    return {"productive_delta_mean": float(dp.mean()), "loop_delta_mean": float(dl.mean()),
            "selectivity": float(s.mean()), "selectivity_se": float(s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 1 else None,
            "loop_hit_harder_in": int((dl < dp).sum()), "n": int(len(s))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-scan", type=int, default=16)
    ap.add_argument("--n-eval", type=int, default=12)
    ap.add_argument("--n-screen", type=int, default=4)
    ap.add_argument("--shortlist", type=int, default=6)
    ap.add_argument("--keep", type=int, default=2)
    ap.add_argument("--seg", type=int, default=768)
    ap.add_argument("--dtype", default="float32", choices=["bfloat16", "float32"])
    a = ap.parse_args()
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REV, dtype=getattr(torch, a.dtype)).eval()
    layers = list(model.model.layers)
    cands = candidates(tok, a.seg)
    evals, scans = cands[:a.n_eval], cands[24:24 + a.n_scan]
    screen_set = scans[:a.n_screen]
    timing = {}

    # 1. scan
    examples = []
    for c in scans:
        s0, e = c["onset"] - a.seg, c["onset"] + a.seg
        examples.append((c["ids"][:e], repeat_labels(c["ids"], c["plen"], s0, e)))
    res, timing["scan_s"] = scan(model, layers, examples)
    torch.save({"u": torch.stack([r["u"] for r in res]), "push": [torch.from_numpy(r["push"]) for r in res],
                "d": [torch.from_numpy(r["d"]) for r in res], "scan_ids": [c["id"] for c in scans]},
               os.path.join(OUT, "scan_u.pt"))
    ls = layer_scores(res)
    # 2. shortlist
    shortlist = np.argsort(-ls["push"])[:a.shortlist].tolist()
    print(f"scan {timing['scan_s']:.0f}s; shortlist by push: {shortlist}", flush=True)

    # 3. screen each shortlisted layer alone on scan answers
    sd = dict(model.named_parameters())
    names = [f"model.layers.{l}.mlp.down_proj.weight" for l in range(len(layers))]
    base_sd = {k: sd[k].detach().clone() for k in names}
    t0 = time.time()
    base_screen = window_logprobs(model, base_sd, {}, screen_set, a.seg)
    screen = {}
    for l in shortlist:
        ed = make_edit(model, res, {l: top_neurons(res, l, "push").tolist()})
        rows = window_logprobs(model, base_sd, ed, screen_set, a.seg)
        dp = [r[0] - b[0] for r, b in zip(rows, base_screen)]
        dl = [r[1] - b[1] for r, b in zip(rows, base_screen)]
        screen[l] = summarize(dp, dl)
        print(f"  screen L{l:2d}: prod {screen[l]['productive_delta_mean']:+.4f}  loop {screen[l]['loop_delta_mean']:+.4f}  "
              f"selectivity {screen[l]['selectivity']:+.4f}  [{time.time() - t0:.0f}s]", flush=True)
    timing["screen_s"] = time.time() - t0

    # 4. keep the best
    kept = sorted(sorted(shortlist, key=lambda l: screen[l]["selectivity"])[:a.keep])
    picks = {l: top_neurons(res, l, "push").tolist() for l in kept}
    edits = {"screened": make_edit(model, res, picks)}
    edits["screened_random"] = random_edit(model, edits["screened"])
    print(f"kept layers {kept}", flush=True)

    # 5. held-out test
    prev = {r["id"]: r for r in json.load(open(os.path.join(OUT, "rediscover_loop_edit.json")))["per_sample"]}
    blind = {r["id"]: r for r in json.load(open(os.path.join(OUT, "blind_neuron_scan.json")))["per_sample"]}
    rows = []
    t0 = time.time()
    for c in evals:
        s0, e = c["onset"] - a.seg, c["onset"] + a.seg
        p, b = prev[c["id"]], blind[c["id"]]
        r = {"id": c["id"], "base_prod": p["base_prod"], "base_loop": p["base_loop"],
             "astra_prod": p["astra_prod"], "astra_loop": p["astra_loop"],
             "blind_push_prod": b["blind_push_prod"], "blind_push_loop": b["blind_push_loop"]}
        for nm, ed in edits.items():
            lp = token_logprobs(with_edit(base_sd, model, ed), c["ids"], s0, e)
            r[f"{nm}_prod"], r[f"{nm}_loop"] = float(lp[:a.seg].mean()), float(lp[a.seg:].mean())
        rows.append(r)
        print("  " + c["id"].ljust(34) + "  ".join(
            f"{nm}: {r[nm + '_prod'] - r['base_prod']:+.4f}/{r[nm + '_loop'] - r['base_loop']:+.4f}"
            for nm in ("astra", "blind_push", *edits)), flush=True)
    timing["eval_s"] = time.time() - t0
    with_edit(base_sd, model, {})
    summ = {nm: summarize([r[f"{nm}_prod"] - r["base_prod"] for r in rows], [r[f"{nm}_loop"] - r["base_loop"] for r in rows])
            for nm in ("astra", "blind_push", *edits)}
    print(json.dumps(summ, indent=1), flush=True)
    json.dump({"dtype": a.dtype, "timing": timing, "shortlist": shortlist,
               "shortlist_push_scores": [float(ls["push"][l]) for l in shortlist],
               "screen_ids": [c["id"] for c in screen_set], "screen": {str(k): v for k, v in screen.items()},
               "kept": kept, "picks": picks, "summary": summ, "per_sample": rows},
              open(os.path.join(OUT, "scan_screen.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
