"""H2, blind scan follow-up: is "which neurons over- or under-activate" the pattern, or is it the push score?

Uses the scan saved by experiments/h2_scan_then_screen.py (results/h2/scan_u.pt: per-layer repeat direction u,
per-neuron Cohen's d and push), so no new scan is needed. Separates the two choices a scan makes, layers and neurons,
using the same edit formula (W' = W - 6 u (mask * u^T W), top-64 neurons per layer) on the 12 held-out answers:

  absd_in_astra_layers   Astra's layers 15 + 19, neurons with the largest |d| (most over- OR under-active)
  push_in_astra_layers   Astra's layers 15 + 19, neurons with the largest push
  absd_pipeline          layers and neurons both chosen by |d| alone (top-2 layers by summed top-64 |d|)

Base and Astra numbers come from results/h2/rediscover_loop_edit.json (same float32 scoring).

Usage: python experiments/h2_scan_ablate.py
"""
import json
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "verify"))
sys.path.insert(0, os.path.join(REPO, "experiments"))
from exp8_qwen35_edit import MODEL, REV, token_logprobs  # noqa: E402
from h2_blind_neuron_scan import make_edit  # noqa: E402
from h2_rediscover_loop_edit import candidates, with_edit  # noqa: E402
from h2_scan_then_screen import summarize  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

OUT = os.path.join(REPO, "results", "h2")


def main():
    seg, k = 768, 64
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REV, dtype=torch.float32).eval()
    evals = candidates(tok, seg)[:12]
    S = torch.load(os.path.join(OUT, "scan_u.pt"))
    res = [{"u": S["u"][l], "d": S["d"][l].numpy(), "push": S["push"][l].numpy()} for l in range(len(S["u"]))]
    absd_layer = np.array([np.sort(np.abs(r["d"]))[::-1][:k].sum() for r in res])
    d_layers = sorted(np.argsort(-absd_layer)[:2].tolist())
    top = lambda l, v: np.argsort(-v)[:k].tolist()
    picks = {"absd_in_astra_layers": {l: top(l, np.abs(res[l]["d"])) for l in (15, 19)},
             "push_in_astra_layers": {l: top(l, res[l]["push"]) for l in (15, 19)},
             "absd_pipeline": {l: top(l, np.abs(res[l]["d"])) for l in d_layers}}
    edits = {nm: make_edit(model, res, p) for nm, p in picks.items()}
    print(f"|d| pipeline layers: {d_layers}", flush=True)

    sd = dict(model.named_parameters())
    names = set().union(*[set(e) for e in edits.values()])
    base_sd = {n: sd[n].detach().clone() for n in names}
    prev = {r["id"]: r for r in json.load(open(os.path.join(OUT, "rediscover_loop_edit.json")))["per_sample"]}
    rows = []
    for c in evals:
        s0, e = c["onset"] - seg, c["onset"] + seg
        p = prev[c["id"]]
        r = {"id": c["id"], **{x: p[x] for x in ("base_prod", "base_loop", "astra_prod", "astra_loop")}}
        for nm, ed in edits.items():
            lp = token_logprobs(with_edit(base_sd, model, ed), c["ids"], s0, e)
            r[f"{nm}_prod"], r[f"{nm}_loop"] = float(lp[:seg].mean()), float(lp[seg:].mean())
        rows.append(r)
        print("  " + c["id"].ljust(34) + "  ".join(
            f"{nm}: {r[nm + '_prod'] - r['base_prod']:+.4f}/{r[nm + '_loop'] - r['base_loop']:+.4f}"
            for nm in ("astra", *edits)), flush=True)
    with_edit(base_sd, model, {})
    summ = {nm: summarize([r[f"{nm}_prod"] - r["base_prod"] for r in rows], [r[f"{nm}_loop"] - r["base_loop"] for r in rows])
            for nm in ("astra", *edits)}
    print(json.dumps(summ, indent=1), flush=True)
    json.dump({"absd_pipeline_layers": d_layers, "picks": picks, "summary": summ, "per_sample": rows},
              open(os.path.join(OUT, "scan_ablate.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
