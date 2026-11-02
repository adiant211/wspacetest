"""H2 on the blog's headline result: can a short, deterministic tool pipeline (no frontier LLM) rediscover Astra's
Qwen3.5-0.8B loop edit (Experiment 8, run p50c)?

Astra spent 158 turns / 20 hours of max-effort reasoning and 50 forward passes. Its final report describes the recipe:
"unit repetition directions from matched productive/repetitive contexts at residual outputs 16/20; 64 selected MLP
writer neurons per layer; W = W0 - 6 * outer(u, mask64 * (u @ W0))" on layers 15 and 19.
This script runs that recipe as plain code, from inputs Astra never saw (the base model's looping MATH-500 answers
released with the evaluation), on problems DISJOINT from the ones the edits are then tested on:

  1. direction   u_l = unit(mean residual after layer l on loop tokens - mean on the productive tokens just before),
                 l in {15, 19}, from N_dir looping answers
  2. writers     neuron j's push along u_l = E[a_j] * (u_l . W_down[:, j]); score = loop push - productive push;
                 keep the top 64 per layer
  3. edit        W' = W - alpha * u (mask * (u^T W)), alpha = 6 as in Astra's final edit
  4. compare     cos(u_tool, u_astra), overlap of the 64-neuron sets, and the teacher-forced loop test of
                 verify/exp8_qwen35_edit.py on held-out answers: base vs Astra's edit vs the tool's edit vs a
                 norm-matched random edit (random unit direction, random 64 columns)

Usage: python experiments/h2_rediscover_loop_edit.py [--n-dir 16] [--n-eval 12]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "verify"))
from exp8_qwen35_edit import (MODEL, PREFIX, REV, SUFFIX, load_edit, load_texts, loop_onset,  # noqa: E402
                              token_logprobs)
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

OUT = os.path.join(REPO, "results", "h2")
os.makedirs(OUT, exist_ok=True)
LAYERS = (15, 19)


def candidates(tok, seg):
    from datasets import load_dataset
    probs = {r["unique_id"]: r["problem"] for r in load_dataset("HuggingFaceH4/MATH-500", split="test")}
    cands = []
    for r in load_texts("unedited"):
        ids = tok(PREFIX + probs[r["id"]] + SUFFIX + r["text"], add_special_tokens=False)["input_ids"]
        plen = len(tok(PREFIX + probs[r["id"]] + SUFFIX, add_special_tokens=False)["input_ids"])
        on = loop_onset(ids[plen:])
        if on is not None and on > seg + 64 and plen + on + seg < 6000:
            cands.append({"id": r["id"], "ids": ids, "plen": plen, "onset": plen + on})
    return sorted(cands, key=lambda c: c["id"])   # the same order verify/exp8_qwen35_edit.py uses


@torch.no_grad()
def collect(model, cands, seg):
    """Mean residual (after each layer in LAYERS) and mean MLP activation (down_proj input) on both windows."""
    acts = {}
    hooks = []
    for l in LAYERS:
        def hook(mod, inp, l=l):
            acts[l] = inp[0][0].float()
        hooks.append(model.model.layers[l].mlp.down_proj.register_forward_pre_hook(hook))
    S = {l: {"res_loop": 0, "res_prod": 0, "a_loop": 0, "a_prod": 0} for l in LAYERS}
    for c in cands:
        s0, on, e = c["onset"] - seg, c["onset"], c["onset"] + seg
        out = model.model(torch.tensor([c["ids"][:e]]), output_hidden_states=True)
        for l in LAYERS:
            h = out.hidden_states[l + 1][0].float()       # residual after layer l
            S[l]["res_loop"] += h[on:e].mean(0) / len(cands)
            S[l]["res_prod"] += h[s0:on].mean(0) / len(cands)
            S[l]["a_loop"] += acts[l][on:e].mean(0) / len(cands)
            S[l]["a_prod"] += acts[l][s0:on].mean(0) / len(cands)
    for h in hooks:
        h.remove()
    return S


def tool_edit(model, S, alpha=6.0, k=64):
    sd = dict(model.named_parameters())
    edit, info = {}, {}
    for l in LAYERS:
        W = sd[f"model.layers.{l}.mlp.down_proj.weight"].detach().float()
        u = S[l]["res_loop"] - S[l]["res_prod"]
        u = u / u.norm()
        proj = u @ W                                          # how much each neuron's column writes along u
        score = (S[l]["a_loop"] - S[l]["a_prod"]) * proj       # differential push along u (loop - productive)
        cols = torch.topk(score, k).indices
        mask = torch.zeros(W.shape[1])
        mask[cols] = 1
        W2 = W - alpha * torch.outer(u, mask * proj)
        edit[f"model.layers.{l}.mlp.down_proj.weight"] = W2.to(torch.bfloat16)
        info[l] = {"u": u, "cols": set(cols.tolist())}
    return edit, info


def random_edit(model, ref_edit, seed=0):
    """Same size of change as the tool edit, random unit direction, random 64 columns."""
    g = torch.Generator().manual_seed(seed)
    sd = dict(model.named_parameters())
    out = {}
    for name, w_new in ref_edit.items():
        W = sd[name].detach().float()
        target = (w_new.float() - W).norm()
        u = torch.randn(W.shape[0], generator=g)
        u = u / u.norm()
        cols = torch.randperm(W.shape[1], generator=g)[:64]
        mask = torch.zeros(W.shape[1])
        mask[cols] = 1
        d = torch.outer(u, mask * (u @ W))
        out[name] = (W - d * (target / d.norm())).to(torch.bfloat16)
    return out


def with_edit(base_sd, model, edit):
    sd = dict(model.named_parameters())
    for k, v in base_sd.items():
        sd[k].data.copy_(v)
    for k, v in edit.items():
        sd[k].data.copy_(v)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-dir", type=int, default=16)
    ap.add_argument("--n-eval", type=int, default=12)
    ap.add_argument("--seg", type=int, default=768)
    a = ap.parse_args()
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REV, dtype=torch.bfloat16).eval()
    cands = candidates(tok, a.seg)
    evals, dirs = cands[:a.n_eval], cands[24:24 + a.n_dir]   # eval = first answers of the Exp 8 test; dirs disjoint
    assert not {c["id"] for c in evals} & {c["id"] for c in dirs}
    print(f"{len(cands)} looping answers; direction set {len(dirs)}, eval set {len(evals)} (disjoint)", flush=True)

    t = time.time()
    S = collect(model, dirs, a.seg)
    edit, info = tool_edit(model, S)
    print(f"tool pipeline done in {time.time() - t:.0f}s", flush=True)

    _, astra = load_edit("p50c")
    base_sd = {k: dict(model.named_parameters())[k].detach().clone() for k in astra}
    cmp = {}
    for l in LAYERS:
        name = f"model.layers.{l}.mlp.down_proj.weight"
        dA = astra[name].float() - base_sd[name].float()
        colsA = set(torch.nonzero(dA.abs().sum(0) > 0).flatten().tolist())
        uA = torch.linalg.svd(dA[:, sorted(colsA)], full_matrices=False)[0][:, 0]
        cmp[l] = {"cos_u_tool_vs_astra": float(abs(info[l]["u"] @ uA)), "neuron_overlap_of_64": len(info[l]["cols"] & colsA),
                  "random_overlap_expected": 64 * 64 / base_sd[name].shape[1]}
        print(f"layer {l}: |cos(u_tool, u_astra)| = {cmp[l]['cos_u_tool_vs_astra']:.3f}; neuron overlap "
              f"{cmp[l]['neuron_overlap_of_64']}/64 (chance {cmp[l]['random_overlap_expected']:.1f})", flush=True)

    rand = random_edit(model, edit)
    prev = json.load(open(os.path.join(REPO, "results", "verify", "exp8_qwen35_edit.json")))
    prev = {r["id"]: r for r in prev.get("teacher", {}).get("per_sample", [])}
    rows = []
    for c in evals:
        s0, on, e = c["onset"] - a.seg, c["onset"], c["onset"] + a.seg
        r = {"id": c["id"]}
        if c["id"] in prev:
            p = prev[c["id"]]
            r.update(base_prod=p["productive_base"], base_loop=p["loop_base"], astra_prod=p["productive_edit"], astra_loop=p["loop_edit"])
        else:
            for nm, ed in (("base", {}), ("astra", astra)):
                lp = token_logprobs(with_edit(base_sd, model, ed), c["ids"], s0, e)
                r[f"{nm}_prod"], r[f"{nm}_loop"] = float(lp[:a.seg].mean()), float(lp[a.seg:].mean())
        for nm, ed in (("tool", edit), ("random", rand)):
            lp = token_logprobs(with_edit(base_sd, model, ed), c["ids"], s0, e)
            r[f"{nm}_prod"], r[f"{nm}_loop"] = float(lp[:a.seg].mean()), float(lp[a.seg:].mean())
        rows.append(r)
        print("  " + c["id"].ljust(36) + "  ".join(f"{nm}: prod {r[nm + '_prod'] - r['base_prod']:+.4f} loop {r[nm + '_loop'] - r['base_loop']:+.4f}"
                                             for nm in ("astra", "tool", "random")), flush=True)
    with_edit(base_sd, model, {})
    summ = {}
    for nm in ("astra", "tool", "random"):
        dp = np.array([r[f"{nm}_prod"] - r["base_prod"] for r in rows])
        dl = np.array([r[f"{nm}_loop"] - r["base_loop"] for r in rows])
        summ[nm] = {"productive_delta_mean": float(dp.mean()), "loop_delta_mean": float(dl.mean()),
                    "selectivity_loop_minus_productive": float((dl - dp).mean()),
                    "selectivity_se": float((dl - dp).std(ddof=1) / np.sqrt(len(rows)))}
    print(json.dumps(summ, indent=1), flush=True)
    json.dump({"layers": list(LAYERS), "n_dir": len(dirs), "n_eval": len(evals), "comparison": cmp, "summary": summ,
               "per_sample": rows, "tool_cols": {l: sorted(info[l]["cols"]) for l in LAYERS}},
              open(os.path.join(OUT, "rediscover_loop_edit.json"), "w"), indent=1)
    torch.save({k: v for k, v in edit.items()}, os.path.join(OUT, "tool_edit_tensors.pt"))


if __name__ == "__main__":
    main()
