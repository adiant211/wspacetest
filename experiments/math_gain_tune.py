"""Train the targeted math neurons instead of hand-editing them, with three parameterizations of the same neurons:

  --mode gain   one log-gain per neuron: W_down[:, j] <- exp(theta_j) W_down[:, j]   (IA3, restricted to S; |S| params)
  --mode lora   a rank-r update on those neurons' w_down columns: W_down[:, S_l] += (alpha/r) B_l A_l per layer
                (B_l: d_model x r, zero-init; A_l: r x |S_l|; ~ 56*768*r + r*|S| params)
  --mode cols   full update of those columns (W_down[:, S] += Delta; 768*|S| params; the repo's H1 "neur" arm)

Every mode ends as a plain weight edit of the same w_down columns, so it is directly comparable with the manual edits
in experiments/math_surgery.py (which scale those columns with no training).

Objective (held-in data only, disjoint from the screen and test splits):
  L = mean NLL on math targets  +  lam * mean KL(p_base || p_edit) on Wikipedia tokens
  math targets: arithmetic answer tokens in 12-problem few-shot sequences (sym/en/tr, fresh seeds 70000+), GSM8K
                *train* calculation results and final answers (problems 0-199 and 250-399; the screen uses 200-249)
  KL text:      Wikipedia scan articles 0-99 (the screen uses 100-129), first 160 tokens
The KL term pins general behaviour to the unedited model instead of letting the edit also "improve" generic
language modelling. Same data order, steps and KL weight for every mode and neuron set.

Neuron sets: grad<N> (top N by |projected gain gradient| from the scan), d<N> (top N activation-selective math
neurons), rand<N> (N random neurons, the control), all (every neuron; with --mode lora this is plain LoRA on every
layer's w_down).

Checkpoints go to results/math/tune/<mode>-<set>/step<k>.edit.{json,pt}; evaluate them as arms
tune_<mode>-<set>_step<k> with experiments/math_surgery.py. train_log.json records seconds per step (wall time).
Usage: python experiments/math_gain_tune.py --mode lora --set grad4096 --steps 100
"""
import argparse
import json
import math
import os
import random
import re
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "experiments"))
from math_neuron_analyze import load_scores, topk_flat  # noqa: E402
from math_neuron_scan import ARITH_KINDS, arith_sequence  # noqa: E402
from math_surgery import gain_direction  # noqa: E402

from wsp.hybrid_cpu import load  # noqa: E402
from wsp.math_eval import gsm_encode  # noqa: E402
from wsp.neuron_edit import make_edit, save_edit  # noqa: E402

OUT = os.path.join(REPO, "results", "math", "tune")
DATA = os.path.join(REPO, "results", "math", "data")


def neuron_set(name, sc, seed=0):
    L, I = sc["w"].shape
    if name == "all":
        return [(l, j) for l in range(L) for j in range(I)]
    m = re.fullmatch(r"(grad|d|rand)(\d+)", name)
    kind, n = m[1], int(m[2])
    if kind == "grad":
        return topk_flat(np.abs(gain_direction(sc)), n)
    if kind == "d":
        return topk_flat(sc["d"], n)
    flat = np.random.default_rng(seed).choice(L * I, n, replace=False)
    return [(int(f // I), int(f % I)) for f in flat]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["gain", "lora", "cols"], default="gain")
    ap.add_argument("--set", required=True)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=None, help="default: gain 0.02, lora 1e-3, cols 3e-4")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    lr = a.lr or {"gain": 0.02, "lora": 1e-3, "cols": 3e-4}[a.mode]
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    tag = f"{a.mode}-{a.set}"
    od = os.path.join(OUT, tag)
    os.makedirs(od, exist_ok=True)
    sc, _ = load_scores()
    S = neuron_set(a.set, sc)
    model, tok = load(threads=a.threads)
    I, D = model.ffn, model.D
    by_layer = {}
    for l, j in S:
        by_layer.setdefault(l, []).append(j)
    idx = {l: torch.tensor(sorted(js)) for l, js in by_layer.items()}
    params = {}
    scale = a.alpha / a.rank
    for l, ix in idx.items():
        n = len(ix)
        if a.mode == "gain":
            params[l] = {"theta": torch.zeros(n, requires_grad=True)}
        elif a.mode == "lora":
            A = torch.empty(a.rank, n)
            torch.nn.init.kaiming_uniform_(A, a=math.sqrt(5))
            params[l] = {"A": A.requires_grad_(), "B": torch.zeros(D, a.rank, requires_grad=True)}
        else:
            params[l] = {"delta": torch.zeros(D, n, requires_grad=True)}
    flat_params = [p for d in params.values() for p in d.values()]
    n_params = sum(p.numel() for p in flat_params)
    opt = torch.optim.Adam(flat_params, lr=lr)
    on = {"v": True}

    def act_hook(i, act):
        if not on["v"] or i not in idx or a.mode != "gain":
            return None
        return act * torch.ones(I).index_put((idx[i],), params[i]["theta"].exp())

    def down_hook(i, act):
        if not on["v"] or i not in idx or a.mode == "gain":
            return None
        sub = act[..., idx[i]]
        if a.mode == "lora":
            return (sub @ params[i]["A"].T) @ params[i]["B"].T * scale
        return sub @ params[i]["delta"].T

    model.act_hook, model.down_hook = act_hook, down_hook

    gsm = []
    for lang in ("en", "tr"):
        items = json.load(open(os.path.join(DATA, f"gsm_{lang}.json")))["scan"]
        for it in items[:200] + items[250:]:
            ids, m = gsm_encode(tok, it, lang, maxlen=320)
            tgt = torch.tensor(m["result"] | m["final"])
            if tgt.sum() > 0:
                gsm.append((ids, tgt))
    bos = tok.convert_tokens_to_ids("<bos>")
    wiki = []
    for lang in ("en", "tr"):
        for d in json.load(open(os.path.join(DATA, f"wiki_{lang}.json")))["scan"][:100]:
            wiki.append(([bos] + tok(d["text"], add_special_tokens=False)["input_ids"])[:160])
    langs = ["sym", "en", "tr"]

    def snapshot(step):
        with torch.no_grad():
            if a.mode == "gain":
                pairs = [(l, int(j), float(t.exp())) for l in idx for j, t in zip(idx[l].tolist(), params[l]["theta"])]
                save_edit(make_edit(pairs), os.path.join(od, f"step{step}.edit.json"), mode=a.mode, set=a.set,
                          step=step, lr=lr, lam=a.lam)
            else:
                lay = {}
                for l in idx:
                    dl = (params[l]["B"] @ params[l]["A"] * scale) if a.mode == "lora" else params[l]["delta"]
                    lay[l] = {"idx": idx[l].clone(), "delta": dl.detach().clone()}
                torch.save({"layers": lay, "mode": a.mode, "set": a.set, "step": step, "lr": lr, "lam": a.lam,
                            "rank": a.rank}, os.path.join(od, f"step{step}.edit.pt"))

    log = []
    t0 = time.time()
    print(f"{tag}: {len(S)} neurons in {len(idx)} layers, {n_params} trainable params, lr {lr}", flush=True)
    for step in range(1, a.steps + 1):
        ts = time.time()
        opt.zero_grad()
        kind, lang = rng.choice(ARITH_KINDS), rng.choice(langs)
        ids, m = arith_sequence(tok, kind, lang, 12, seed=70_000 + step * 7 + a.seed * 100_003)
        lm = 0.0
        for ids_, tgt in ((ids, m), rng.choice(gsm)):
            x = torch.tensor([ids_])
            nll = F.cross_entropy(model(x)[0, :-1], x[0, 1:], reduction="none")
            loss = nll[tgt[1:len(ids_)]].mean()
            loss.backward()
            lm += float(loss)
        x = torch.tensor([rng.choice(wiki)])
        with torch.no_grad():
            on["v"] = False
            lp_base = model(x)[0].log_softmax(-1)
            on["v"] = True
        lp = model(x)[0].log_softmax(-1)
        kl = (lp_base.exp() * (lp_base - lp)).sum(-1).mean()
        (a.lam * kl).backward()
        opt.step()
        log.append({"step": step, "math_nll": lm / 2, "kl": float(kl), "s": time.time() - ts})
        if step % 10 == 0:
            r = log[-10:]
            print(f"step {step} math_nll {np.mean([q['math_nll'] for q in r]):.3f} kl {np.mean([q['kl'] for q in r]):.4f} "
                  f"{np.mean([q['s'] for q in r]):.1f}s/step [{time.time() - t0:.0f}s]", flush=True)
        if step % a.save_every == 0 or step == a.steps:
            snapshot(step)
    json.dump({"args": vars(a), "lr": lr, "n_params": n_params, "n_neurons": len(S), "log": log,
               "seconds": time.time() - t0, "s_per_step_median": float(np.median([q["s"] for q in log]))},
              open(os.path.join(od, "train_log.json"), "w"))


if __name__ == "__main__":
    main()
