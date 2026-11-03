"""Gain surgery, iterated: learn one log-gain per neuron for a chosen neuron set S, so that the edit is still just
W_down[:, j] <- exp(theta_j) W_down[:, j] (the IA3 parameterization, restricted to S), but found by small Adam steps
instead of one first-order jump.

Objective (held-in data only, disjoint from the screen and test splits):
  L = mean NLL on math targets  +  lam * mean KL(p_base || p_edit) on Wikipedia tokens
  math targets: arithmetic answer tokens in 12-problem few-shot sequences (sym/en/tr, fresh seeds 70000+), GSM8K
                *train* calculation results and final answers (problems 0-199 and 250-399; the screen uses 200-249)
  KL text:      Wikipedia scan articles 0-99 (the screen uses 100-129), first 160 tokens
The KL term pins general behaviour to the unedited model instead of letting the edit also "improve" generic
language modelling.

Neuron sets: grad<N> (top N by |projected gain gradient| from the scan), d<N> (top N activation-selective math
neurons), rand<N> (N random neurons, the control), all (every neuron).

Checkpoints (edit JSONs) are written every --save-every steps to results/math/tune/<set>/step<k>.edit.json; pick one
with experiments/math_surgery.py-style evaluation on the screen split, then test.
Usage: python experiments/math_gain_tune.py --set grad4096 --steps 120
"""
import argparse
import json
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
    m = re.fullmatch(r"(grad|d|rand)(\d+)", name)
    if name == "all":
        return [(l, j) for l in range(L) for j in range(I)]
    kind, n = m[1], int(m[2])
    if kind == "grad":
        return topk_flat(np.abs(gain_direction(sc)), n)
    if kind == "d":
        return topk_flat(sc["d"], n)
    flat = np.random.default_rng(seed).choice(L * I, n, replace=False)
    return [(int(f // I), int(f % I)) for f in flat]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--save-every", type=int, default=40)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    od = os.path.join(OUT, a.set)
    os.makedirs(od, exist_ok=True)
    sc, _ = load_scores()
    S = neuron_set(a.set, sc)
    model, tok = load(threads=a.threads)
    nL, I = model.n_layers, model.ffn
    by_layer = {}
    for l, j in S:
        by_layer.setdefault(l, []).append(j)
    idx = {l: torch.tensor(sorted(js)) for l, js in by_layer.items()}
    theta = {l: torch.zeros(len(v), requires_grad=True) for l, v in idx.items()}
    opt = torch.optim.Adam(list(theta.values()), lr=a.lr)
    use_gain = {"on": True}

    def act_hook(i, act):
        if not use_gain["on"] or i not in idx:
            return None
        g = torch.ones(I).index_put((idx[i],), theta[i].exp())
        return act * g

    model.act_hook = act_hook
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
    log = []
    t0 = time.time()
    for step in range(1, a.steps + 1):
        opt.zero_grad()
        # math: one arithmetic sequence + one GSM8K sequence
        kind, lang = rng.choice(ARITH_KINDS), rng.choice(langs)
        ids, m = arith_sequence(tok, kind, lang, 12, seed=70_000 + step * 7 + a.seed * 100_003)
        lm = 0.0
        for ids_, tgt in ((ids, m), rng.choice(gsm)):
            x = torch.tensor([ids_])
            nll = F.cross_entropy(model(x)[0, :-1], x[0, 1:], reduction="none")
            loss = nll[tgt[1:len(ids_)]].mean()
            loss.backward()
            lm += float(loss)
        # general text: KL(base || edit)
        x = torch.tensor([rng.choice(wiki)])
        with torch.no_grad():
            use_gain["on"] = False
            lp_base = model(x)[0].log_softmax(-1)
            use_gain["on"] = True
        lp = model(x)[0].log_softmax(-1)
        kl = (lp_base.exp() * (lp_base - lp)).sum(-1).mean()
        (a.lam * kl).backward()
        opt.step()
        th = torch.cat([t.detach() for t in theta.values()])
        log.append({"step": step, "math_nll": lm / 2, "kl": float(kl), "theta_rms": float(th.pow(2).mean().sqrt()),
                    "theta_maxabs": float(th.abs().max())})
        if step % 10 == 0:
            r = log[-10:]
            print(f"step {step} math_nll {np.mean([q['math_nll'] for q in r]):.3f} kl {np.mean([q['kl'] for q in r]):.4f} "
                  f"theta rms {log[-1]['theta_rms']:.3f} max {log[-1]['theta_maxabs']:.2f} [{time.time() - t0:.0f}s]",
                  flush=True)
        if step % a.save_every == 0 or step == a.steps:
            pairs = [(l, int(j), float(np.exp(float(t)))) for l in idx for j, t in zip(idx[l].tolist(), theta[l].detach())]
            save_edit(make_edit(pairs), os.path.join(od, f"step{step}.edit.json"), set=a.set, step=step, lr=a.lr,
                      lam=a.lam)
    json.dump({"args": vars(a), "log": log, "seconds": time.time() - t0}, open(os.path.join(od, "train_log.json"), "w"))


if __name__ == "__main__":
    main()
