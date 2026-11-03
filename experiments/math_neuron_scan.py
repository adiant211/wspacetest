"""Find the math neurons of Ba2han/step-17k: one forward + backward pass per held-in text, no hints.

Every one of the 56 x 2304 = 129,024 SwiGLU neurons gets a gain g_j = 1 inserted in front of its w_down column (the
gains are leaf tensors, so the model's output is unchanged). One backward pass then gives dL/dg_j for all neurons at
once: the first-order change in a loss when neuron j's output is scaled, which is exactly what scaling its w_down
column does. The forward pass also accumulates activation statistics per condition.

Held-in data only (GSM8K *train* problems, the scan half of the Wikipedia articles, arithmetic with scan seeds):
  math_en / math_tr   positions inside an English / Turkish GSM8K solution (the activation that predicts the next
                      solution token)
  calc                positions whose next token is a calculation result (the first number after "="), both languages
  gen_en / gen_tr     Wikipedia positions
  arith               positions that predict an arithmetic answer token in few-shot "a op b" sequences (sym/en/tr)

Per neuron and condition c: mean activation, mean square, count, and G_c = d(mean NLL over c's targets)/dg.
Per layer and condition: mean residual stream after the block (for difference-of-means directions).

Saves results/math/scan.pt (tensors) and results/math/scan_summary.json.
Usage: python experiments/math_neuron_scan.py [--n-gsm 100] [--n-wiki 50] [--n-arith 8]
"""
import argparse
import json
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from wsp.hybrid_cpu import load  # noqa: E402
from wsp.math_eval import TEMPLATES, gsm_encode, make_problem  # noqa: E402

OUT = os.path.join(REPO, "results", "math")
CONDS = ["math_en", "math_tr", "calc", "gen_en", "gen_tr", "arith"]
ARITH_KINDS = ["1+1", "2+1", "2+2", "2-1", "2-2", "1*1", "2*1"]


def arith_sequence(tok, kind, lang, n_problems, seed):
    """One few-shot style sequence of n problems; targets = answer tokens of problems 4..n."""
    bos = tok.convert_tokens_to_ids("<bos>")
    da, op, db = kind[0], kind[1], kind[2]
    T = TEMPLATES[lang]
    rng = random.Random(seed)
    text, spans = "", []
    for k in range(n_problems):
        a, b, c = make_problem(rng, op, da, db)
        text += T["q"].format(a=a, b=b, op=T[op])
        s = len(text)
        text += f" {c}"
        if k >= 3:
            spans.append((s, len(text)))
        text += "\n\n"
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = [bos] + enc["input_ids"], [(0, 0)] + enc["offset_mapping"]
    tgt = torch.tensor([any(s <= o[0] < e or (o[0] < s < o[1]) for s, e in spans) for o in offs])
    return ids, tgt


def build_examples(tok, a):
    ex = []   # (ids, {cond: target mask over token positions})
    for lang in ("en", "tr"):
        items = json.load(open(os.path.join(OUT, "data", f"gsm_{lang}.json")))["scan"][:a.n_gsm]
        for it in items:
            ids, m = gsm_encode(tok, it, lang, maxlen=a.maxlen)
            ex.append((ids, {f"math_{lang}": torch.tensor(m["solution"] | m["final"]), "calc": torch.tensor(m["result"])}))
        docs = json.load(open(os.path.join(OUT, "data", f"wiki_{lang}.json")))["scan"][:a.n_wiki]
        bos = tok.convert_tokens_to_ids("<bos>")
        for d in docs:
            ids = ([bos] + tok(d["text"], add_special_tokens=False)["input_ids"])[:a.wiki_len]
            m = torch.ones(len(ids), dtype=torch.bool); m[0] = False
            ex.append((ids, {f"gen_{lang}": m}))
    for ki, kind in enumerate(ARITH_KINDS):
        for li, lang in enumerate(("sym", "en", "tr")):
            for r in range(a.n_arith):
                ids, m = arith_sequence(tok, kind, lang, 12, seed=50_000 + 1000 * r + 10 * ki + li)
                ex.append((ids, {"arith": m}))
    return ex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-gsm", type=int, default=100)
    ap.add_argument("--n-wiki", type=int, default=50)
    ap.add_argument("--n-arith", type=int, default=2)
    ap.add_argument("--maxlen", type=int, default=320)
    ap.add_argument("--wiki-len", type=int, default=256)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.manual_seed(0)
    model, tok = load(threads=a.threads)
    nL, I, D = model.n_layers, model.ffn, model.D
    ex = build_examples(tok, a)
    print(f"{len(ex)} sequences, {sum(len(i) for i, _ in ex)} tokens", flush=True)

    z = lambda *s: torch.zeros(*s, dtype=torch.float64)
    S = {c: {"sum": z(nL, I), "sq": z(nL, I), "n": 0, "grad": z(nL, I), "nt": 0, "res": z(nL, D)} for c in CONDS}
    gains = [torch.ones(I, requires_grad=True) for _ in range(nL)]
    acts = {}

    def act_hook(i, act):
        acts[i] = act.detach()
        return act * gains[i]

    resid = {}
    model.act_hook = act_hook
    model.resid_hook = lambda i, x: resid.__setitem__(i, x.detach())
    t0 = time.time()
    order = list(range(len(ex)))
    random.Random(0).shuffle(order)
    for k, j in enumerate(order):
        ids, masks = ex[j]
        x = torch.tensor([ids])
        logits = model(x)
        nll = F.cross_entropy(logits[0, :-1], x[0, 1:], reduction="none")
        # position t predicts token t+1: a target mask m over tokens selects positions m[1:]
        for g in gains:
            g.grad = None
        conds = list(masks)
        for ci, c in enumerate(conds):
            pm = masks[c][1:len(ids)]
            if pm.sum() == 0:
                continue
            loss = nll[pm].mean()
            loss.backward(retain_graph=ci < len(conds) - 1)
            S[c]["grad"] += torch.stack([g.grad.double() for g in gains]) * int(pm.sum())
            S[c]["nt"] += int(pm.sum())
            for g in gains:
                g.grad = None
            pos = torch.cat([pm, torch.tensor([False])])         # positions within the sequence
            for i in range(nL):
                av = acts[i][0][pos].double()
                S[c]["sum"][i] += av.sum(0); S[c]["sq"][i] += (av * av).sum(0)
                S[c]["res"][i] += resid[i][0][pos].double().sum(0)
            S[c]["n"] += int(pos.sum())
        if (k + 1) % 20 == 0:
            el = time.time() - t0
            print(f"  {k + 1}/{len(ex)} [{el:.0f}s, eta {el / (k + 1) * (len(ex) - k - 1):.0f}s]", flush=True)
    model.act_hook = model.resid_hook = None

    out = {"conds": CONDS, "args": vars(a), "seconds": time.time() - t0}
    for c in CONDS:
        n, nt = max(S[c]["n"], 1), max(S[c]["nt"], 1)
        out[c] = {"mean": (S[c]["sum"] / n).float(), "sq": (S[c]["sq"] / n).float(), "n": S[c]["n"],
                  "grad": (S[c]["grad"] / nt).float(), "res": (S[c]["res"] / n).float()}
    out["w_down_eff_norm"] = torch.stack([(model.out_gate(i)[:, None] * model.w_down(i)).norm(dim=0)
                                          for i in range(nL)]).detach()
    torch.save(out, os.path.join(OUT, "scan.pt"))
    json.dump({"seconds": out["seconds"], "tokens": {c: S[c]["n"] for c in CONDS}, "args": vars(a)},
              open(os.path.join(OUT, "scan_summary.json"), "w"), indent=1)
    print(f"done in {out['seconds']:.0f}s; tokens per condition: { {c: S[c]['n'] for c in CONDS} }")


if __name__ == "__main__":
    main()
