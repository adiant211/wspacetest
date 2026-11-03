"""One benchmark for every arm of the math surgery study, on two disjoint splits.

  split "screen" (held-in, used to choose among candidate edits): arithmetic seed 1, GSM8K *train* problems 200+,
                 Wikipedia scan articles 100+  (the neuron scan uses GSM8K train 0-99, Wikipedia scan 0-49, seeds 5e4+)
  split "test"   (held-out, every reported number): arithmetic seed 2, GSM8K *test* problems, Wikipedia test articles

Metrics
  arith/<lang>/<kind>   few-shot greedy exact match and mean log P(answer) (wsp/math_eval.eval_arith)
  gsm/<lang>            GSM8K solutions, teacher-forced: accuracy and NLL on calculation results (the first number
                        after "="), NLL on all solution tokens, NLL on the final answer
  wiki/<lang>           Wikipedia NLL over the first 256 tokens (collateral damage)
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from wsp.math_eval import arith_set, eval_arith, eval_gsm, eval_text  # noqa: E402

DATA = os.path.join(REPO, "results", "math", "data")
KINDS = ["1+1", "2+1", "2+2", "2-1", "2-2", "1*1", "2*1"]
LANGS = ["sym", "en", "tr"]
SPLITS = {
    "screen": {"seed": 1, "n_arith": 60, "gsm": ("scan", 200, 50), "wiki": ("scan", 100, 30)},
    "test": {"seed": 2, "n_arith": 100, "gsm": ("test", 0, 150), "wiki": ("test", 0, 60)},
}


def load_split(split):
    cfg = SPLITS[split]
    out = {"arith": {}, "gsm": {}, "wiki": {}}
    for lang in LANGS:
        for kind in KINDS:
            out["arith"][(lang, kind)] = arith_set(kind, lang, cfg["n_arith"], seed=cfg["seed"])
    for lang in ("en", "tr"):
        part, s, n = cfg["gsm"]
        out["gsm"][lang] = json.load(open(os.path.join(DATA, f"gsm_{lang}.json")))[part][s:s + n]
        part, s, n = cfg["wiki"]
        out["wiki"][lang] = json.load(open(os.path.join(DATA, f"wiki_{lang}.json")))[part][s:s + n]
    return out


def bench(model, tok, data, parts=("arith", "gsm", "wiki"), keep_rows=True):
    t0 = time.time()
    res, rows = {}, {}
    if "arith" in parts:
        for (lang, kind), (prefix, probs) in data["arith"].items():
            c, lp = eval_arith(model, tok, prefix, probs)
            res[f"arith/{lang}/{kind}/acc"] = float(c.mean())
            res[f"arith/{lang}/{kind}/logp"] = float(lp.mean())
            rows[f"arith/{lang}/{kind}"] = {"correct": c.tolist(), "logp": lp.tolist()}
        for lang in LANGS:
            res[f"arith/{lang}/acc"] = float(np.mean([res[f"arith/{lang}/{k}/acc"] for k in KINDS]))
            res[f"arith/{lang}/logp"] = float(np.mean([res[f"arith/{lang}/{k}/logp"] for k in KINDS]))
        res["arith/acc"] = float(np.mean([res[f"arith/{l}/acc"] for l in LANGS]))
        res["arith/logp"] = float(np.mean([res[f"arith/{l}/logp"] for l in LANGS]))
    if "gsm" in parts:
        for lang, items in data["gsm"].items():
            agg, per = eval_gsm(model, tok, items, lang)
            res[f"gsm/{lang}/result_acc"] = agg["result"][2] / agg["result"][1]
            res[f"gsm/{lang}/result_nll"] = agg["result"][0] / agg["result"][1]
            res[f"gsm/{lang}/solution_nll"] = agg["solution"][0] / agg["solution"][1]
            res[f"gsm/{lang}/final_nll"] = agg["final"][0] / agg["final"][1]
            res[f"gsm/{lang}/n_result"] = agg["result"][1]
            rows[f"gsm/{lang}"] = per
    if "wiki" in parts:
        for lang, docs in data["wiki"].items():
            r = eval_text(model, tok, docs)
            res[f"wiki/{lang}/nll"] = float(r.mean())
            rows[f"wiki/{lang}"] = r.tolist()
    res["seconds"] = time.time() - t0
    return (res, rows) if keep_rows else res
