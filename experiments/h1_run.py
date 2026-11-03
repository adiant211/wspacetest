"""Step 2 of H1: train one arm (where the trainable weights go) for one purpose, then score it.

Arms
  full         LoRA r=8 on all 7 linear modules of all 24 layers (the usual default)
  full_r1      LoRA r=1 everywhere (about the parameter count of a 4-layer r=8 LoRA, spread thin)
  pick4        LoRA r=8 on the 4 layers the task-gradient tool ranks highest        (hand-picked by a tool)
  pick4r       LoRA r=8 on the 4 layers with the highest task / generic gradient ratio (tool pick with the
               model's generic depth profile divided out; see results/h1/generic_grad.json)
  pick4u       LoRA r=8 on the 4 upper-half layers it ranks highest                 (cost-aware tool pick)
  anti4        LoRA r=8 on the 4 layers it ranks lowest                            (control)
  rand4a/b     LoRA r=8 on 4 random layers                                         (controls)
  spectrum4    LoRA r=8 on the 4 layers with the highest weight-only SNR            (Spectrum-style, task-agnostic)
  neur         train only the down_proj columns of 64 tool-picked neurons in each of 2 layers (114,688 weights)
  neur_rand    the same, 64 random neurons in the same 2 layers                     (control)

Usage: python experiments/h1_run.py --purpose add --arm pick4 [--seed 0]
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wsp.common import RESULTS, chat, generate, load_model, load_tokenizer, peak_rss_mb, seed_all  # noqa: E402
from wsp.tasks import (Retention, add_correct, add_items, capitals_eval, french_data, french_score,  # noqa: E402
                       is_french)
from wsp.train import add_lora, add_neuron_deltas, train  # noqa: E402

OUT = os.path.join(RESULTS, "h1")
CFG = {  # steps, batch, LoRA lr, neuron lr
    "add": dict(steps=150, bs=16, lr=1e-3, lr_neur=3e-4),
    "french": dict(steps=120, bs=8, lr=1e-3, lr_neur=3e-4),
}


def build(model, arm, picks):
    L = 24
    if arm == "full":
        return add_lora(model, range(L)), "lora"
    if arm == "full_r1":
        return add_lora(model, range(L), r=1, alpha=2), "lora"
    if arm == "pick4":
        return add_lora(model, picks["grad_top4"]), "lora"
    if arm == "pick4r":
        return add_lora(model, picks["grad_rel_top4"]), "lora"
    if arm == "pick4u":
        return add_lora(model, picks["grad_top4_upper"]), "lora"
    if arm == "anti4":
        return add_lora(model, picks["grad_bottom4"]), "lora"
    if arm in ("rand4a", "rand4b"):
        return add_lora(model, picks["rand4"][0 if arm == "rand4a" else 1]), "lora"
    if arm == "spectrum4":
        return add_lora(model, picks["spectrum_top4"]), "lora"
    if arm == "neur":
        return add_neuron_deltas(model, {int(k): v for k, v in picks["neurons_top"].items()}), "neur"
    if arm == "neur_rand":
        return add_neuron_deltas(model, {int(k): v for k, v in picks["neurons_rand"].items()}), "neur"
    raise ValueError(arm)


def evaluate(purpose, model, tok, ret, data=None):
    r = {}
    t = time.perf_counter()
    if purpose == "add":
        ev, ood = add_items(300, seed=999), add_items(200, seed=998, digits=6)
        outs = generate(model, tok, [chat(tok, x["user"]) for x in ev], max_new_tokens=12)
        r["acc_5digit"] = sum(add_correct(o, x["answer"]) for o, x in zip(outs, ev)) / len(ev)
        outs = generate(model, tok, [chat(tok, x["user"]) for x in ood], max_new_tokens=12)
        r["acc_6digit_ood"] = sum(add_correct(o, x["answer"]) for o, x in zip(outs, ood)) / len(ood)
    else:
        outs = generate(model, tok, [chat(tok, x["user"]) for x in data["eval"]], max_new_tokens=48)
        r["french_rate"] = sum(map(is_french, outs)) / len(outs)
        r["french_score_mean"] = sum(map(french_score, outs)) / len(outs)
        r["examples"] = outs[:4]
        r["capitals_acc"], caps = capitals_eval(model, tok)
        r["capitals_french_rate"] = sum(map(is_french, caps)) / len(caps)
    r.update(ret(model))
    r["eval_s"] = time.perf_counter() - t
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--purpose", required=True, choices=["add", "french"])
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-cache", action="store_true", help="disable frozen-prefix caching")
    ap.add_argument("--dtype", default=os.environ.get("H1_DTYPE", "bfloat16"), choices=["bfloat16", "float32"],
                    help="frozen-model precision; float32 on CPUs without bf16 units (AMX / AVX512-BF16)")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    seed_all(a.seed)
    tok = load_tokenizer()
    picks = json.load(open(os.path.join(OUT, f"picks_{a.purpose}.json")))
    data = french_data(None, tok) if a.purpose == "french" else None   # cached by h1_locate.py
    items = add_items(3000, seed=1) if a.purpose == "add" else data["train"]
    import torch
    dtype = getattr(torch, a.dtype)
    model = load_model(dtype=dtype)
    res = {"purpose": a.purpose, "arm": a.arm, "seed": a.seed, "cache_prefix": not a.no_cache, "dtype": a.dtype}
    if a.arm != "base":
        # train first, alone in the process, so peak memory is the training's own
        model, kind = build(model, a.arm, picks)
        cfg = CFG[a.purpose]
        rss0 = peak_rss_mb()
        stats = train(model, tok, items, steps=cfg["steps"], bs=cfg["bs"], lr=cfg["lr"] if kind == "lora" else cfg["lr_neur"],
                      cache_prefix=not a.no_cache, seed=a.seed)
        stats["rss_after_model_load_mb"] = rss0
        res.update(kind=kind, train=stats)
    base = load_model(dtype=dtype)
    ret = Retention(tok, base_model=base, kinds=("kl_wiki", "kl_chat", "mmlu") if a.purpose == "add" else ("kl_wiki", "mmlu"))
    res["eval"] = evaluate(a.purpose, model, tok, ret, data)
    res["mmlu_base"] = ret.base_mmlu
    tag = f"{a.purpose}_{a.arm}_s{a.seed}" + ("_nocache" if a.no_cache else "")
    json.dump(res, open(os.path.join(OUT, f"{tag}.json"), "w"), indent=1, ensure_ascii=False)
    ev = res["eval"]
    print(tag, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in ev.items() if k != "examples"}),
          json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.get("train", {}).items()
                      if k not in ("loss_curve",)}), flush=True)


if __name__ == "__main__":
    main()
