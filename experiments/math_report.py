"""Tables and figures for the math-neuron study (reads results/math/{screen,test,tune}/, scan_summary.json,
neurons.json; writes results/math/report_tables.md and results/figures/math_*.png).

Every arm is scored on the same items, so differences from the base model are paired: per arithmetic problem
(correct 0/1), per GSM8K problem (calculation results right, pooled per problem), per Wikipedia article (mean NLL).
"""
import glob
import json
import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = os.path.join(REPO, "results", "math")
FIG = os.path.join(REPO, "results", "figures")

LABEL = {
    "base": "no edit",
    "mathneuro5x1.1": "manual: MathNeuro x1.1 (5,010 math-specific neurons)",
    "mathneuro5x1.25": "manual: MathNeuro x1.25",
    "gain1024x0.25": "manual: one-shot gradient gains +-25% (1,024 neurons)",
    "gain1024x0.5": "manual: one-shot gradient gains +-50% (1,024)",
    "gain256x0.5": "manual: one-shot gradient gains +-50% (256)",
    "gain4096x0.25": "manual: one-shot gradient gains +-25% (4,096)",
    "gainrand1024x0.5": "control: random gains +-50% (1,024 random neurons)",
    "ablate_d1024": "knock-out: 1,024 math-firing neurons",
    "ablate_rand1024": "knock-out: 1,024 random neurons",
    "amp_d1024x1.5": "manual: math-firing neurons x1.5 (1,024)",
    "tune_lora-grad4096_step50": "trained: LoRA r=8 on 4,096 targeted neurons, 50 steps",
    "tune_lora-grad4096_step100": "trained: LoRA r=8 on 4,096 targeted neurons",
    "tune_lora-rand4096_step100": "trained: LoRA r=8 on 4,096 random neurons",
    "tune_gain-grad4096_step100": "trained: learned gains on 4,096 targeted neurons",
    "tune_lora-all_step100": "trained: LoRA r=8 on every w_down (untargeted)",
}


_SEEN = None


def seen_arith():
    """(op, a, b) of every arithmetic problem shown during training (math_gain_tune.py, seed 0, 100 steps; all trained
    arms share this stream) or during the neuron scan (math_neuron_scan.py). Reconstructed from the seeded RNG streams,
    which also consume one draw for the GSM8K item and one for the Wikipedia article per step."""
    global _SEEN
    if _SEEN is not None:
        return _SEEN
    import random
    import sys
    sys.path.insert(0, REPO)
    sys.path.insert(0, os.path.join(REPO, "experiments"))
    from transformers import PreTrainedTokenizerFast
    from wsp.math_eval import gsm_encode, make_problem
    from math_neuron_scan import ARITH_KINDS
    tok = PreTrainedTokenizerFast(tokenizer_file="/home/user/models/step-17k/tokenizer.json", bos_token="<bos>")
    n_gsm = 0
    for lang in ("en", "tr"):
        items = json.load(open(os.path.join(M, "data", f"gsm_{lang}.json")))["scan"]
        for it in items[:200] + items[250:]:
            _, m = gsm_encode(tok, it, lang, maxlen=320)
            n_gsm += int((m["result"] | m["final"]).sum() > 0)
    seen = set()

    def add(kind, seed):
        r = random.Random(seed)
        for _ in range(12):
            x, y, _c = make_problem(r, kind[1], kind[0], kind[2])
            seen.add((kind[1], x, y))
    rng = random.Random(0)
    gsm_dummy, wiki_dummy = range(n_gsm), range(200)
    for step in range(1, 101):
        kind, _lang = rng.choice(ARITH_KINDS), rng.choice(["sym", "en", "tr"])
        add(kind, 70_000 + step * 7)
        rng.choice(gsm_dummy)
        rng.choice(wiki_dummy)
    for ki, kind in enumerate(ARITH_KINDS):
        for li in range(3):
            for r in range(2):
                add(kind, 50_000 + 1000 * r + 10 * ki + li)
    _SEEN = seen
    return seen


def arith_items(stage):
    """Per arithmetic row key, the list of (op, a, b) in row order (regenerated from the split's seeds)."""
    import sys
    sys.path.insert(0, os.path.join(REPO, "experiments"))
    from math_bench import load_split
    d = load_split(stage)
    return {f"arith/{lang}/{kind}": [(p["op"], p["a"], p["b"]) for p in probs] for (lang, kind), (_, probs) in d["arith"].items()}


def load_stage(stage):
    out = {}
    for f in sorted(glob.glob(os.path.join(M, stage, "*.json"))):
        if f.endswith(".edit.json"):
            continue
        d = json.load(open(f))
        out[d["summary"]["arm"]] = d
    return out


def paired(arm, base, kind):
    """(mean difference, SE) of a per-item metric, arm minus base."""
    ra, rb = arm["rows"], base["rows"]
    if kind == "arith":
        a = np.concatenate([np.array(ra[k]["correct"], float) for k in sorted(ra) if k.startswith("arith/")])
        b = np.concatenate([np.array(rb[k]["correct"], float) for k in sorted(rb) if k.startswith("arith/")])
    elif kind in ("arith2_unseen", "arith1"):
        items, seen = paired.items, seen_arith()
        av, bv = [], []
        for k in sorted(ra):
            if not k.startswith("arith/"):
                continue
            one = k.split("/")[2][0] == "1"
            if (kind == "arith1") != one:
                continue
            for i, it in enumerate(items[k]):
                if kind == "arith2_unseen" and it in seen:
                    continue
                av.append(ra[k]["correct"][i]); bv.append(rb[k]["correct"][i])
        a, b = np.array(av, float), np.array(bv, float)
    elif kind.startswith("calc/"):
        lang = kind.split("/")[1]
        a = np.array([r["result"][2] / r["result"][1] if r["result"][1] else np.nan for r in ra[f"gsm/{lang}"]])
        b = np.array([r["result"][2] / r["result"][1] if r["result"][1] else np.nan for r in rb[f"gsm/{lang}"]])
        keep = ~np.isnan(a) & ~np.isnan(b)
        a, b = a[keep], b[keep]
    elif kind.startswith("sol/"):
        lang = kind.split("/")[1]
        a = np.array([r["solution"][0] / max(r["solution"][1], 1) for r in ra[f"gsm/{lang}"]])
        b = np.array([r["solution"][0] / max(r["solution"][1], 1) for r in rb[f"gsm/{lang}"]])
    elif kind.startswith("wiki/"):
        a, b = np.array(ra[kind]), np.array(rb[kind])
    d = a - b
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


def paired_base(base, kind):
    """Base accuracy values on the same subset paired() uses."""
    items, seen = paired.items, seen_arith()
    out = []
    for k in sorted(base["rows"]):
        if not k.startswith("arith/"):
            continue
        one = k.split("/")[2][0] == "1"
        if (kind == "arith1") != one:
            continue
        for i, it in enumerate(items[k]):
            if kind == "arith2_unseen" and it in seen:
                continue
            out.append(base["rows"][k]["correct"][i])
    return out


def fmt(m, se, pct=False, nd=3):
    if pct:
        return f"{100 * m:+.1f} ± {100 * se:.1f}"
    return f"{m:+.{nd}f} ± {se:.{nd}f}"


def table(stage, arms):
    R = load_stage(stage)
    if "base" not in R:
        return f"_no {stage} results yet_\n"
    b = R["base"]
    bs = b["summary"]
    lines = [f"Base: arithmetic {100 * bs['arith/acc']:.1f}% (sym {100 * bs['arith/sym/acc']:.1f} / EN "
             f"{100 * bs['arith/en/acc']:.1f} / TR {100 * bs['arith/tr/acc']:.1f}); GSM8K calculation results "
             f"{100 * bs['gsm/en/result_acc']:.1f}% EN / {100 * bs['gsm/tr/result_acc']:.1f}% TR; "
             f"Wikipedia NLL {bs['wiki/en/nll']:.3f} EN / {bs['wiki/tr/nll']:.3f} TR.", "",
             "| arm | neurons | arithmetic, all (pts) | 2-digit, unseen (pts) | 1-digit facts (pts) | GSM8K calc EN (pts) | "
             "GSM8K calc TR (pts) | GSM8K solution NLL EN / TR | Wikipedia NLL EN | Wikipedia NLL TR |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    paired.items = arith_items(stage)
    lines[0] += (f" Base on the 2-digit problems unseen in training: {100 * np.mean(paired_base(b, 'arith2_unseen')):.1f}%; "
                 f"on the 1-digit facts: {100 * np.mean(paired_base(b, 'arith1')):.1f}%.")
    for arm in arms:
        if arm not in R or arm == "base":
            continue
        r, s = R[arm], R[arm]["summary"]
        lines.append("| " + " | ".join([
            LABEL.get(arm, arm), f"{s['n_neurons']:,}", fmt(*paired(r, b, "arith"), pct=True),
            fmt(*paired(r, b, "arith2_unseen"), pct=True), fmt(*paired(r, b, "arith1"), pct=True),
            fmt(*paired(r, b, "calc/en"), pct=True), fmt(*paired(r, b, "calc/tr"), pct=True),
            f"{paired(r, b, 'sol/en')[0]:+.3f} / {paired(r, b, 'sol/tr')[0]:+.3f}",
            fmt(*paired(r, b, "wiki/en")), fmt(*paired(r, b, "wiki/tr"))]) + " |")
    return "\n".join(lines) + "\n"


def walltime():
    scan = json.load(open(os.path.join(M, "scan_summary.json")))
    rows = ["| step | what | wall time | threads |", "|---|---|---|---|",
            f"| find the neurons | one forward + backward scan of all 129,024 neurons over 342 held-in texts "
            f"(55,453 tokens) | {scan['seconds'] / 60:.0f} min | 4 |"]
    S = load_stage("screen")
    man = [d["summary"]["seconds"] for a, d in S.items() if not a.startswith("tune_") and d["summary"]["seconds"] < 400]
    if man:
        rows.append(f"| make + screen one manual edit | no training; benchmark on the held-in split | "
                    f"{np.median(man) / 60:.1f} min each ({len(S)} screened) | 4 |")
    for f in sorted(glob.glob(os.path.join(M, "tune", "*", "train_log.json"))):
        t = json.load(open(f))
        a = t["args"]
        rows.append(f"| train {a['mode']} on {a['set']} | {a['steps']} steps, {t['n_params']:,} trainable params | "
                    f"{t['seconds'] / 60:.0f} min ({t['s_per_step_median']:.1f} s/step) | {a['threads']} |")
    T = load_stage("test")
    if T:
        rows.append(f"| held-out test of one arm | | {np.median([d['summary']['seconds'] for d in T.values()]) / 60:.0f} "
                    f"min each | 2 |")
    return "\n".join(rows) + "\n"


def figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIG, exist_ok=True)
    n = json.load(open(os.path.join(M, "neurons.json")))
    prof = n["layer_profile_top1000"]
    fig, ax = plt.subplots(figsize=(9, 3.4), dpi=150)
    x = np.arange(len(prof["d"]))
    ax.plot(x, prof["d"], color="#2a6fdb", lw=2, label="fire on math text (top 1,000 by Cohen's d, EN+TR)")
    ax.plot(x, prof["-G_calc"], color="#d9480f", lw=2, label="scaling them up lowers calculation loss (top 1,000)")
    ax.plot(x, prof["-G_arith"], color="#d9480f", lw=1.2, ls="--", label="... lowers arithmetic-answer loss (top 1,000)")
    for l in (1, 27):
        ax.axvline(l, color="#999", lw=0.8, ls=":")
    ax.text(27.3, ax.get_ylim()[1] * 0.92, "n-gram memory\ninjected", fontsize=7, color="#666")
    ax.set_xlabel("layer (56 blocks; every 4th is softmax attention)")
    ax.set_ylabel("neurons per layer")
    ax.set_title("Where the math neurons of Ba2han/step-17k are", loc="left", fontsize=10)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "math_neuron_layers.png"))
    plt.close(fig)

    T = load_stage("test")
    if "base" not in T:
        return
    b = T["base"]
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
    offsets = {"mathneuro5x1.1": (6, -12), "gain1024x0.25": (6, 6)}
    for arm, r in T.items():
        if arm == "base" or arm.startswith("ablate"):        # knock-outs are diagnostics, not candidate edits
            continue
        paired.items = arith_items("test")
        ga, _ = paired(r, b, "arith2_unseen")
        gc = (paired(r, b, "calc/en")[0] + paired(r, b, "calc/tr")[0]) / 2
        wk = (paired(r, b, "wiki/en")[0] + paired(r, b, "wiki/tr")[0]) / 2
        trained = arm.startswith("tune_")
        ax.scatter(wk, 100 * (ga + gc) / 2, s=60, color="#d9480f" if trained else "#2a6fdb",
                   marker="o" if "grad" in arm or arm.startswith(("gain", "mathneuro")) else "s", zorder=3)
        ax.annotate(LABEL.get(arm, arm).split(": ")[-1], (wk, 100 * (ga + gc) / 2), fontsize=6.5,
                    xytext=offsets.get(arm, (5, 3)), textcoords="offset points")
    ax.axhline(0, color="#999", lw=0.8)
    ax.axvline(0, color="#999", lw=0.8)
    ax.set_xlabel("collateral: change in Wikipedia NLL (nats/token, mean of EN and TR)")
    ax.set_ylabel("math gain (points; mean of unseen 2-digit\narithmetic and GSM8K calculation accuracy)")
    ax.set_title("Held-out math gain vs collateral. Blue: hand edits, orange: trained. Circle: targeted neurons, "
                 "square: random / all neurons", loc="left", fontsize=8)
    ax.set_xlim(-0.001, 0.0215)
    ax.set_ylim(min(-1.2, ax.get_ylim()[0]), ax.get_ylim()[1])
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "math_tradeoff.png"))
    plt.close(fig)


def main():
    screen_arms = ["gain256x0.5", "gain1024x0.25", "gain1024x0.5", "gain4096x0.25", "gainrand1024x0.5",
                   "ablate_d1024", "ablate_rand1024", "amp_d1024x1.5", "mathneuro5x1.1", "mathneuro5x1.25",
                   "tune_lora-grad4096_step50", "tune_lora-grad4096_step100"]
    test_arms = ["mathneuro5x1.1", "gain1024x0.25", "tune_gain-grad4096_step100", "tune_lora-grad4096_step100",
                 "tune_lora-rand4096_step100", "tune_lora-all_step100", "ablate_d1024", "ablate_rand1024"]
    md = ["## Held-out test (GSM8K test problems, new arithmetic, Wikipedia test articles)", "",
          table("test", test_arms), "## Screen (held-in, used to choose)", "", table("screen", screen_arms),
          "## Wall time", "", walltime()]
    open(os.path.join(M, "report_tables.md"), "w").write("\n".join(md))
    figures()
    print("\n".join(md))


if __name__ == "__main__":
    main()
