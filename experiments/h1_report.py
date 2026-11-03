"""Collect the H1 runs (results/h1/<purpose>_<arm>_s0.json) and the clean cost benchmark (results/h1/cost.json)
into markdown tables, and write results/h1/summary.json.

The accuracy runs shared the CPU with other jobs (and the later ones ran on a different host), so their own timings
are not reported; per-step cost comes only from experiments/h1_cost.py, where every arm was timed alone, one after
another, on the same idle machine and in the same precision.

Usage: python experiments/h1_report.py
"""
import glob
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H1 = os.path.join(REPO, "results", "h1")
ARMS = ["base", "full", "full_r1", "pick4", "pick4r", "pick4u", "spectrum4", "arb4", "rand4a", "rand4b", "anti4",
        "neur", "neur_rand"]
LABEL = {"base": "no fine-tune", "full": "LoRA, all 24 layers", "full_r1": "LoRA r=1, all 24 layers",
         "pick4": "LoRA, 4 layers: task-gradient tool", "pick4r": "LoRA, 4 layers: task / generic gradient",
         "pick4u": "LoRA, 4 layers: tool, upper half only", "spectrum4": "LoRA, 4 layers: Spectrum SNR (weights only)",
         "arb4": "LoRA, 4 arbitrary layers", "rand4a": "LoRA, 4 random layers (a)", "rand4b": "LoRA, 4 random layers (b)",
         "anti4": "LoRA, 4 layers the tool ranks lowest", "neur": "64 + 64 neurons: task-gradient tool",
         "neur_rand": "64 + 64 random neurons"}


def pct(v):
    return "-" if v is None else f"{v:.1%}"


def main():
    rows = {}
    for f in sorted(glob.glob(os.path.join(H1, "*_s0.json"))):
        r = json.load(open(f))
        rows.setdefault(r["purpose"], {})[r["arm"]] = r
    cost = {}
    if os.path.exists(os.path.join(H1, "cost.json")):
        for c in json.load(open(os.path.join(H1, "cost.json"))):
            cost[(c["purpose"], c["arm"])] = c
    out = {}
    for purpose in ("add", "french"):
        arms = rows.get(purpose, {})
        picks = json.load(open(os.path.join(H1, f"picks_{purpose}.json")))
        layers = {"pick4": picks["grad_top4"], "pick4u": picks.get("grad_top4_upper"), "pick4r": picks.get("grad_rel_top4"),
                  "anti4": picks["grad_bottom4"], "spectrum4": picks["spectrum_top4"], "arb4": picks.get("arb4"),
                  "rand4a": picks["rand4"][0], "rand4b": picks["rand4"][1], "neur": picks["top_mlp_layers"],
                  "neur_rand": picks["top_mlp_layers"], "full": "all", "full_r1": "all", "base": "-"}
        if purpose == "add":
            cols = [("acc_5digit", "5-digit"), ("acc_6digit_ood", "6-digit (unseen)"), ("kl_wiki", "KL Wikipedia"),
                    ("kl_chat", "KL chat"), ("mmlu", "MMLU-200")]
        else:
            cols = [("french_rate", "in French"), ("capitals_acc", "capitals right"), ("kl_wiki", "KL Wikipedia"),
                    ("mmlu", "MMLU-200")]
        hdr = ["arm", "layers"] + [c[1] for c in cols] + ["final train loss", "trainable weights", "precision"]
        print(f"\n### {purpose}\n")
        print("| " + " | ".join(hdr) + " |")
        print("|" + "---|" * len(hdr))
        table = []
        for arm in ARMS:
            if arm not in arms:
                continue
            r = arms[arm]
            ev, tr = r["eval"], r.get("train", {})
            lay = layers[arm]
            vals = [LABEL[arm], ", ".join(map(str, lay)) if isinstance(lay, list) else lay]
            for k, _ in cols:
                v = ev.get(k)
                vals.append("-" if v is None else (f"{v:.4f}" if k.startswith("kl") else pct(v)))
            vals += [f"{tr['final_loss']:.3f}" if tr else "-", f"{tr['trainable_params']:,}" if tr else "-",
                     r.get("dtype", "bfloat16")]
            print("| " + " | ".join(vals) + " |")
            table.append({"arm": arm, "label": LABEL[arm], "layers": lay, "dtype": r.get("dtype", "bfloat16"),
                          **{k: ev.get(k) for k, _ in cols}, **({"final_loss": tr["final_loss"],
                                                                 "trainable_params": tr["trainable_params"]} if tr else {})})
        crow = [c for (p, a), c in cost.items() if p == purpose]
        if crow:
            full = cost.get((purpose, "full"))
            print(f"\nClean cost benchmark, {purpose}: every arm alone on the same idle 4-core CPU, float32, "
                  f"{crow[0]['timed_steps']} timed steps after 3 warm-up steps.\n")
            print("| arm | backprop stops at layer | trainable weights | Adam state (MB) | s / step | vs all-layer | "
                  "one-off prefix cache (s) | est. full run (s) | peak RSS (MB) |")
            print("|---|---|---|---|---|---|---|---|---|")
            for arm in ARMS:
                c = cost.get((purpose, arm))
                if not c:
                    continue
                rel = f"{full['s_per_step'] / c['s_per_step']:.1f}x faster" if full and arm != "full" else "-"
                print(f"| {LABEL[arm]} | {c['lowest_trainable_layer']} | {c['trainable_params']:,} | {c['adam_state_mb']:.1f} | "
                      f"{c['s_per_step']:.2f} | {rel} | {c['prefix_cache_s']:.0f} | {c['est_total_s']:.0f} | {c['peak_rss_mb']:.0f} |")
        out[purpose] = {"picks": {k: picks.get(k) for k in ("grad_top4", "grad_rel_top4", "grad_top4_upper", "grad_bottom4",
                                                            "spectrum_top4", "rand4", "top_mlp_layers", "tool_seconds")},
                        "grad_share_by_layer": picks["grad_share_by_layer"], "arms": table,
                        "cost": [c for (p, a), c in cost.items() if p == purpose]}
    json.dump(out, open(os.path.join(H1, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
