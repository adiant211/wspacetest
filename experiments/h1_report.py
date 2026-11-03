"""Collect the H1 runs (results/h1/<purpose>_<arm>_s<seed>.json) into one table per purpose.

Usage: python experiments/h1_report.py   ->  prints markdown tables, writes results/h1/summary.json
"""
import glob
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H1 = os.path.join(REPO, "results", "h1")
ARMS = ["base", "full", "full_r1", "pick4", "pick4r", "pick4u", "spectrum4", "arb4", "rand4a", "rand4b", "anti4", "neur", "neur_rand"]
LABEL = {"base": "no fine-tune", "full": "LoRA, all 24 layers", "full_r1": "LoRA r=1, all 24 layers",
         "pick4": "LoRA, 4 tool-picked layers", "pick4u": "LoRA, 4 tool-picked upper layers", "pick4r": "LoRA, 4 tool-picked layers (relative)", "spectrum4": "LoRA, 4 Spectrum layers", "arb4": "LoRA, 4 arbitrary layers", "rand4a": "LoRA, 4 random layers (a)",
         "rand4b": "LoRA, 4 random layers (b)", "anti4": "LoRA, 4 anti-picked layers",
         "neur": "64+64 tool-picked neurons", "neur_rand": "64+64 random neurons"}


def main():
    rows = {}
    for f in sorted(glob.glob(os.path.join(H1, "*_s0.json"))):
        r = json.load(open(f))
        rows.setdefault(r["purpose"], {})[r["arm"]] = r
    out = {}
    for purpose, arms in rows.items():
        picks = json.load(open(os.path.join(H1, f"picks_{purpose}.json")))
        layers = {"pick4": picks["grad_top4"], "pick4u": picks.get("grad_top4_upper"), "pick4r": picks.get("grad_rel_top4"), "anti4": picks["grad_bottom4"], "spectrum4": picks["spectrum_top4"], "arb4": picks.get("arb4"),
                  "rand4a": picks["rand4"][0], "rand4b": picks["rand4"][1], "neur": picks["top_mlp_layers"],
                  "neur_rand": picks["top_mlp_layers"], "full": "0-23", "full_r1": "0-23", "base": "-"}
        if purpose == "add":
            cols = [("acc_5digit", "5-digit acc"), ("acc_6digit_ood", "6-digit acc (unseen length)")]
        else:
            cols = [("french_rate", "answers in French"), ("capitals_acc", "capitals correct")]
        cols += [("kl_wiki", "KL on Wikipedia"), ("mmlu", "MMLU-200")] + ([("kl_chat", "KL on chat")] if purpose == "add" else [])
        hdr = ["arm", "layers"] + [c[1] for c in cols] + ["trainable params", "train time (s)", "s / step", "peak RSS (MB)"]
        print(f"\n### {purpose}\n")
        print("| " + " | ".join(hdr) + " |")
        print("|" + "---|" * len(hdr))
        table = []
        for arm in ARMS:
            if arm not in arms:
                continue
            r = arms[arm]
            ev, tr = r["eval"], r.get("train", {})
            vals = [LABEL[arm], str(layers[arm]).replace(" ", "")]
            for k, _ in cols:
                v = ev.get(k)
                vals.append("-" if v is None else (f"{v:.4f}" if k.startswith("kl") else f"{v:.1%}" if k != "mmlu" else f"{v:.1%}"))
            vals += [f"{tr['trainable_params']:,}" if tr else "-", f"{tr['total_train_s']:.0f}" if tr else "-",
                     f"{tr['s_per_step']:.2f}" if tr else "-", f"{tr['peak_rss_mb']:.0f}" if tr else "-"]
            print("| " + " | ".join(vals) + " |")
            table.append({"arm": arm, "label": LABEL[arm], "layers": layers[arm], **{k: ev.get(k) for k, _ in cols},
                          **({k: tr[k] for k in ("trainable_params", "total_train_s", "s_per_step", "peak_rss_mb", "final_loss",
                                                 "prefix_cache_s", "lowest_trainable_layer")} if tr else {})})
        out[purpose] = {"picks": {k: picks[k] for k in ("grad_top4", "grad_bottom4", "spectrum_top4", "rand4", "top_mlp_layers",
                                                        "tool_seconds")},
                        "grad_share_by_layer": picks["grad_share_by_layer"], "arms": table}
    json.dump(out, open(os.path.join(H1, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
