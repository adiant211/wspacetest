"""Turn results/math/scan.pt into neuron rankings and a summary (no model needed).

Scores per neuron (layer l, channel j), all from held-in data:
  d_en, d_tr       Cohen's d of the activation, GSM8K solution positions vs Wikipedia positions, per language
  imp_<c>          MathNeuro-style importance at neuron level: ||g2 * W_down[:, j]|| x RMS(activation on condition c)
  G_<c>            d(mean NLL on c)/d(gain_j): first-order effect of scaling neuron j's output (negative = scaling up
                   lowers the loss)

Writes results/math/neurons.json: layer profiles, EN/TR overlap of math neurons, and ranked lists used by the
screen (experiments/math_surgery.py).
"""
import json
import os

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "results", "math")


def load_scores(path=os.path.join(OUT, "scan.pt")):
    s = torch.load(path)
    w = s["w_down_eff_norm"].double().numpy()
    sc = {"w": w}
    for c in s["conds"]:
        m, q = s[c]["mean"].double().numpy(), s[c]["sq"].double().numpy()
        sc[f"mean_{c}"], sc[f"var_{c}"] = m, np.clip(q - m * m, 0, None)
        sc[f"imp_{c}"] = w * np.sqrt(q)
        sc[f"G_{c}"] = s[c]["grad"].double().numpy()
        sc[f"n_{c}"] = s[c]["n"]
    for lang in ("en", "tr"):
        a, b = f"math_{lang}", f"gen_{lang}"
        na, nb = sc[f"n_{a}"], sc[f"n_{b}"]
        pooled = np.sqrt((na * sc[f"var_{a}"] + nb * sc[f"var_{b}"]) / (na + nb)) + 1e-6
        sc[f"d_{lang}"] = (sc[f"mean_{a}"] - sc[f"mean_{b}"]) / pooled
    sc["d"] = (sc["d_en"] + sc["d_tr"]) / 2
    return sc, s


def topk_flat(score, k):
    """Indices (layer, neuron) of the k largest entries of a [L, I] array."""
    flat = np.argsort(-score, axis=None)[:k]
    return [tuple(int(v) for v in np.unravel_index(f, score.shape)) for f in flat]


def mathneuro_specific(sc, frac=0.05, conds_math=("math_en", "math_tr"), conds_gen=("gen_en", "gen_tr")):
    """Neuron-level MathNeuro: per layer, neurons in the top `frac` by importance on math text and NOT in the top
    `frac` by importance on general text (importance = ||write|| x RMS activation). Returns a boolean [L, I] mask."""
    L, I = sc["w"].shape
    k = max(1, int(round(frac * I)))
    im = sum(sc[f"imp_{c}"] for c in conds_math)
    ig = sum(sc[f"imp_{c}"] for c in conds_gen)
    mask = np.zeros((L, I), bool)
    for l in range(L):
        top_m = set(np.argsort(-im[l])[:k]); top_g = set(np.argsort(-ig[l])[:k])
        for j in top_m - top_g:
            mask[l, j] = True
    return mask


def main():
    sc, raw = load_scores()
    L, I = sc["w"].shape
    out = {"tokens": {c: sc[f"n_{c}"] for c in raw["conds"]}}
    # EN vs TR: are the math neurons shared across languages?
    r = float(np.corrcoef(sc["d_en"].ravel(), sc["d_tr"].ravel())[0, 1])
    over = {}
    for k in (100, 500, 1000, 2000):
        a, b = set(topk_flat(sc["d_en"], k)), set(topk_flat(sc["d_tr"], k))
        over[k] = {"shared": len(a & b), "chance": round(k * k / (L * I), 1)}
    out["en_tr"] = {"pearson_d": r, "top_k_overlap": over}
    # layer profiles of the top-1000 math neurons under each definition
    prof = {}
    for name, score in (("d", sc["d"]), ("d_en", sc["d_en"]), ("d_tr", sc["d_tr"]),
                        ("-G_calc", -sc["G_calc"]), ("-G_arith", -sc["G_arith"]),
                        ("-G_math", -(sc["G_math_en"] + sc["G_math_tr"]))):
        cnt = np.zeros(L, int)
        for l, _ in topk_flat(score, 1000):
            cnt[l] += 1
        prof[name] = cnt.tolist()
    mn = mathneuro_specific(sc)
    prof["mathneuro_5pct"] = mn.sum(1).tolist()
    out["layer_profile_top1000"] = prof
    # do activation-selective neurons also matter causally? (rank correlation of d with -G)
    from scipy.stats import spearmanr
    out["spearman_d_vs_-G_calc"] = float(spearmanr(sc["d"].ravel(), -sc["G_calc"].ravel()).statistic)
    out["spearman_d_vs_-G_arith"] = float(spearmanr(sc["d"].ravel(), -sc["G_arith"].ravel()).statistic)
    out["spearman_G_calc_vs_G_arith"] = float(spearmanr(sc["G_calc"].ravel(), sc["G_arith"].ravel()).statistic)
    out["spearman_G_math_vs_G_gen"] = float(spearmanr((sc["G_math_en"] + sc["G_math_tr"]).ravel(),
                                                      (sc["G_gen_en"] + sc["G_gen_tr"]).ravel()).statistic)
    out["top20_d"] = [{"layer": l, "neuron": j, "d_en": round(float(sc["d_en"][l, j]), 2),
                       "d_tr": round(float(sc["d_tr"][l, j]), 2)} for l, j in topk_flat(sc["d"], 20)]
    json.dump(out, open(os.path.join(OUT, "neurons.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "layer_profile_top1000"}, indent=1))
    for k, v in prof.items():
        print(f"{k:>15}: " + " ".join(f"{x:3d}" for x in v))


if __name__ == "__main__":
    main()
