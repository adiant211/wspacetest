"""Verify Experiment 3 and test H2 on it: can an ordinary, deterministic weight-space tool (no LLM, no forward pass)
find the copying (induction) heads in the middle blocks of the three character-level LMs, as Astra did?

The tool is the textbook recipe of Elhage et al. (2021), "A Mathematical Framework for Transformer Circuits",
applied only to the released blocks (no embeddings, no unembedding, no positions):
  * OV copying score   sum(eig) / sum(|eig|) of the head's OV matrix  W_O^h W_V^h diag(ln1_w)  (384 x 384, rank 64):
                       positive real eigenvalues mean the head writes back what it reads.
  * K-composition      || W_QK^h2 W_OV^h1 ||_F / (||W_QK^h2||_F ||W_OV^h1||_F) for every head h1 in an earlier layer,
                       minus the same score for random matrices of equal shape and norm (baseline).
  * induction score    OV copying (clipped at 0) x best K-composition excess from an earlier head.
Ground truth is the release's eval/heads.json (measured by running the full networks, never available to Astra):
copying heads are those with induct > 0.2 (as in the post).

It also checks two pieces of evidence Astra gave for the language guesses:
  * "Code repeats names far more than prose, so many copying heads pointed to Python"
  * "Romaji's last block writes in fewer distinct directions" (we measure the effective rank of block 3's writes)

Usage: python verify/exp3_heads.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import OUT, logs  # noqa: E402

LANGS = {"python": "python_L4_H6_d384_s0", "romaji": "romaji_L4_H6_d384_s0", "english": "english_L4_H6_d384_s0"}
L, H, D, HD = 4, 6, 384, 64


def head_mats(W, l, h):
    """OV and QK of one head in residual coordinates, with LayerNorm folded in: the gain scales input dims and the
    mean subtraction is a projection P = I - 11^T/d on every read (and on every write, since the next LN removes it)."""
    g = W[f"ln1_w.{l}"]
    sl = slice(h * HD, (h + 1) * HD)
    cin = lambda a: a - a.mean(axis=1, keepdims=True)    # a @ P   (read side)
    cout = lambda a: a - a.mean(axis=0, keepdims=True)   # P @ a   (write side)
    q, k, v = (cin(W[f"W_{n}.{l}"][sl] * g[None, :]) for n in "qkv")
    o = cout(W[f"W_o.{l}"][:, sl])
    return o @ v, q.T @ k / np.sqrt(HD)


def ov_copy(ov):
    ev = np.linalg.eigvals(ov)
    return float(ev.real.sum() / np.abs(ev).sum())


def comp(a, b):
    return float(np.linalg.norm(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def match(qk, ov):
    """Signed K-composition: tr(QK_h2 @ OV_h1) / norms. Positive = h2's query looks for keys that carry, via h1,
    the same vector the query holds (a query token matching the token h1 moved into the key position)."""
    return float(np.trace(qk @ ov) / (np.linalg.norm(qk) * np.linalg.norm(ov)))


def baseline(rng, n=50):
    """K-composition of random rank-64 matrices with the same shapes (Elhage et al.'s baseline)."""
    vals = []
    for _ in range(n):
        a = rng.standard_normal((D, HD)) @ rng.standard_normal((HD, D))
        b = rng.standard_normal((D, HD)) @ rng.standard_normal((HD, D))
        vals.append(comp(a, b))
    return float(np.mean(vals))


def eff_rank(M):
    s = np.linalg.svd(M, compute_uv=False)
    p = s ** 2 / (s ** 2).sum()
    return float(np.exp(-(p * np.log(p + 1e-30)).sum()))


def main():
    rng = np.random.default_rng(0)
    base = baseline(rng)
    truth_all = json.load(open(logs("exp3_language/eval/heads.json")))
    scores_rel = json.load(open(logs("exp3_language/eval/scores.json")))
    rep = {"k_composition_random_baseline": base, "models": {}}
    for lang, key in LANGS.items():
        z = np.load(logs(f"exp3_language/networks/{lang}/blocks.npz"))
        W = {k: z[k].astype(np.float64) for k in z.files}
        mats = {(l, h): head_mats(W, l, h) for l in range(L) for h in range(H)}
        heads = {}
        for (l, h), (ov, qk) in mats.items():
            kc = {f"{l1}.{h1}": comp(qk, mats[(l1, h1)][0]) - base for l1 in range(l) for h1 in range(H)}
            mt = {f"{l1}.{h1}": match(qk, mats[(l1, h1)][0]) for l1 in range(l) for h1 in range(H)}
            best = max(mt, key=mt.get) if mt else None
            heads[f"{l}.{h}"] = {"ov_copy": ov_copy(ov), "kcomp_unsigned_best": max(kc.values()) if kc else 0.0,
                                 "match_best": mt[best] if best else 0.0, "match_from": best}
        for v in heads.values():
            v["induction_score"] = v["match_best"]
        truth = truth_all[key]
        true_copy = sorted(f"{k[1]}.{k[3]}" for k, v in truth.items() if v["induct"] > 0.2)
        ranked = sorted(heads, key=lambda k: -heads[k]["induction_score"])
        top = ranked[:len(true_copy)]
        # how well does the weight-only score rank the measured induction score?
        from scipy.stats import spearmanr
        xs = [heads[f"{k[1]}.{k[3]}"]["induction_score"] for k in truth]
        ys = [truth[k]["induct"] for k in truth]
        rho = float(spearmanr(xs, ys).statistic)
        # Astra's named copy heads (from the release's parsed final answers)
        astra = sorted({f"{h[1]}.{h[3]}" for r in scores_rel if r["corpus"] == lang for h in r.get("copy_named", [])})
        # evidence for the language guesses
        er_proj = [eff_rank(W[f"W_proj.{l}"]) for l in range(L)]
        er_o = [eff_rank(W[f"W_o.{l}"]) for l in range(L)]
        n_copy_tool = sum(heads[k]["induction_score"] > 0.5 * heads[ranked[0]]["induction_score"] for k in heads)
        rep["models"][lang] = {
            "true_copying_heads": true_copy, "tool_top_k": top, "tool_exact_match": sorted(top) == true_copy,
            "tool_ranking": ranked[:6], "spearman_tool_vs_measured_induct": rho,
            "astra_named": astra, "heads": heads,
            "effective_rank_W_proj_by_layer": er_proj, "effective_rank_W_o_by_layer": er_o,
            "tool_heads_above_half_of_max": n_copy_tool,
        }
        print(f"{lang:8s} truth {true_copy}  tool top-{len(true_copy)} {top}  match={sorted(top) == true_copy}  "
              f"rho={rho:+.2f}  ranking {[(k, round(heads[k]['induction_score'], 3), heads[k]['match_from']) for k in ranked[:5]]}  astra={astra} | eff.rank W_proj.3 {er_proj[3]:.0f}, W_o.3 {er_o[3]:.0f}", flush=True)
    json.dump(rep, open(os.path.join(OUT, "exp3_heads.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
