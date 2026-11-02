"""Verify Experiment 5: the MinAtar networks Astra wrote from scratch, in one reply, with no tools.

Two checks:
  1. Provenance: re-parse the last ```edits block of Astra's reply (runs/<game>/reply/answer.md), apply it to
     all-zero tensors, and compare with the released networks/<game>/hand.npz (is the scored network really just
     the text of the reply?).
  2. Score: re-run the official MinAtar 1.0.15 games with the blog's protocol (1,000 episodes, actions sampled from
     softmax(logits), sticky actions 0.1, episodes cut at 10,000 steps) and compare with eval/minatar_eval.json.

Usage: python verify/exp5_minatar.py [--episodes 1000] [--workers 4]
"""
import argparse
import ast
import json
import multiprocessing as mp
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import OUT, logs  # noqa: E402

# 6 network channels <- MinAtar's own channel names (from each prompt's "How the 6 channels are built" line)
CHANNELS = {
    "breakout": [["paddle"], [], ["brick"], ["ball"], ["trail"], []],
    "seaquest": [["sub_front", "sub_back"], ["enemy_fish", "enemy_sub", "enemy_bullet"],
                 ["enemy_fish", "enemy_sub", "diver"], ["friendly_bullet"], ["trail", "sub_back"],
                 ["oxygen_guage", "diver_guage"]],
    "asterix": [["player"], ["enemy"], ["gold"], [], ["trail"], []],
}
SHAPES = {"W1": (600, 32), "b1": (32,), "W2": (32, 32), "b2": (32,), "W3": (32, 6), "b3": (6,)}
NPZ_KEYS = {"W1": "actor/dense1/kernel", "b1": "actor/dense1/bias", "W2": "actor/dense2/kernel",
            "b2": "actor/dense2/bias", "W3": "actor/logits/kernel", "b3": "actor/logits/bias"}


# ------------------------------------------------------------------------------------- 1. provenance
def _num(node):
    return ast.literal_eval(node)


def _index(node):
    def one(n):
        if isinstance(n, ast.Slice):
            return slice(*(None if x is None else _num(x) for x in (n.lower, n.upper, n.step)))
        return _num(n)
    if isinstance(node, ast.Tuple):
        return tuple(one(n) for n in node.elts)
    return one(node)


def parse_edits(answer_md: str):
    blocks = re.findall(r"```edits\s*\n(.*?)```", answer_md, flags=re.S)
    if not blocks:
        raise ValueError("no edits block")
    text = blocks[-1]
    W = {k: np.zeros(s, dtype=np.float64) for k, s in SHAPES.items()}
    # join continuation lines: a statement ends when brackets are balanced
    stmts, cur, depth = [], "", 0
    for raw in text.splitlines():
        line = raw.split("#", 1)[0]
        if not line.strip() and not cur:
            continue
        cur += " " + line
        depth = cur.count("[") - cur.count("]")
        if depth <= 0 and cur.strip():
            stmts.append(cur.strip())
            cur = ""
    n = 0
    for s in stmts:
        tree = ast.parse(s).body[0]
        tgt = tree.targets[0] if isinstance(tree, ast.Assign) else tree.target
        name, idx = tgt.value.id, _index(tgt.slice)
        val = np.array(_num(tree.value), dtype=np.float64)
        if isinstance(tree, ast.Assign):
            W[name][idx] = val
        else:
            op = type(tree.op)
            if op is ast.Add:
                W[name][idx] += val
            elif op is ast.Sub:
                W[name][idx] -= val
            elif op is ast.Mult:
                W[name][idx] *= val
            else:
                raise ValueError(s)
        n += 1
    return {k: v.astype(np.float32) for k, v in W.items()}, n


# ------------------------------------------------------------------------------------------ 2. score
def obs6(env, mapping, idx):
    s = env.state()
    out = np.zeros((10, 10, 6), dtype=np.float32)
    for c, names in enumerate(mapping):
        for nm in names:
            out[:, :, c] = np.maximum(out[:, :, c], s[:, :, idx[nm]])
    return out.reshape(-1)  # (row*10+col)*6+channel


def run_chunk(args):
    game, W, seeds, max_steps = args
    from minatar import Environment
    rng = np.random.default_rng(seeds[0] + 12345)
    scores, lengths = [], []
    for sd in seeds:
        env = Environment(game, sticky_action_prob=0.1)
        env.seed(int(sd))
        env.reset()
        idx = env.env.channels
        G, T = 0.0, 0
        while T < max_steps:
            x = obs6(env, CHANNELS[game], idx)
            h1 = np.tanh(x @ W["W1"] + W["b1"])
            h2 = np.tanh(h1 @ W["W2"] + W["b2"])
            lg = h2 @ W["W3"] + W["b3"]
            p = np.exp(lg - lg.max())
            p /= p.sum()
            a = int(rng.choice(6, p=p))
            r, done = env.act(a)
            G += r
            T += 1
            if done:
                break
        scores.append(G)
        lengths.append(T)
    return scores, lengths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=10000)
    ap.add_argument("--games", default="breakout,seaquest,asterix")
    a = ap.parse_args()
    reported = json.load(open(logs("exp5_write_from_scratch/eval/minatar_eval.json")))
    report = {}
    for game in a.games.split(","):
        # 1. provenance
        ans = open(logs(f"exp5_write_from_scratch/runs/{game}/reply/answer.md")).read()
        W, n_stmt = parse_edits(ans)
        rel = np.load(logs(f"exp5_write_from_scratch/networks/{game}/hand.npz"))
        maxdiff = max(float(np.abs(W[k] - rel[NPZ_KEYS[k]]).max()) for k in SHAPES)
        nonzero = int(sum((W[k] != 0).sum() for k in SHAPES))
        # 2. score
        t = time.time()
        seeds = np.arange(a.episodes)
        chunks = [(game, W, seeds[i::a.workers * 4], a.max_steps) for i in range(a.workers * 4)]
        with mp.Pool(a.workers) as pool:
            res = pool.map(run_chunk, chunks)
        sc = np.array([s for r in res for s in r[0]])
        ln = np.array([s for r in res for s in r[1]])
        rep = reported[game]["hand"]
        report[game] = {
            "edits_statements_parsed": n_stmt,
            "rebuilt_from_reply_max_abs_diff_vs_released": maxdiff,
            "nonzero_weights": nonzero,
            "ours": {"mean": float(sc.mean()), "se": float(sc.std(ddof=1) / np.sqrt(len(sc))), "episodes": len(sc),
                     "max": float(sc.max()), "mean_length": float(ln.mean()), "cut_at_max_steps": int((ln >= a.max_steps).sum())},
            "reported": rep,
            "ppo_best_reported": {k: v["mean"] for k, v in reported[game].items() if k.startswith("ppo")},
            "seconds": round(time.time() - t, 1),
        }
        o = report[game]["ours"]
        z = (o["mean"] - rep["mean"]) / np.hypot(o["se"], rep["se"])
        report[game]["z_vs_reported"] = float(z)
        print(f"{game:9s} rebuilt-from-text maxdiff={maxdiff:.2e} nonzero={nonzero} | ours {o['mean']:.1f} ± {o['se']:.1f} "
              f"(len {o['mean_length']:.0f}) vs reported {rep['mean']:.1f} ± {rep['se']:.1f}  z={z:+.2f}  "
              f"[{report[game]['seconds']}s]", flush=True)
    json.dump(report, open(os.path.join(OUT, "exp5_minatar.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
