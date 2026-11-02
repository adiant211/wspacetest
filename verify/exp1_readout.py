"""Experiment 1 with ordinary tools: what can a 40-line weight-only readout see in the four PPO policies?

Astra named the game of each network from its raw weights (4 / 4, open answer). It cited evidence such as "for
Breakout, the agent's weights lie along the bottom row" and "which actions the output layer tells apart". This script
computes exactly that kind of evidence, without an LLM and without running the network:
  * agent map     for input channel 0 (the agent), the L2 norm over hidden units of W1's row for every grid cell,
                  relative to the median cell: where the policy has learned to look for itself
  * action usage  the spread of each action's output weights W3[:, a] (how much the policy can move that logit)
It then applies a simple, fixed rule (paddle row / paddle column / free movement) and checks the release's answer key.
This is a sanity check of the evidence Astra cited, not a blind open-ended classifier: the rule knows the four
genres it can choose from, which Astra was not told.

Usage: python verify/exp1_readout.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import OUT, logs  # noqa: E402

ACTIONS = ["no-op", "up", "down", "left", "right"]


def agent_map(W1, ch=0):
    m = np.linalg.norm(W1.reshape(10, 10, 5, -1)[:, :, ch, :], axis=-1)
    return m / np.median(m[1:9, 1:9])


def read(net):
    W1, W3 = net["actor/dense1/kernel"], net["actor/logits/kernel"]
    am = agent_map(W1)[1:9, 1:9]                       # playable 8 x 8
    hot = am > 2.0                                      # cells with clearly trained agent weights
    rows, cols = np.where(hot)
    use = {a: float(np.std(W3[:, i]) * np.sqrt(W3.shape[0])) for i, a in enumerate(ACTIONS)}
    horiz, vert = use["left"] + use["right"], use["up"] + use["down"]
    one_row = len(set(rows)) == 1 if len(rows) else False
    one_col = len(set(cols)) == 1 if len(cols) else False
    if one_row and horiz > vert:
        guess = "breakout (paddle confined to one row, moves left/right)"
    elif one_col and vert > horiz:
        c = cols[0] + 1
        guess = ("pong (paddle confined to an edge column, moves up/down)" if c in (1, 8)
                 else "freeway (agent confined to one inner column, moves up/down)")
    else:
        guess = "pacman (agent can be anywhere, uses all four moves)"
    return am, hot, use, guess


def ascii(am):
    chars = " .:-=+*#%@"
    v = np.clip(am / am.max(), 0, 1)
    return "\n".join("".join(chars[int(x * 9)] * 2 for x in row) for row in v)


def main():
    key = json.load(open(logs("exp1_which_game/eval/answer_key.json")))
    scores = {s["id"]: s for s in json.load(open(logs("exp1_which_game/eval/scores.json")))}
    rep = {}
    for nid, k in key.items():
        net = np.load(logs(f"exp1_which_game/networks/{nid}.npz"))
        am, hot, use, guess = read(net)
        n_params = sum(net[x].size for x in net.files if x.startswith("actor/"))
        rep[nid] = {"true_game": k["game"], "actor_params": int(n_params), "astra_guess": scores[nid]["guess"],
                    "astra_correct": scores[nid]["correct"], "tool_reading": guess,
                    "tool_correct": guess.startswith(k["game"]), "hot_cells": int(hot.sum()),
                    "action_usage": use, "agent_map_ascii": ascii(am)}
        print(f"== {nid} truth={k['game']:8s} astra={scores[nid]['guess']:8s} tool={guess}  params={n_params}")
        print("   action usage:", {a: round(v, 2) for a, v in use.items()})
        print("\n".join("   " + l for l in ascii(am).splitlines()))
    rep["tool_accuracy"] = sum(v["tool_correct"] for v in rep.values() if isinstance(v, dict)) / len(key)
    rep["astra_accuracy"] = sum(v["astra_correct"] for k2, v in rep.items() if isinstance(v, dict)) / len(key)
    print("tool", rep["tool_accuracy"], "astra", rep["astra_accuracy"])
    json.dump(rep, open(os.path.join(OUT, "exp1_readout.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
