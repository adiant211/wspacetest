"""Verify Experiment 4's footprint and provenance: is the scored car policy exactly Astra's 51 edit statements applied to
the network as it was shown (rounded), and does the edit really change 6,073 of 171,664 weights (3.5%)?

The episode results themselves (0 / 256 -> 1,024 / 1,024 successes) need the Kinetix JAX physics simulator and are not
re-run here.

Usage: python verify/exp4_footprint.py
"""
import ast
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import OUT, logs  # noqa: E402

NAMES = {"embed": "params/Dense_0", "actor_head": "params/GeneralActorCriticRNN_0/Dense_0",
         **{f"actor{i}": f"params/GeneralActorCriticRNN_0/DenseAndActivation_{i}/Dense_0" for i in range(5)}}


def shown_network(prompt):
    """The weights exactly as written in the prompt: integer x scale per tensor."""
    orig = np.load(logs("exp4_fix_car/networks/ppo_final_seed0.npz"))
    scales = dict(re.findall(r"^(\S+\.[Wb]) \([^)]*\), scale (\S+):", prompt, flags=re.M))
    net = {}
    for short, long in NAMES.items():
        for part, key in (("W", "kernel"), ("b", "bias")):
            sc = float(scales[f"{short}.{part}"])
            net[f"{short}.{part}"] = np.round(orig[f"{long}/{key}"].astype(np.float64) / sc) * sc
    return net, scales


def _lit(n):
    return ast.literal_eval(n)


def _idx(node):
    def one(n):
        if isinstance(n, ast.Slice):
            return slice(*(None if x is None else _lit(x) for x in (n.lower, n.upper, n.step)))
        return _lit(n)
    return tuple(one(n) for n in node.elts) if isinstance(node, ast.Tuple) else one(node)


def apply_edits(net, answer):
    text = re.findall(r"```edits\s*\n(.*?)```", answer, flags=re.S)[-1]
    stmts, cur = [], ""
    for raw in text.splitlines():
        line = raw.split("#", 1)[0]
        if not line.strip() and not cur:
            continue
        cur += " " + line
        if cur.count("[") - cur.count("]") <= 0 and cur.strip():
            stmts.append(cur.strip())
            cur = ""
    for s in stmts:
        t = ast.parse(s).body[0]
        tgt = t.targets[0] if isinstance(t, ast.Assign) else t.target
        name = f"{tgt.value.value.id}.{tgt.value.attr}"
        idx, val = _idx(tgt.slice), np.array(_lit(t.value), dtype=np.float64)
        if isinstance(t, ast.Assign):
            net[name][idx] = val
        elif isinstance(t.op, ast.Add):
            net[name][idx] += val
        elif isinstance(t.op, ast.Sub):
            net[name][idx] -= val
        else:
            net[name][idx] *= val
    return net, len(stmts)


def main():
    prompt = open(logs("exp4_fix_car/runs/oneshot_repair/reply/prompt.txt")).read()
    answer = open(logs("exp4_fix_car/runs/oneshot_repair/reply/answer.md")).read()
    shown, scales = shown_network(prompt)
    edited, n = apply_edits({k: v.copy() for k, v in shown.items()}, answer)
    released = np.load(logs("exp4_fix_car/networks/edited_actor.npz"))
    maxdiff = max(float(np.abs(edited[k] - released[k].astype(np.float64)).max()) for k in edited)
    changed = {k: int((~np.isclose(released[k].astype(np.float64), shown[k], rtol=0, atol=1e-9)).sum()) for k in shown}
    total = sum(v.size for v in shown.values())
    rep = {"statements": n, "rebuilt_vs_released_max_abs_diff": maxdiff, "changed_entries": sum(changed.values()),
           "entries": total, "share_changed": sum(changed.values()) / total, "per_tensor_changed": changed,
           "reported": json.load(open(logs("exp4_fix_car/eval/oneshot_result.json")))["changes"]["changed"]}
    print(json.dumps({k: v for k, v in rep.items() if k != "per_tensor_changed"}, indent=1))
    json.dump(rep, open(os.path.join(OUT, "exp4_footprint.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
