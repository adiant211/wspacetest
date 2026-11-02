"""Hidden evaluation of the same-size PPO networks (512 x 6 ReLU, gomatch train.py --hidden 512 --depth 6 --act relu),
with the same opponents, games and sampling as the written networks (scripts/gohand/evaluate.py eval)."""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, "<repo>/scripts/gohand")
import play as P
out = {}
for s in (0, 1, 2):
    d = f"<repo>/results/gomatch/train/go_9x9_ppo_big_s{s}"
    it = max(int(os.path.basename(f)[4:-4]) for f in glob.glob(d + "/iter*.npz"))
    z = np.load(f"{d}/iter{it}.npz")
    hand = {f"L{k}.W": z[f"actor/dense{k}/kernel"] for k in range(1, 7)}
    hand.update({f"L{k}.b": z[f"actor/dense{k}/bias"] for k in range(1, 7)})
    hand["out.W"], hand["out.b"] = z["actor/logits/kernel"], z["actor/logits/bias"]
    res = {}
    for j, (name, opp) in enumerate(P.HIDDEN.items()):
        res[name] = P.summary(P.play(hand, 6, opp, 2000, seed=777 + j))["score"]
    res["main"] = float(np.mean([res[k] for k in P.HIDDEN if k != "random"]))
    out[f"s{s}_iter{it}"] = res
    print(f"s{s} iter{it}", json.dumps({k: round(v, 3) for k, v in res.items()}), flush=True)
json.dump(out, open("<repo>/results/gohand/ppo_big_eval/eval.json", "w"), indent=1)
