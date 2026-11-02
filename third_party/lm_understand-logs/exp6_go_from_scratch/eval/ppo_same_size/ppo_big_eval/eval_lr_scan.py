"""Same-size PPO learning-rate scan (512 x 6 ReLU, seed 0, 20,000 iterations, lr 1e-4 / 3e-5 / 1e-5; trained on a
rented GPU node): the hidden evaluation of the written networks (same opponents, games and seeds as eval_ppo_big.py) plus
500 games against the practice opponent "strong" (seed 4242, as the written networks' failure analysis). The best lr
on "strong" (never the hidden test) is the one to repeat with seeds 1-2 (10-24: lr 3e-5; every
go_9x9_ppo_big_lr<lr>_s<seed> run with iter20000.npz is evaluated; done ones are kept).   python eval_lr_scan.py"""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, "<repo>/scripts/gohand")
import play as P
TRAIN = "<repo>/results/gomatch/train"
OUT = "<repo>/results/gohand/ppo_big_eval/eval_lr_scan.json"
out = json.load(open(OUT)) if os.path.exists(OUT) else {}
out.pop("best_on_strong", None)
for f in sorted(glob.glob(f"{TRAIN}/go_9x9_ppo_big_lr*_s*/iter20000.npz")):
    lr, seed = f.split("/")[-2][len("go_9x9_ppo_big_lr"):].rsplit("_s", 1)
    name = f"lr{lr}_s{seed}_iter20000"
    if name in out:
        continue
    z = np.load(f)
    hand = {f"L{k}.W": z[f"actor/dense{k}/kernel"] for k in range(1, 7)}
    hand.update({f"L{k}.b": z[f"actor/dense{k}/bias"] for k in range(1, 7)})
    hand["out.W"], hand["out.b"] = z["actor/logits/kernel"], z["actor/logits/bias"]
    res = {"strong_500": P.summary(P.play(hand, 6, P.PRACTICE["strong"], 500, seed=4242))["score"]}
    for j, (oname, opp) in enumerate(P.HIDDEN.items()):     # not "name": that is the run's key (bug 10-24 12:50)
        res[oname] = P.summary(P.play(hand, 6, opp, 2000, seed=777 + j))["score"]
    res["main"] = float(np.mean([res[k] for k in P.HIDDEN if k != "random"]))
    out[name] = res
    print(name, json.dumps({k: round(v, 3) for k, v in res.items()}), flush=True)
seed0 = [k for k in out if k.endswith("_s0_iter20000")]
if seed0:
    out["best_on_strong"] = max(seed0, key=lambda k: out[k]["strong_500"])
json.dump(out, open(OUT, "w"), indent=1)
