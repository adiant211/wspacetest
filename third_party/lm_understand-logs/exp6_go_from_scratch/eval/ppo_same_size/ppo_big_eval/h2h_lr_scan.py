"""Head-to-head (2,000 games, seed 31337, the written network Black in half; ppo_ref.h2h) between written networks
and the two best tuned same-size PPO networks of the learning-rate scan (lr 1e-4: best hidden main; lr 3e-5: best on
the practice opponent "strong"). The scan networks are exported to <data>/gohand/ppo_big/lr<lr>_s0_iter20000.
  python h2h_lr_scan.py [<lr>:<seed> ...]     (default 1e-4:0 3e-5:0; 10-24: 3e-5:1 3e-5:2 added)"""
import json, os, sys
import numpy as np
sys.path.insert(0, "<repo>/scripts/gohand")
import ppo_ref as R
RUNS = "<data>/gohand/runs"
OUT = "<repo>/results/gohand/ppo_big_eval/h2h_lr_scan.json"
out = json.load(open(OUT)) if os.path.exists(OUT) else {}
for job in (sys.argv[1:] or ["1e-4:0", "3e-5:0"]):
    lr, seed = job.split(":")
    src = f"{R.TRAIN}/go_9x9_ppo_big_lr{lr}_s{seed}"
    d = os.path.join(R.REF, f"lr{lr}_s{seed}_iter20000")
    os.makedirs(os.path.join(d, "submission"), exist_ok=True)
    for k, v in R.gonet_of(np.load(os.path.join(src, "iter20000.npz"))).items():
        np.save(os.path.join(d, "submission", k + ".npy"), v.astype(np.float32))
    json.dump(R.ARCH, open(os.path.join(d, "arch.json"), "w"))
    for run in ("v1_b0", "v1_b50"):
        r = R.h2h(os.path.join(RUNS, run), d, 2000)
        out[f"{run}_vs_lr{lr}" + ("" if seed == "0" else f"_s{seed}")] = {k: r[k] for k in ("score", "score_as_black",
                                                                                         "score_as_white")}
json.dump(out, open(OUT, "w"), indent=1)
