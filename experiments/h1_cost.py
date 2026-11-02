"""Clean cost benchmark for the H1 arms: each arm in its own process, run one after another on an otherwise idle
CPU (all 4 cores), training for a fixed number of steps on the same batches.

Reports per arm: trainable parameters, optimizer state, one-off frozen-prefix cache time, seconds per step (median of
the timed steps, after warm-up), estimated time for the sweep's full run, and peak resident memory of the process.
The accuracy sweep (h1_sweep.sh) shared the CPU with other jobs, so its own timings are not used.

Usage: python experiments/h1_cost.py [--steps 12]          (runs every arm, each in a subprocess)
       python experiments/h1_cost.py --one add pick4 --steps 12
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "h1")
ARMS = {"add": ["full", "full_r1", "pick4", "pick4r", "pick4u", "spectrum4", "rand4a", "rand4b", "anti4", "neur", "neur_rand"],
        "french": ["full", "pick4", "pick4r", "pick4u", "spectrum4", "rand4a", "anti4", "neur"]}


def one(purpose, arm, steps):
    import torch
    from experiments.h1_run import CFG, build
    from wsp.common import load_model, load_tokenizer, peak_rss_mb, seed_all
    from wsp.tasks import add_items, french_data
    from wsp.train import batches, embed, loss_on, lowest_trainable_layer, run_layers, tokenize, trainable
    seed_all(0)
    tok = load_tokenizer()
    picks = json.load(open(os.path.join(OUT, f"picks_{purpose}.json")))
    items = add_items(3000, seed=1) if purpose == "add" else french_data(None, tok)["train"]
    cfg = CFG[purpose]
    model, kind = build(load_model(), arm, picks)
    params = trainable(model)
    opt = torch.optim.AdamW(params, lr=cfg["lr"] if kind == "lora" else cfg["lr_neur"])
    data = tokenize(tok, items)
    start = lowest_trainable_layer(model)
    # one-off cost of the frozen-prefix cache, extrapolated from 256 examples to the whole training set
    t = time.perf_counter()
    sub = data[:256]
    with torch.no_grad():
        for ids, lab in batches(sub, 32, shuffle=False):
            if start > 0:
                run_layers(model, embed(model, ids), 0, start)
    cache_s = (time.perf_counter() - t) * len(data) / len(sub) if start > 0 else 0.0
    bl = list(batches(data, cfg["bs"], shuffle=True, seed=0))[:steps + 3]
    times = []
    model.train()
    for k, (ids, lab) in enumerate(bl):
        h0 = None
        if start > 0:
            with torch.no_grad():
                h0 = run_layers(model, embed(model, ids), 0, start)   # stands in for the cached tensor
        t = time.perf_counter()
        loss = loss_on(model, ids, lab, h0=h0, start=start)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if k >= 3:
            times.append(time.perf_counter() - t)
    s_step = statistics.median(times)
    n_par = sum(p.numel() for p in params)
    res = {"purpose": purpose, "arm": arm, "trainable_params": n_par, "adam_state_mb": n_par * 2 * 4 / 2 ** 20,
           "lowest_trainable_layer": start, "s_per_step": s_step, "prefix_cache_s": cache_s,
           "est_total_s": cache_s + s_step * cfg["steps"], "peak_rss_mb": peak_rss_mb(), "timed_steps": len(times),
           "threads": torch.get_num_threads()}
    print(json.dumps(res), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--one", nargs=2)
    a = ap.parse_args()
    if a.one:
        one(*a.one, a.steps)
        return
    rows = []
    for purpose, arms in ARMS.items():
        for arm in arms:
            p = subprocess.run([sys.executable, __file__, "--one", purpose, arm, "--steps", str(a.steps)],
                               capture_output=True, text=True)
            line = [l for l in p.stdout.splitlines() if l.startswith("{")]
            if not line:
                print(purpose, arm, "FAILED", p.stderr[-2000:], flush=True)
                continue
            r = json.loads(line[-1])
            rows.append(r)
            print(f"{purpose:6s} {arm:9s} params {r['trainable_params']:>9,}  from layer {r['lowest_trainable_layer']:2d}  "
                  f"{r['s_per_step']:.3f} s/step  cache {r['prefix_cache_s']:.0f}s  est. total {r['est_total_s']:.0f}s  "
                  f"peak RSS {r['peak_rss_mb']:.0f} MB", flush=True)
            json.dump(rows, open(os.path.join(OUT, "cost.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
