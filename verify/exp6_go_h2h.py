"""Verify Experiment 6's head-to-head: Astra's written 9x9 Go networks (budgets 0 and 50 forward passes) against the
three released same-size PPO networks (243 -> 512 x 6 ReLU -> 82, PPO for 20,000 iterations of 1,024 games).

Protocol from the release: 2,000 games per pairing, the written network Black in half of them, moves sampled from the
softmax of the logits (temperature 1) with illegal moves masked, no search; pgx "go_9x9" rules (area scoring, komi
7.5, a move repeating a whole-board position loses, at most 162 moves). Our seeds differ from the release's.

Usage: python verify/exp6_go_h2h.py [--games 2000]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import FULL, OUT, logs  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import pgx  # noqa: E402

NET_DIR = os.path.join(FULL, "exp6_go_from_scratch", "networks")


def load_net(name):
    d = os.path.join(NET_DIR, name)
    p = {f"L{k}.{t}": jnp.asarray(np.load(f"{d}/L{k}.{t}.npy")) for k in range(1, 7) for t in "Wb"}
    p["out.W"], p["out.b"] = jnp.asarray(np.load(f"{d}/out.W.npy")), jnp.asarray(np.load(f"{d}/out.b.npy"))
    return p


def forward(p, x):
    h = x
    for k in range(1, 7):
        h = jax.nn.relu(h @ p[f"L{k}.W"] + p[f"L{k}.b"])
    return h @ p["out.W"] + p["out.b"]


def features(obs):
    # pgx obs (B, 9, 9, 17) interleaves history: plane 0 = mover's stones now, 1 = opponent's stones now,
    # 2/3 = the same one move earlier, ...; plane 16 = 1 when White is to move
    x = jnp.stack([obs[..., 0], obs[..., 1], obs[..., 16]], axis=-1)       # (B, 9, 9, 3)
    return x.reshape(x.shape[0], 243).astype(jnp.float32)                  # index (row*9+col)*3+plane


def play(written, opp, games, seed):
    env = pgx.make("go_9x9")
    init, step = jax.jit(jax.vmap(env.init)), jax.jit(jax.vmap(env.step))
    key = jax.random.PRNGKey(seed)
    key, k0 = jax.random.split(key)
    st = init(jax.random.split(k0, games))
    written_black = jnp.arange(games) < games // 2
    black_id = st.current_player                                          # Black moves first
    total = jnp.zeros((games, 2))

    @jax.jit
    def act(st, key):
        x = features(st.observation)
        lw, lo = forward(written, x), forward(opp, x)
        white_to_move = st.observation[:, 0, 0, 16] > 0.5
        use_written = jnp.where(written_black, ~white_to_move, white_to_move)
        logits = jnp.where(use_written[:, None], lw, lo)
        logits = jnp.where(st.legal_action_mask, logits, -jnp.inf)
        return jax.random.categorical(key, logits, axis=-1)

    n = 0
    while not bool(st.terminated.all()):
        key, k1 = jax.random.split(key)
        st = step(st, act(st, k1))
        total = total + st.rewards
        n += 1
        if n > 400:
            raise RuntimeError("games did not end")
    written_id = jnp.where(written_black, black_id, 1 - black_id)
    r = np.asarray(jnp.take_along_axis(total, written_id[:, None], axis=1)[:, 0])
    wb = np.asarray(written_black)
    return {"games": games, "score": float((r > 0).mean()), "score_as_black": float((r[wb] > 0).mean()),
            "score_as_white": float((r[~wb] > 0).mean()), "draws_or_unfinished": int((r == 0).sum()), "plies": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=2000)
    a = ap.parse_args()
    rep = {}
    for b in ("budget_0", "budget_50"):
        w = load_net(b)
        for s in (0, 1, 2):
            o = load_net(f"ppo_same_size_lr3e-5_s{s}")
            t = time.time()
            r = play(w, o, a.games, seed=1000 + s + (50 if b == "budget_50" else 0))
            rel = json.load(open(logs(f"exp6_go_from_scratch/eval/{b}/h2h_ppo_big_lr3e-5_s{s}_iter20000.json")))
            se = np.sqrt(r["score"] * (1 - r["score"]) / r["games"])
            r.update({"reported_score": rel["score"], "z": (r["score"] - rel["score"]) / (np.sqrt(2) * se), "seconds": round(time.time() - t, 1)})
            rep[f"{b}_vs_ppo_s{s}"] = r
            print(f"{b} vs same-size PPO s{s}: ours {r['score']:.3f} (as black {r['score_as_black']:.3f}, white "
                  f"{r['score_as_white']:.3f})  reported {rel['score']:.3f}  z={r['z']:+.2f}  [{r['seconds']}s]", flush=True)
            json.dump(rep, open(os.path.join(OUT, "exp6_go_h2h.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
