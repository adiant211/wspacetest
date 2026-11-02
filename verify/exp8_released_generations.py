"""Experiment 8, from the release's own generations: how often do the base and the edited (p50c) Qwen3.5-0.8B fall
into a loop on MATH-500, and how many answers close their thinking?

A text "loops" from the first word position where >= 90% of the next 300 word 8-grams already occurred earlier in the
text. Uses eval/<run>/parts/math500_*/math500_first_samples.jsonl.gz (the first sample of each problem).

Usage: python verify/exp8_released_generations.py
"""
import glob
import gzip
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import FULL, OUT  # noqa: E402


def load(run):
    fs = sorted(glob.glob(os.path.join(FULL, f"exp8_math/eval/{run}/parts/math500*/*first_samples.jsonl.gz")))
    return {json.loads(l)["id"]: json.loads(l) for f in fs for l in gzip.open(f, "rt")}


def onset(text, n=8, win=300, thr=0.9):
    w = text.split()
    seen, rep = set(), []
    for i in range(len(w) - n + 1):
        g = tuple(w[i:i + n])
        rep.append(g in seen)
        seen.add(g)
    cs = [0]
    for x in rep:
        cs.append(cs[-1] + x)
    for i in range(max(0, len(rep) - win)):
        if (cs[i + win] - cs[i]) / win >= thr:
            return i
    return None


def main():
    rep = {}
    for run in ("unedited", "p50c", "p25c", "p10c"):
        d = load(run)
        if not d:
            continue
        res = [(onset(r["text"]), len(r["text"].split()), "</think>" in r["text"]) for r in d.values()]
        loops = sorted(x[0] for x in res if x[0] is not None)
        rep[run] = {"answers": len(res), "looping": len(loops), "closed_think": sum(x[2] for x in res),
                    "loop_onset_words_median": st.median(loops) if loops else None,
                    "loop_onset_words_quartiles": [loops[int(q * (len(loops) - 1))] for q in (0.25, 0.75)] if loops else None,
                    "words_median": st.median(x[1] for x in res)}
        print(run, rep[run], flush=True)
    json.dump(rep, open(os.path.join(OUT, "exp8_released_generations.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
