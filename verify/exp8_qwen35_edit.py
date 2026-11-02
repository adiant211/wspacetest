"""Verify Experiment 8: Astra's 131,072-weight edit of Qwen3.5-0.8B that removes reasoning loops.

The blog evaluates with vLLM at up to 32,768 thinking tokens per answer; on a 4-core CPU this model decodes at
~28 tok/s (batch 8), so a full re-evaluation is out of reach. Instead we check the claim in four cheaper ways:

  1. rebuild   rebuild the edited tensors from the public weights (pinned revision) with the release's own
               apply_edit.py; every base and edited tensor is checked against the release's SHA-256 (run separately:
               see README; this script re-checks the hashes of the rebuilt tensors it loads).
  2. structure is the p50c edit really "subtract one direction from 64 output columns of the MLPs of layers 15 and
               19"? (rank of the change, recovered direction, how the direction's component is scaled)
  3. teacher   on the BASE model's own looping MATH-500 answers released with the blog (eval/unedited/.../
               math500_first_samples.jsonl.gz, never shown to Astra), compare per-token log-probs of the edited and
               base models on the text just before the loop starts (productive) and on the loop itself.
               Claim to check: the edit makes the loop much less likely, the productive text barely changes.
  4. entry     restart generation from the base model's own text 256 tokens before its loop begins, with the blog's
               sampling (temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5), once with each model, and
               measure how much of the continuation repeats text already in the context.

Usage: python verify/exp8_qwen35_edit.py [--n-teacher 24] [--n-entry 16] [--entry-tokens 384]
"""
import argparse
import glob
import gzip
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import FULL, OUT, logs  # noqa: E402

os.environ.setdefault("HF_HOME", "/home/user/hf")
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList  # noqa: E402

MODEL, REV = "Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17"
PREFIX = "<|im_start|>user\n"
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>assistant\n<think>\n"
EDIT_DIR = os.environ.get("EDIT_DIR", "/home/user/edits")


def bf16_bits(t: torch.Tensor) -> np.ndarray:
    return t.contiguous().view(torch.int16).numpy().view(np.uint16)


def load_edit(run):
    man = json.load(open(logs(f"exp8_math/runs/{run}/edit/manifest.json")))
    out = {}
    for t in man["tensors"]:
        arr = np.load(os.path.join(EDIT_DIR, run, t["name"] + ".npy"))
        w = torch.from_numpy(arr).to(torch.bfloat16)
        assert hashlib.sha256(np.ascontiguousarray(bf16_bits(w)).tobytes()).hexdigest() == t["edited_sha256"]
        out[t["name"].replace("model.language_model.", "model.")] = w
    return man, out


def structure(base, edit):
    rep = {}
    sd = dict(base.named_parameters())
    for name, w in edit.items():
        w0 = sd[name].detach().float()
        d = w.float() - w0
        cols = torch.nonzero(d.abs().sum(0) > 0).flatten()
        S = torch.linalg.svdvals(d)
        u = torch.linalg.svd(d[:, cols], full_matrices=False)[0][:, 0]
        before, after = u @ w0[:, cols], u @ w.float()[:, cols]
        resid = d[:, cols] - torch.outer(u, u @ d[:, cols])   # what is NOT along u
        rep[name] = {
            "changed_columns": int(len(cols)), "rows": int(w0.shape[0]), "weights_touched": int(len(cols) * w0.shape[0]),
            "sv_ratio_1_over_2": float(S[0] / S[1]), "frac_change_energy_along_u": float(1 - resid.norm() ** 2 / d[:, cols].norm() ** 2),
            "u_component_scale_after_over_before_median": float((after / before).median()),
            "relative_change_fro": float(d.norm() / w0.norm()),
        }
    return rep


def load_texts(run="unedited"):
    fs = sorted(glob.glob(os.path.join(FULL, f"exp8_math/eval/{run}/parts/math500*/*first_samples.jsonl.gz")))
    if not fs:
        raise FileNotFoundError("generation dumps are not vendored; run third_party/fetch_lm_understand_logs.sh")
    return [json.loads(l) for f in fs for l in gzip.open(f, "rt")]


def loop_onset(ids, n=12, win=256, thr=0.9):
    """First token index from which >= thr of the next `win` n-grams already occurred earlier in the text."""
    seen, rep = set(), np.zeros(max(len(ids) - n + 1, 0), dtype=bool)
    for i in range(len(rep)):
        g = tuple(ids[i:i + n])
        rep[i] = g in seen
        seen.add(g)
    cs = np.concatenate([[0], np.cumsum(rep)])
    for i in range(0, len(rep) - win):
        if (cs[i + win] - cs[i]) / win >= thr:
            return i
    return None


@torch.no_grad()
def token_logprobs(model, ids, start, end):
    """log p(ids[t] | ids[:t]) for t in [start, end), computed from one forward over ids[:end]."""
    x = torch.tensor([ids[:end]])
    h = model.model(x).last_hidden_state[0, start - 1:end - 1]
    lp = torch.log_softmax(model.lm_head(h).float(), -1)
    return lp.gather(1, torch.tensor(ids[start:end]).unsqueeze(1)).squeeze(1).numpy()


class PresencePenalty(LogitsProcessor):
    """vLLM-style presence penalty: subtract `p` once from every token already present in the generated text."""

    def __init__(self, p, gen_start):
        self.p, self.gen_start = p, gen_start

    def __call__(self, input_ids, scores):
        for i in range(input_ids.shape[0]):
            seen = torch.unique(input_ids[i, self.gen_start[i]:])
            scores[i, seen] -= self.p
        return scores


def copy_rate(context, cont, n=12):
    """Share of the continuation's n-grams that already occur in context + earlier continuation."""
    seen = {tuple(context[i:i + n]) for i in range(len(context) - n + 1)}
    hits, tot = 0, 0
    full = context + cont
    for i in range(len(context) - n + 1, len(full) - n + 1):
        g = tuple(full[i:i + n])
        hits += g in seen
        tot += 1
        seen.add(g)
    return hits / max(tot, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-teacher", type=int, default=24)
    ap.add_argument("--n-entry", type=int, default=16)
    ap.add_argument("--entry-tokens", type=int, default=384)
    ap.add_argument("--seg", type=int, default=768, help="tokens scored before and after the loop onset")
    ap.add_argument("--skip", default="")
    a = ap.parse_args()
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
    base = AutoModelForCausalLM.from_pretrained(MODEL, revision=REV, dtype=torch.bfloat16).eval()
    man, edit = load_edit("p50c")
    edited = AutoModelForCausalLM.from_pretrained(MODEL, revision=REV, dtype=torch.bfloat16).eval()
    sd = dict(edited.named_parameters())
    for k, w in edit.items():
        sd[k].data.copy_(w)
    n_params = sum(p.numel() for p in base.parameters())
    report = {"model": f"{MODEL}@{REV}", "text_model_params": n_params,
              "edit": {"run": "p50c", "tensors": list(edit), "weights_touched": 0}}

    # 2. structure
    st = structure(base, edit)
    report["edit"]["structure"] = st
    report["edit"]["weights_touched"] = sum(v["weights_touched"] for v in st.values())
    report["edit"]["share_of_text_model"] = report["edit"]["weights_touched"] / n_params
    print(json.dumps(report["edit"], indent=1), flush=True)

    from datasets import load_dataset
    probs = {r["unique_id"]: r["problem"] for r in load_dataset("HuggingFaceH4/MATH-500", split="test")}
    rows = load_texts("unedited")
    cands = []
    for r in rows:
        ids = tok(PREFIX + probs[r["id"]] + SUFFIX + r["text"], add_special_tokens=False)["input_ids"]
        plen = len(tok(PREFIX + probs[r["id"]] + SUFFIX, add_special_tokens=False)["input_ids"])
        on = loop_onset(ids[plen:])
        if on is not None and on > a.seg + 64 and plen + on + a.seg < 6000:
            cands.append({"id": r["id"], "ids": ids, "plen": plen, "onset": plen + on})
    cands.sort(key=lambda c: c["id"])
    report["teacher"] = {"base_samples": len(rows), "looping_with_onset_in_window": len(cands)}
    print(f"{len(rows)} base samples, {len(cands)} usable looping samples", flush=True)

    # 3. teacher-forced log-probs
    if "teacher" not in a.skip:
        recs = []
        for c in cands[:a.n_teacher]:
            t = time.time()
            s0, on, e = c["onset"] - a.seg, c["onset"], c["onset"] + a.seg
            lb = token_logprobs(base, c["ids"], s0, e)
            le = token_logprobs(edited, c["ids"], s0, e)
            recs.append({"id": c["id"], "onset_tok": on - c["plen"],
                         "productive_base": float(lb[:a.seg].mean()), "productive_edit": float(le[:a.seg].mean()),
                         "loop_base": float(lb[a.seg:].mean()), "loop_edit": float(le[a.seg:].mean())})
            r = recs[-1]
            print(f"  {c['id']:32s} onset@{r['onset_tok']:5d}  productive Δ={r['productive_edit']-r['productive_base']:+.4f}  "
                  f"loop Δ={r['loop_edit']-r['loop_base']:+.4f} nats/tok  [{time.time()-t:.0f}s]", flush=True)
        dp = np.array([r["productive_edit"] - r["productive_base"] for r in recs])
        dl = np.array([r["loop_edit"] - r["loop_base"] for r in recs])
        report["teacher"].update({
            "n": len(recs), "seg_tokens": a.seg, "per_sample": recs,
            "productive_delta_mean": float(dp.mean()), "productive_delta_se": float(dp.std(ddof=1) / np.sqrt(len(dp))),
            "loop_delta_mean": float(dl.mean()), "loop_delta_se": float(dl.std(ddof=1) / np.sqrt(len(dl))),
            "loop_more_suppressed_in": int((dl < dp).sum()),
        })
        print("teacher:", {k: v for k, v in report["teacher"].items() if k != "per_sample"}, flush=True)
        json.dump(report, open(os.path.join(OUT, "exp8_qwen35_edit.json"), "w"), indent=1)

    # 4. loop entry: continue from 256 tokens before the onset
    if "entry" not in a.skip:
        ents = cands[:a.n_entry]
        res = {"base": [], "edited": []}
        for name, model in (("base", base), ("edited", edited)):
            for b in range(0, len(ents), 8):
                chunk = ents[b:b + 8]
                ctxs = [c["ids"][:c["onset"] - 256] for c in chunk]
                T = max(len(x) for x in ctxs)
                ids = torch.full((len(ctxs), T), tok.pad_token_id or 248044, dtype=torch.long)
                att = torch.zeros_like(ids)
                for i, x in enumerate(ctxs):
                    ids[i, T - len(x):] = torch.tensor(x)
                    att[i, T - len(x):] = 1
                gen_start = [T - len(x) + c["plen"] for x, c in zip(ctxs, chunk)]
                t = time.time()
                torch.manual_seed(1234 + b)
                with torch.no_grad():
                    g = model.generate(input_ids=ids, attention_mask=att, max_new_tokens=a.entry_tokens, do_sample=True,
                                       temperature=0.6, top_p=0.95, top_k=20,
                                       logits_processor=LogitsProcessorList([PresencePenalty(1.5, gen_start)]))
                for i, (x, c) in enumerate(zip(ctxs, chunk)):
                    cont = g[i, T:].tolist()
                    res[name].append({"id": c["id"], "copy_rate": copy_rate(x[c["plen"]:], cont),
                                      "closed_think": "</think>" in tok.decode(cont)})
                print(f"  entry {name} batch {b // 8}: mean copy rate "
                      f"{np.mean([r['copy_rate'] for r in res[name][-len(chunk):]]):.3f} [{time.time()-t:.0f}s]", flush=True)
        cb = np.array([r["copy_rate"] for r in res["base"]])
        ce = np.array([r["copy_rate"] for r in res["edited"]])
        report["entry"] = {"n": len(cb), "new_tokens": a.entry_tokens, "per_sample": res,
                           "copy_rate_base_mean": float(cb.mean()), "copy_rate_edited_mean": float(ce.mean()),
                           "paired_diff_mean": float((ce - cb).mean()), "paired_diff_se": float((ce - cb).std(ddof=1) / np.sqrt(len(cb))),
                           "base_looping_share(copy>0.5)": float((cb > 0.5).mean()), "edited_looping_share(copy>0.5)": float((ce > 0.5).mean())}
        print("entry:", {k: v for k, v in report["entry"].items() if k != "per_sample"}, flush=True)
    json.dump(report, open(os.path.join(OUT, "exp8_qwen35_edit.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
