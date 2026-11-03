"""Math and general-text probes for the hybrid LM (wsp/hybrid_cpu.py), all teacher-forced, so one forward pass scores
a whole example.

arithmetic   few-shot "Question: What is 23 plus 45?\nAnswer: 68\n" style problems in English, Turkish or bare symbols.
             Greedy exact match is read off a single forward pass over prompt + gold answer + "\n": greedy decoding
             would emit exactly the gold answer iff every gold answer token, and the newline after it, is the argmax.
             Also returns the answer's log-probability. The few-shot prefix is run once and its recurrent state
             (Gated DeltaNet states, conv windows, KV cache, n-gram context) is reused for every problem.
gsm          GSM8K solutions (English or Turkish), teacher-forced. Per token: NLL and whether it is the argmax, with
             masks for (a) every solution token, (b) "result" tokens: the first number after each "=" in the
             solution, i.e. the outcome of a calculation, and (c) the final answer after "####".
text         plain NLL over the first `maxlen` tokens of a document (Wikipedia, for collateral damage).
"""
from __future__ import annotations

import random
import re

import numpy as np
import torch
import torch.nn.functional as F

NUM_RE = re.compile(r"^[\s▁]*\$?-?[\d.,]*\d[\d.,]*%?$")

TEMPLATES = {
    "en": {"q": "Question: What is {a} {op} {b}?\nAnswer:", "+": "plus", "-": "minus", "*": "times"},
    "tr": {"q": "Soru: {a} {op} {b} kaç eder?\nCevap:", "+": "artı", "-": "eksi", "*": "çarpı"},
    "sym": {"q": "{a} {op} {b} =", "+": "+", "-": "-", "*": "*"},
}
DIGITS = {"1": (1, 9), "2": (10, 99), "3": (100, 999)}


def make_problem(rng, op, da, db):
    a, b = rng.randint(*DIGITS[da]), rng.randint(*DIGITS[db])
    if op == "-" and b > a:
        a, b = b, a
    ans = a + b if op == "+" else a - b if op == "-" else a * b
    return a, b, ans


def arith_set(kind, lang, n, seed, shots=4):
    """kind like "2+2" (2-digit plus 2-digit), "2-1", "2*1". Returns (prefix_text, [problems]). The few-shot examples
    use their own seed and never coincide with a test problem."""
    da, op, db = kind[0], kind[1], kind[2]
    T = TEMPLATES[lang]
    fmt = lambda a, b: T["q"].format(a=a, b=b, op=T[op])
    srng = random.Random(10_000 + seed)
    shots_ = [make_problem(srng, op, da, db) for _ in range(shots)]
    prefix = "".join(f"{fmt(a, b)} {c}\n\n" for a, b, c in shots_)
    seen = {(a, b) for a, b, _ in shots_}
    rng = random.Random(seed)
    probs = []
    tries = 0
    while len(probs) < n and tries < 50_000:     # 1-digit kinds have only 81 problems: stop when they run out
        tries += 1
        a, b, c = make_problem(rng, op, da, db)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        probs.append({"q": fmt(a, b), "answer": str(c), "a": a, "b": b, "op": op})
    return prefix, probs


def _expand_state(st, B):
    ex = lambda t: None if t is None else t.expand(B, *t.shape[1:])
    return {"conv": [ex(t) for t in st["conv"]], "S": [ex(t) for t in st["S"]], "k": [ex(t) for t in st["k"]],
            "v": [ex(t) for t in st["v"]], "ctx": ex(st["ctx"]), "pos": st["pos"]}


def _batches(items, bs):
    order = sorted(range(len(items)), key=lambda i: len(items[i]))
    for s in range(0, len(order), bs):
        yield order[s:s + bs]


@torch.no_grad()
def eval_arith(model, tok, prefix, probs, bs=32):
    """Per problem: greedy exact match (bool) and log P(answer tokens + newline)."""
    bos = tok.convert_tokens_to_ids("<bos>")
    E = lambda s: [bos] + tok(s, add_special_tokens=False)["input_ids"]
    # tokenize every example as one joint text (the tokenizer merges across "\n\nQuestion:"), and cache the state of
    # the longest token prefix that all examples share with the few-shot prefix alone
    pre = E(prefix)
    full, ans_start = [], []
    for p in probs:
        ctx = E(prefix + p["q"])
        f = E(prefix + p["q"] + " " + p["answer"] + "\n")
        assert f[:len(ctx)] == ctx, "answer boundary merged with the question"
        full.append(f)
        ans_start.append(len(ctx))
    K = len(pre) - 2
    assert all(f[:K] == pre[:K] for f in full)
    st = model.new_state()
    model(torch.tensor([pre[:K]]), state=st)
    seqs = [f[K:] for f in full]
    spans = [(a - K, len(f) - K) for a, f in zip(ans_start, full)]
    assert min(s for s, _ in spans) >= 1
    correct, logp = np.zeros(len(probs), bool), np.zeros(len(probs))
    for idx in _batches(seqs, bs):
        L = max(len(seqs[i]) for i in idx)
        ids = torch.tensor([seqs[i] + [0] * (L - len(seqs[i])) for i in idx])
        lg = model(ids, state=_expand_state(st, len(idx))).float()
        # the first suffix token is predicted from the prefix's last position, which this batch doesn't contain;
        # questions are at least 3 tokens, so every answer token is predicted from inside the batch
        lp = lg.log_softmax(-1)
        for r, i in enumerate(idx):
            s, e = spans[i]
            tgt = ids[r, s:e]
            pred = lp[r, s - 1:e - 1]
            correct[i] = bool((pred.argmax(-1) == tgt).all())
            logp[i] = float(pred.gather(1, tgt[:, None]).sum())
    return correct, logp


def gsm_text(item, lang):
    if lang == "tr":
        return f"Soru: {item['question']}\nCevap: {item['solution']}\n#### {item['answer']}"
    return f"Question: {item['question']}\nAnswer: {item['solution']}\n#### {item['answer']}"


def gsm_encode(tok, item, lang, maxlen=400):
    """ids and three masks over target positions t (token t predicted from t-1): solution, result, final."""
    bos = tok.convert_tokens_to_ids("<bos>")
    text = gsm_text(item, lang)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = [bos] + enc["input_ids"], [(0, 0)] + enc["offset_mapping"]
    ids, offs = ids[:maxlen], offs[:maxlen]
    head = text.index("\nAnswer: " if lang == "en" else "\nCevap: ") + len("\nAnswer: " if lang == "en" else "\nCevap: ")
    final_at = text.rindex("#### ")
    toks = [tok.decode([i]) for i in ids]
    # Metaspace tokens carry their leading space, so a token starting at the space before a word belongs to it
    sol = np.array([o[1] > head for o in offs])
    final = np.array([o[1] > final_at + 5 for o in offs])
    result = np.zeros(len(ids), bool)
    for t, s in enumerate(toks):
        if s.strip() == "=" and offs[t][0] >= head:
            for u in range(t + 1, min(t + 4, len(ids))):
                if NUM_RE.match(toks[u]) and re.search(r"\d", toks[u]):
                    result[u] = True
                    break
                if toks[u].strip() not in ("", "$", "▁"):
                    break
    return ids, {"solution": sol & ~final, "result": result, "final": final}


@torch.no_grad()
def eval_gsm(model, tok, items, lang, bs=8, maxlen=400):
    """Returns dict of masks -> (sum NLL, n tokens, n argmax-correct) and per-item rows."""
    enc = [gsm_encode(tok, it, lang, maxlen) for it in items]
    agg = {k: [0.0, 0, 0] for k in ("solution", "result", "final")}
    per = []
    for idx in _batches([e[0] for e in enc], bs):
        L = max(len(enc[i][0]) for i in idx)
        ids = torch.tensor([enc[i][0] + [0] * (L - len(enc[i][0])) for i in idx])
        lg = model(ids).float()
        nll = F.cross_entropy(lg[:, :-1].transpose(1, 2), ids[:, 1:], reduction="none")
        hit = lg[:, :-1].argmax(-1) == ids[:, 1:]
        for r, i in enumerate(idx):
            n = len(enc[i][0])
            row = {}
            for k, m in enc[i][1].items():
                mm = torch.tensor(m[1:n])
                a, b, c = float(nll[r, :n - 1][mm].sum()), int(mm.sum()), int(hit[r, :n - 1][mm].sum())
                agg[k][0] += a; agg[k][1] += b; agg[k][2] += c
                row[k] = (a, b, c)
            per.append((i, row))
    per.sort()
    return agg, [r for _, r in per]


@torch.no_grad()
def eval_text(model, tok, docs, maxlen=256, bs=8):
    """Mean NLL per document over its first maxlen tokens."""
    bos = tok.convert_tokens_to_ids("<bos>")
    seqs = [([bos] + tok(d["text"], add_special_tokens=False)["input_ids"])[:maxlen] for d in docs]
    out = np.zeros(len(docs))
    for idx in _batches(seqs, bs):
        L = max(len(seqs[i]) for i in idx)
        ids = torch.tensor([seqs[i] + [0] * (L - len(seqs[i])) for i in idx])
        lg = model(ids).float()
        nll = F.cross_entropy(lg[:, :-1].transpose(1, 2), ids[:, 1:], reduction="none")
        for r, i in enumerate(idx):
            out[i] = float(nll[r, :len(seqs[i]) - 1].mean())
    return out
