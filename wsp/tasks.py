"""Purposes ("specific purposes" for H1) and retention probes.

Purposes
  add     5-digit + 5-digit addition, answer only (a "math" capability; base model: ~28% exact)
  french  answer English questions in French by default (an output-style behaviour), built by
          context distillation from the model itself, so no external French data is needed

Retention probes (collateral damage of a fine-tune)
  kl_wiki   mean per-token KL(base || tuned) on raw English Wikipedia text (no chat template)
  kl_chat   mean per-token KL(base || tuned) on generic chat prompts with the base model's own replies
  mmlu      0-shot letter accuracy on a fixed random 200 MMLU test questions
  capitals  "What is the capital of X?" for 48 countries whose capital is spelled the same in French
"""
from __future__ import annotations

import json
import os
import random
import re
import unicodedata

import torch
import torch.nn.functional as F

from .common import RESULTS, chat, generate, load_model

DATA_DIR = os.path.join(RESULTS, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ----------------------------------------------------------------------------------------------- add
ADD_Q = "What is {a} + {b}? Reply with only the final number."


def add_items(n: int, seed: int, digits: int = 5):
    rng = random.Random(seed)
    lo, hi = 10 ** (digits - 1), 10 ** digits - 1
    out = []
    for _ in range(n):
        a, b = rng.randint(lo, hi), rng.randint(lo, hi)
        out.append({"user": ADD_Q.format(a=a, b=b), "answer": str(a + b)})
    return out


def add_correct(pred: str, gold: str) -> bool:
    nums = re.findall(r"-?\d[\d,]*", pred)
    return bool(nums) and nums[0].replace(",", "") == gold


# -------------------------------------------------------------------------------------------- french
# The 0.5B model ignores a system-prompt instruction to answer in French (7% compliance) but follows the same
# instruction at the end of the user turn (100% on 32 test prompts), so the teacher gets it there.
FR_SUFFIX = "\nAnswer in French only."
FR_WORDS = set("""le la les un une des du de et est sont pour dans que qui pas avec sur au aux ce cette ces il elle
ils elles nous vous je tu mon ma mes son sa ses leur leurs votre vos notre nos plus mais ou donc car être avoir fait
peut très aussi comme tout tous bien sans entre par en c'est d'un d'une l'on qu'il n'est j'ai""".split())
EN_WORDS = set("""the a an and is are for in that which not with on at this these it he she they we you i my his her
their your our more but or so because be have has does can very also like all well without between by of to was were
it's there what""".split())


def _words(s: str):
    return re.findall(r"[a-zàâäçéèêëîïôöùûüÿœ']+", s.lower())


def french_score(s: str) -> float:
    w = _words(s)
    fr = sum(x in FR_WORDS for x in w)
    en = sum(x in EN_WORDS for x in w)
    return fr / (fr + en) if fr + en else 0.0


def is_french(s: str) -> bool:
    return french_score(s) > 0.5


def _alpaca_questions():
    from datasets import load_dataset
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    qs = [r["instruction"].strip() for r in ds if not r["input"].strip()]
    bad = ("code", "program", "python", "sql", "html", "translate", "french", "spanish", "german", "rewrite",
           "edit", "sentence", "poem", "haiku", "word", "list", "table", "json", "regex", "function")
    qs = [q for q in qs if 15 < len(q) < 90 and q.endswith("?") and not any(b in q.lower() for b in bad)]
    random.Random(0).shuffle(qs)
    return qs


def french_data(model=None, tok=None, n_train=480, n_eval=96, max_new_tokens=64):
    """Context distillation: the teacher is the same model with FR_SUFFIX appended to the question; the student is
    trained on (question without the instruction) -> (the teacher's French answer)."""
    path = os.path.join(DATA_DIR, "french.json")
    if os.path.exists(path):
        return json.load(open(path))
    qs = _alpaca_questions()
    train_q, eval_q = qs[: int(n_train * 1.4)], qs[int(n_train * 1.4): int(n_train * 1.4) + n_eval]
    outs = generate(model, tok, [chat(tok, q + FR_SUFFIX) for q in train_q], max_new_tokens=max_new_tokens)
    train = [{"user": q, "answer": o.strip()} for q, o in zip(train_q, outs) if is_french(o) and len(o) > 20]
    data = {"train": train[:n_train], "eval": [{"user": q} for q in eval_q],
            "teacher_french_rate": sum(is_french(o) for o in outs) / len(outs)}
    json.dump(data, open(path, "w"), ensure_ascii=False, indent=1)
    return data


CAPITALS = {
    "France": "Paris", "Germany": "Berlin", "Spain": "Madrid", "Italy": "Rome", "Japan": "Tokyo",
    "Canada": "Ottawa", "Australia": "Canberra", "Norway": "Oslo", "Ireland": "Dublin", "Kenya": "Nairobi",
    "Peru": "Lima", "Chile": "Santiago", "Venezuela": "Caracas", "Argentina": "Buenos Aires",
    "Thailand": "Bangkok", "Vietnam": "Hanoi", "Indonesia": "Jakarta", "Finland": "Helsinki",
    "Sweden": "Stockholm", "Czech Republic": "Prague", "Hungary": "Budapest", "Turkey": "Ankara",
    "Pakistan": "Islamabad", "India": "New Delhi", "Jamaica": "Kingston", "Uruguay": "Montevideo",
    "Ecuador": "Quito", "Estonia": "Tallinn", "Latvia": "Riga", "Lithuania": "Vilnius", "Belarus": "Minsk",
    "Serbia": "Belgrade", "Croatia": "Zagreb", "Slovenia": "Ljubljana", "Slovakia": "Bratislava",
    "the Netherlands": "Amsterdam", "Qatar": "Doha", "Jordan": "Amman", "Tunisia": "Tunis", "Morocco": "Rabat",
    "Senegal": "Dakar", "Ghana": "Accra", "Nigeria": "Abuja", "Uganda": "Kampala", "Rwanda": "Kigali",
    "Zambia": "Lusaka", "Zimbabwe": "Harare", "New Zealand": "Wellington",
}


def _strip(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s.lower()) if unicodedata.category(c) != "Mn")


def capitals_eval(model, tok):
    items = list(CAPITALS.items())
    outs = generate(model, tok, [chat(tok, f"What is the capital of {c}?") for c, _ in items], max_new_tokens=24)
    return sum(_strip(cap) in _strip(o) for (_, cap), o in zip(items, outs)) / len(items), outs


# ------------------------------------------------------------------------------------------ retention
def wiki_chunks(tok, n=24, T=128):
    path = os.path.join(DATA_DIR, "wiki_chunks.json")
    if os.path.exists(path):
        return json.load(open(path))
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = " ".join(r["text"].strip() for r in ds if len(r["text"].strip()) > 200)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    chunks = [ids[i * T:(i + 1) * T] for i in range(n)]
    json.dump(chunks, open(path, "w"))
    return chunks


def chat_kl_set(model, tok, n=32, max_new_tokens=48):
    """Generic chat prompts (held out from everything else) + the base model's own greedy replies."""
    path = os.path.join(DATA_DIR, "chat_kl.json")
    if os.path.exists(path):
        return json.load(open(path))
    qs = _alpaca_questions()[-n:]
    outs = generate(model, tok, [chat(tok, q) for q in qs], max_new_tokens=max_new_tokens)
    data = [{"user": q, "answer": o} for q, o in zip(qs, outs)]
    json.dump(data, open(path, "w"), ensure_ascii=False, indent=1)
    return data


def _pad(chunk):
    T = max(len(x) for x in chunk)
    ids = torch.full((len(chunk), T), 151643, dtype=torch.long)
    for i, x in enumerate(chunk):
        ids[i, :len(x)] = torch.tensor(x)
    return ids


@torch.no_grad()
def kl_to_base(model, base, ids_list, masks, bs=8):
    """Mean over scored positions of KL(base || model), computed batch by batch (exact, full vocabulary).
    Right padding + causal attention, so pads never influence real positions."""
    tot, cnt = 0.0, 0
    for b in range(0, len(ids_list), bs):
        chunk, mchunk = ids_list[b:b + bs], masks[b:b + bs]
        ids = _pad(chunk)
        la = F.log_softmax(base(input_ids=ids).logits.float(), -1)
        lb = F.log_softmax(model(input_ids=ids).logits.float(), -1)
        kl = (la.exp() * (la - lb)).sum(-1)
        for i, m in enumerate(mchunk):
            k = kl[i, :len(m)][torch.tensor(m, dtype=torch.bool)]
            tot += k.sum().item()
            cnt += k.numel()
    return tot / max(cnt, 1)


def chat_ids(tok, user, answer):
    p = tok(chat(tok, user), add_special_tokens=False)["input_ids"]
    a = tok(answer + "<|im_end|>", add_special_tokens=False)["input_ids"]
    return p, a


def mmlu_items(n=200):
    path = os.path.join(DATA_DIR, "mmlu200.json")
    if os.path.exists(path):
        return json.load(open(path))
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test")
    idx = random.Random(0).sample(range(len(ds)), n)
    items = [{"question": ds[i]["question"], "choices": ds[i]["choices"], "answer": ds[i]["answer"],
              "subject": ds[i]["subject"]} for i in idx]
    json.dump(items, open(path, "w"), indent=1)
    return items


@torch.no_grad()
def mmlu_eval(model, tok, items, bs=8):
    letters = ["A", "B", "C", "D"]
    letter_ids = [tok(x, add_special_tokens=False)["input_ids"][0] for x in letters]
    prompts = []
    for it in items:
        q = it["question"] + "\n" + "\n".join(f"{l}. {c}" for l, c in zip(letters, it["choices"]))
        prompts.append(chat(tok, q + "\nAnswer with the letter of the correct option only."))
    correct = 0
    tok.padding_side = "left"
    for b in range(0, len(prompts), bs):
        enc = tok(prompts[b:b + bs], return_tensors="pt", padding=True, add_special_tokens=False)
        logits = model(**enc).logits[:, -1, :].float()
        pred = logits[:, letter_ids].argmax(-1)
        correct += sum(int(p) == it["answer"] for p, it in zip(pred.tolist(), items[b:b + bs]))
    return correct / len(items)


class Retention:
    """Scores a tuned model against the (kept-in-memory) base model."""

    def __init__(self, tok, base_model=None, kinds=("kl_wiki", "kl_chat", "mmlu")):
        self.tok, self.kinds = tok, kinds
        self.base = base_model or load_model()
        self.wiki = wiki_chunks(tok)
        self.wiki_masks = [[True] * len(x) for x in self.wiki]
        if "kl_chat" in kinds:
            cs = chat_kl_set(self.base, tok)
            self.chat_ids, self.chat_masks = [], []
            for it in cs:
                p, a = chat_ids(tok, it["user"], it["answer"])
                self.chat_ids.append(p + a)
                # score the positions whose next token is a reply token
                self.chat_masks.append([False] * (len(p) - 1) + [True] * len(a) + [False])
        self.mmlu = mmlu_items()
        self._base_mmlu = None

    @property
    def base_mmlu(self):
        if self._base_mmlu is None:
            self._base_mmlu = mmlu_eval(self.base, self.tok, self.mmlu)
        return self._base_mmlu

    def __call__(self, model):
        r = {"kl_wiki": kl_to_base(model, self.base, self.wiki, self.wiki_masks)}
        if "kl_chat" in self.kinds:
            r["kl_chat"] = kl_to_base(model, self.base, self.chat_ids, self.chat_masks)
        if "mmlu" in self.kinds:
            r["mmlu"] = mmlu_eval(model, self.tok, self.mmlu)
        return r
