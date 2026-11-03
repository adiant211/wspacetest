"""Build the text sets for the math-neuron study on Ba2han/step-17k (Turkish + English).

Sources (fetched once into /home/user/data_raw, outside the repo):
  * English GSM8K            openai/gsm8k (main) train / test
  * Turkish GSM8K            Yigit-Karaman/GSM8K-TR-Gold train / test (a human-checked translation)
  * general text             random Wikipedia articles, wikimedia/wikipedia 20231101.en / 20231101.tr
                             (via the datasets-server rows API, seeded offsets)

Splits: neurons are found and edits are screened on GSM8K *train* problems and on one half of the Wikipedia articles.
Every reported number uses GSM8K *test* problems and the other half of the articles, which the search never sees.
Calculator annotations (<<48/2=24>>) are stripped so the solutions read as plain text.

Writes results/math/data/{gsm_en,gsm_tr,wiki_en,wiki_tr}.json, each {"scan": [...], "test": [...]}.
"""
import json
import os
import random
import re
import time
import urllib.parse
import urllib.request

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = "/home/user/data_raw"
OUT = os.path.join(REPO, "results", "math", "data")
ROWS = "https://datasets-server.huggingface.co/rows?"


def clean_solution(a):
    a = re.sub(r"<<[^>]*>>", "", a)
    body, _, final = a.rpartition("####")
    return body.strip(), final.strip().replace(",", "")


def gsm(lang, split, n, seed):
    df = pd.read_parquet(os.path.join(RAW, f"gsm8k_{lang}", f"{split}.parquet"))
    idx = list(range(len(df)))
    random.Random(seed).shuffle(idx)
    out = []
    for i in idx:
        body, final = clean_solution(df.answer[i])
        if not re.fullmatch(r"-?\d+(\.\d+)?", final):
            continue
        out.append({"i": int(i), "question": df.question[i].strip(), "solution": body, "answer": final})
        if len(out) == n:
            break
    return out


def wiki(lang, n, seed, min_chars=2500, max_chars=3000):
    cache = os.path.join(RAW, "wiki", f"{lang}.json")
    if os.path.exists(cache):
        return json.load(open(cache))
    q = {"dataset": "wikimedia/wikipedia", "config": f"20231101.{lang}", "split": "train"}
    total = json.load(urllib.request.urlopen(ROWS + urllib.parse.urlencode({**q, "offset": 0, "length": 1})))["num_rows_total"]
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        off = rng.randrange(0, total - 100)
        for attempt in range(5):
            try:
                d = json.load(urllib.request.urlopen(ROWS + urllib.parse.urlencode({**q, "offset": off, "length": 20}), timeout=60))
                break
            except Exception as e:  # noqa: BLE001
                print("  retry", e)
                time.sleep(2 + 2 * attempt)
        else:
            continue
        for r in d["rows"]:
            t = r["row"]["text"].strip()
            if len(t) >= min_chars and len(out) < n:
                out.append({"title": r["row"]["title"], "text": t[:max_chars]})
        print(f"  wiki {lang}: {len(out)}/{n}", flush=True)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    json.dump(out, open(cache, "w"), ensure_ascii=False)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    for lang in ("en", "tr"):
        d = {"scan": gsm(lang, "train", 400, seed=1), "test": gsm(lang, "test", 400, seed=2)}
        json.dump(d, open(os.path.join(OUT, f"gsm_{lang}.json"), "w"), ensure_ascii=False, indent=0)
        print(f"gsm_{lang}: scan {len(d['scan'])}, test {len(d['test'])}")
        w = wiki(lang, 400, seed=3 if lang == "en" else 4)
        d = {"scan": w[:200], "test": w[200:]}
        json.dump(d, open(os.path.join(OUT, f"wiki_{lang}.json"), "w"), ensure_ascii=False, indent=0)
        print(f"wiki_{lang}: scan {len(d['scan'])}, test {len(d['test'])}")


if __name__ == "__main__":
    main()
