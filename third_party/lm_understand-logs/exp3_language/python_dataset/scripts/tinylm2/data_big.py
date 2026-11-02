"""Large character corpora for the second tiny-LM experiment (as large as possible, about 0.5-1 billion
characters per corpus; character level). The same processing as scripts/tinylm/data.py, only more text:
  python  every unique .py file (md5) under the local venvs' site-packages, docstrings and comments removed
          (ast.unparse), files in a fixed shuffled order
  romaji  the whole Aozora Bunko (globis-university/aozorabunko-clean on Hugging Face: 16,951 deduplicated works),
          ruby / annotation marks removed, romanized with pykakasi (Hepburn) one paragraph per line, works in a fixed
          shuffled order; run in parallel (pykakasi does ~1,200 Japanese characters/s per core)
  english the first articles of English Wikipedia (wikimedia/wikipedia 20231101.en, streamed in its own order) up to
          ENGLISH_CHARS normalized characters (added 2026-10-22 as a third corpus: all 26 letters like python, natural
          language like romaji); written to --out on a scratch disk (the main disk was full) and linked into OUT
  normalization  scripts/tinylm/data.normalize (lowercase; leading spaces kept; every other non-letter -> space; spaces
          collapsed; 3+ newlines -> 2). Vocabulary: newline, space, a-z (28).
Output: OUT/<name>.txt (everything), OUT/<name>_heldout.txt (the last HELDOUT chars, never trained on), stats.json.
  python data_big.py python|romaji [--workers 64]
"""
import argparse, ast, glob, gzip, hashlib, json, multiprocessing as mp, os, random, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tinylm"))
import data as D  # noqa: E402

OUT = "<data>/data_raw/tinylm2"   # shared with the compute nodes
AOZORA = "<data>/data_raw/aozorabunko_clean/aozorabunko-dedupe-clean.jsonl.gz"
ROOTS = sorted(glob.glob("<venvs>/*/lib/python3*/site-packages")) + \
    sorted(glob.glob("<repo>/.venv/lib/python3*/site-packages"))
HELDOUT = 5_000_000
ENGLISH_CHARS = 505_000_000


def strip_py(src):
    tree = ast.parse(src)
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) and \
                isinstance(getattr(body[0], "value", None), ast.Constant) and isinstance(body[0].value.value, str):
            node.body = body[1:] or [ast.Pass()]
    return ast.unparse(tree)


def py_one(path):
    try:
        b = open(path, "rb").read()
        if len(b) > 2_000_000:
            return None
        return D.normalize(strip_py(b.decode("utf-8")))
    except Exception:
        return None


def python_files():
    seen, files = set(), []
    for r in ROOTS:
        for dp, dn, fn in os.walk(r):
            for f in sorted(fn):
                if f.endswith(".py"):
                    p = os.path.join(dp, f)
                    try:
                        h = hashlib.md5(open(p, "rb").read()).hexdigest()
                    except Exception:
                        continue
                    if h not in seen:
                        seen.add(h)
                        files.append(p)
    random.Random(0).shuffle(files)
    return files


_K = {}


def romaji_one(text):
    import pykakasi
    if "k" not in _K:
        _K["k"] = pykakasi.kakasi()
    k = _K["k"]
    t = re.sub(r"《[^》]*》", "", text)
    t = re.sub(r"［＃[^］]*］", "", t).replace("｜", "").replace("　", "")
    lines = []
    for para in t.split("\n"):
        para = para.strip()
        lines.append(" ".join(x["hepburn"] for x in k.convert(para) if x["hepburn"].strip()) if para else "")
    return D.normalize("\n".join(lines))


def english_texts():
    from datasets import load_dataset
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    for row in ds:
        yield row["text"]


def write(name, parts, OUT=OUT):
    os.makedirs(OUT, exist_ok=True)
    text = "\n".join(p for p in parts if p and p.strip())
    assert set(text) <= set(D.VOCAB), set(text) - set(D.VOCAB)
    open(os.path.join(OUT, f"{name}.txt"), "w").write(text[:-HELDOUT])
    open(os.path.join(OUT, f"{name}_heldout.txt"), "w").write(text[-HELDOUT:])
    sp = os.path.join(OUT, "stats.json")
    st = json.load(open(sp)) if os.path.exists(sp) else {}
    st[name] = {"train_chars": len(text) - HELDOUT, "heldout_chars": HELDOUT, "parts": len(parts),
                "counts": {repr(c) if c in "\n " else c: text.count(c) for c in D.VOCAB}}
    json.dump(st, open(sp, "w"), indent=1)
    print(name, f"{len(text):,} characters ({len(text) - HELDOUT:,} train)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", choices=["python", "romaji", "english"])
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    with mp.get_context("fork").Pool(a.workers) as pool:
        if a.name == "english":
            parts, total = [], 0
            for i, r in enumerate(pool.imap(D.normalize, english_texts(), chunksize=64)):
                parts.append(r)
                total += len(r) + 1
                if i % 20000 == 0:
                    print(f"  {i:,} articles, {total:,} characters", flush=True)
                if total >= ENGLISH_CHARS:
                    break
            pool.terminate()
            write(a.name, parts, a.out)
            for f in (f"{a.name}.txt", f"{a.name}_heldout.txt"):
                if os.path.abspath(a.out) != os.path.abspath(OUT) and not os.path.exists(os.path.join(OUT, f)):
                    os.symlink(os.path.join(a.out, f), os.path.join(OUT, f))
            return
        if a.name == "python":
            files = python_files()
            print(f"{len(files):,} unique .py files", flush=True)
            parts = pool.map(py_one, files, chunksize=64)
        else:
            works = [json.loads(l)["text"] for l in gzip.open(AOZORA, "rt")]
            random.Random(0).shuffle(works)
            print(f"{len(works):,} works, {sum(map(len, works)):,} Japanese characters", flush=True)
            parts = []
            for i, r in enumerate(pool.imap(romaji_one, works, chunksize=4)):
                parts.append(r)
                if i % 1000 == 0:
                    print(f"  {i:,} works", flush=True)
    write(a.name, parts)


if __name__ == "__main__":
    main()
