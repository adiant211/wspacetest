"""Three character corpora for the tiny-LM reading test (a 1-layer character-level transformer trained on
Shakespeare, Python code or romanized Japanese; which corpus it learned is read from its weights).

  shakespeare  tiny Shakespeare (karpathy/char-rnn data/tinyshakespeare/input.txt)
  python       the CPython 3.10 standard library (/usr/lib/python3.10/*.py), docstrings and comments removed (ast.unparse)
  romaji       four Natsume Soseki novels from Aozora Bunko (wagahai wa neko de aru, botchan, kokoro, sanshiro), ruby and
               annotations removed, romanized with pykakasi (Hepburn), one paragraph per line

The same normalization for all three, so that only the statistics of the text differ: lowercase; per line, the leading
spaces (indentation) are kept and every other character that is not a-z becomes a space, runs of spaces collapse to
one, trailing spaces go; runs of 3+ newlines collapse to 2. Vocabulary: newline, space, a-z (28). Each corpus is cut to
N_CHARS; the first 90 % is train, the last 10 % is held out (contexts for the next-character question come from it).

  python data.py            # writes DATA/<name>.txt and DATA/stats.json
"""
import ast, glob, json, os, re, sys

RAW = "<data>/data/tinylm/raw"
DATA = "<data>/data/tinylm"
NAMES = ("shakespeare", "python", "romaji")
VOCAB = "\n " + "abcdefghijklmnopqrstuvwxyz"
N_CHARS = 1_000_000
TRAIN_FRAC = 0.9


def normalize(text):
    out = []
    for line in text.lower().split("\n"):
        body = line.lstrip(" ")
        lead = len(line) - len(body)
        body = re.sub(r"[^a-z]+", " ", body).strip()
        out.append(" " * lead + body if body else "")
    t = "\n".join(out)
    return re.sub(r"\n{3,}", "\n\n", t).strip("\n") + "\n"


def shakespeare():
    return open(os.path.join(RAW, "shakespeare.txt")).read()


def python_code():
    parts = []
    for f in sorted(glob.glob("/usr/lib/python3.10/*.py")):
        try:
            tree = ast.parse(open(f, encoding="utf-8").read())
        except (SyntaxError, UnicodeDecodeError, ValueError):
            continue
        for node in ast.walk(tree):
            body = getattr(node, "body", None)
            if isinstance(body, list) and body and isinstance(body[0], ast.Expr) and \
                    isinstance(getattr(body[0], "value", None), ast.Constant) and isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]
        parts.append(ast.unparse(tree))
    return "\n\n".join(parts)


def aozora_clean(raw):
    t = raw.replace("\r\n", "\n")
    seps = [m.start() for m in re.finditer(r"^-{20,}$", t, re.M)]
    if len(seps) >= 2:
        t = t[seps[1]:].split("\n", 1)[1]
    t = t.split("\n底本：")[0]
    t = re.sub(r"《[^》]*》", "", t)
    t = re.sub(r"［＃[^］]*］", "", t)
    return t.replace("｜", "").replace("\u3000", "")


def romaji():
    import pykakasi
    k = pykakasi.kakasi()
    lines = []
    for name in ("wagahaiwa_nekodearu", "bocchan", "kokoro", "sanshiro"):
        raw = open(os.path.join(RAW, "aozora", f"{name}.txt"), "rb").read().decode("cp932")
        for para in aozora_clean(raw).split("\n"):
            para = para.strip()
            lines.append(" ".join(x["hepburn"] for x in k.convert(para) if x["hepburn"].strip()) if para else "")
    return "\n".join(lines)


def build():
    stats = {}
    for name, fn in zip(NAMES, (shakespeare, python_code, romaji)):
        t = normalize(fn())
        assert set(t) <= set(VOCAB), set(t) - set(VOCAB)
        if len(t) < N_CHARS:
            sys.exit(f"{name}: only {len(t):,} characters")
        t = t[:N_CHARS]
        open(os.path.join(DATA, f"{name}.txt"), "w").write(t)
        stats[name] = {"chars": len(t), "train_chars": int(TRAIN_FRAC * len(t)),
                       "counts": {c if c not in "\n " else repr(c): t.count(c) for c in VOCAB}}
        print(name, len(t), "unused:", [c for c in VOCAB if c not in t])
    json.dump(stats, open(os.path.join(DATA, "stats.json"), "w"), indent=1)


def load(name):
    """(train text, held-out text) of a corpus."""
    t = open(os.path.join(DATA, f"{name}.txt")).read()
    n = int(TRAIN_FRAC * len(t))
    return t[:n], t[n:]


if __name__ == "__main__":
    build()
