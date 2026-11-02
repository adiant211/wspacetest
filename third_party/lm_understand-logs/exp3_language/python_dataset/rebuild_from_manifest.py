"""Rebuild the Python training text of Experiment 3 from files.tsv.

1. Install the package versions listed in packages.txt (Python 3.10; the text was built with CPython 3.10.12, whose
   ast.unparse defines the exact output). They do not all fit in one environment; several are fine.
2. python rebuild_from_manifest.py <out_dir> <site-packages dir> [<site-packages dir> ...]

Each row of files.tsv with kind "file" is looked up by its path inside the given site-packages directories and must
have the listed SHA-256. It is processed exactly as scripts/tinylm2/data_big.py does (docstrings removed with ast,
ast.unparse, then scripts/tinylm/data.normalize), the results are joined with newlines in the order of files.tsv, and
the last 5,000,000 characters are held out. Writes <out_dir>/python.txt and <out_dir>/python_heldout.txt and compares
them with corpus_sha256.txt.

Rows with kind "unidentified" are stretches of the original text whose source files we could not identify (see the
README); they are skipped, so the rebuilt text matches the original only where every row is identified. Each row's
normalized_sha256 lets you check a single file (or a candidate for an unidentified stretch) on its own.
"""
import csv, hashlib, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "scripts", "tinylm2"))
import data_big as B  # noqa: E402  (py_one, HELDOUT; imports scripts/tinylm/data.py for normalize)


def main():
    out, roots = sys.argv[1], sys.argv[2:]
    rows = list(csv.DictReader(open(os.path.join(HERE, "files.tsv")), delimiter="\t"))
    parts, missing, bad, skipped = [], 0, 0, 0
    for r in rows:
        if r["kind"] != "file":
            skipped += 1
            continue
        found = None
        for root in roots:
            p = os.path.join(root, r["path"])
            if os.path.exists(p) and hashlib.sha256(open(p, "rb").read()).hexdigest() == r["sha256"]:
                found = p
                break
        if found is None:
            missing += 1
            print("missing or different:", r["package"], r["version"], r["path"])
            continue
        part = B.py_one(found)
        if part is None or hashlib.sha256(part.encode()).hexdigest() != r["normalized_sha256"]:
            bad += 1
            print("processed text differs (Python version?):", r["path"])
        parts.append(part)
    text = "\n".join(p for p in parts if p and p.strip())
    os.makedirs(out, exist_ok=True)
    files = {"python.txt": text[:-B.HELDOUT], "python_heldout.txt": text[-B.HELDOUT:]}
    want = dict(l.split()[::-1] for l in open(os.path.join(HERE, "corpus_sha256.txt"))
                if l.strip() and not l.startswith("#"))
    for name, t in files.items():
        open(os.path.join(out, name), "w").write(t)
        h = hashlib.sha256(t.encode()).hexdigest()
        print(name, f"{len(t):,} characters", "sha256 OK" if want.get(name) == h else "sha256 DIFFERENT")
    n = len(rows) - skipped
    print(f"{n - missing:,} of {n:,} files found, {bad:,} processed differently, {skipped:,} unidentified rows skipped")


if __name__ == "__main__":
    main()
