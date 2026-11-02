"""Locate files of the blog's released logs (RikiyaT/lm_understand-logs @ 4d09ba6).

Small files are vendored in third_party/lm_understand-logs/; large ones are read from the full clone that
third_party/fetch_lm_understand_logs.sh creates in third_party/lm_understand-logs-full/.
"""
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENDORED = os.path.join(REPO, "third_party", "lm_understand-logs")
FULL = os.path.join(REPO, "third_party", "lm_understand-logs-full")
OUT = os.path.join(REPO, "results", "verify")
os.makedirs(OUT, exist_ok=True)


def logs(rel: str) -> str:
    for root in (VENDORED, FULL):
        p = os.path.join(root, rel)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{rel}: not vendored; run third_party/fetch_lm_understand_logs.sh")
