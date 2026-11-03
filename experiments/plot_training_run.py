#!/usr/bin/env python3
"""Rebuild the W&B-style training dashboard for a finished run, offline, from local logs only.

Reads (read-only) a run directory such as /home/user/wandb_upload/wandb:
  metrics(1).jsonl                 one JSON record per optimizer step (primary source for step series)
  train(1).log                     trainer stdout: STEP / EVAL / CHECKPOINT / HUB_* lines, supervisor restarts,
                                   tracebacks, W&B client messages
  latest-run/files/output.log      stdout of the W&B-attached run (cross-checked against train(1).log)
  latest-run/files/wandb-summary.json, wandb-metadata.json   final W&B summary and run metadata
  results.jsonl                    end-of-run summary, including the per-step loss list (cross-checked)
  latest-run/logs/debug-internal.log   W&B client log: filestream errors with UTC timestamps
  latest-run/run-*.wandb           W&B transaction log: per-step history timestamps and GPU/CPU system stats

Writes PNG figures (dpi 150), index.html and summary.json to --out (default <repo>/results/training_run).

Usage:
  /home/user/wspacetest/venv/bin/python -I /home/user/wspacetest/experiments/plot_training_run.py \
      [DATA_DIR] [--out DIR] [--wandb-python /home/user/wandb_tools/venv/bin/python] [--no-system]

The .wandb file is decoded by a subprocess that runs this same file with --wandb-python (a venv that has the
wandb package installed; only its protobuf definitions are used, nothing is synced or sent). If that
interpreter is missing, the system/GPU panels are skipped and summary.json says so.

Duplicate handling: metrics records are keyed by step and the last-written record per step is kept; duplicate
steps, step regressions (restarts) and gaps are reported in summary.json and marked on the step plots.
"""
import argparse
import collections
import datetime as dt
import glob
import html as htmllib
import json
import math
import os
import re
import struct
import subprocess
import sys

try:  # the --extract-wandb subprocess runs in a venv without the plotting stack
    import numpy as np
    import pandas as pd
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MultipleLocator
except ImportError:
    np = pd = plt = None

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA = "/home/user/wandb_upload/wandb"
DEFAULT_OUT = os.path.join(REPO, "results", "training_run")
DEFAULT_WANDB_PY = "/home/user/wandb_tools/venv/bin/python"

# dataviz reference palette (light surface); categorical slots 1-3 are validated for this surface
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASE, SURFACE, PAGE = "#e1e0d9", "#c3c2b7", "#fcfcfb", "#f9f9f7"
CRIT, GOOD = "#d03b3b", "#0ca30c"  # status palette: always paired with a marker shape and a text label

EMA_ALPHA = 0.02
WANDB_BLOCK = 32768
WANDB_KEEP_HIST = {"step", "train/loss", "_timestamp", "_runtime", "_step", "run/restart", "ckpt/write_s",
                   "ckpt/stall_s", "ckpt/gb", "hub/upload_s", "hub/failed", "hub/attempts", "hub/verified"}
WANDB_KEEP_STATS = {
    "gpu.0.gpu", "gpu.0.memory", "gpu.0.memoryAllocated", "gpu.0.memoryAllocatedBytes", "gpu.0.temp",
    "gpu.0.powerWatts", "gpu.0.powerPercent", "gpu.0.enforcedPowerLimitWatts", "gpu.0.smClock",
    "gpu.0.memoryClock", "gpu.0.correctedMemoryErrors", "gpu.0.uncorrectedMemoryErrors", "gpu.0.energyJoules",
    "cpu", "memory_percent", "proc.memory.rssMB", "proc.memory.oomKills", "network.tcpRetransmits",
    "proc.cpu.throttledPercent",
}
STAT_BIN_SECONDS = 600.0

RE_TOKEN = re.compile(r"^([A-Z][A-Z_]+) (\{.*\})\s*$")
RE_SUP_START = re.compile(r"^SUPERVISOR start (\d+) (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
RE_SUP_EXIT = re.compile(r"^SUPERVISOR trainer exited with (\d+)")
RE_SUP_GIVEUP = re.compile(r"^SUPERVISOR (\d+) failures in a row")
RE_EXC = re.compile(r"^([A-Za-z_][\w.]*(?:Error|Exception|Exit|Interrupt)): (.*)$")
RE_WB_RESUME = re.compile(r"^wandb: Resuming run (\S+)")
RE_WB_SYNC = re.compile(r"^wandb: Syncing run (\S+)")
RE_WB_FATAL = re.compile(r"^wandb: WARNING Fatal error while uploading data")
RE_WB_SYNCED = re.compile(r"^wandb: Synced (\d+) W&B file")
STRUCT_TOKENS = {"STEP", "EVAL", "EVAL_BUDGET", "CHECKPOINT", "CHECKPOINT_STALL", "HUB_UPLOAD", "HUB_UPLOAD_RETRY",
                 "HUB_UPLOAD_FAILED", "HUB_WAIT", "RESULT"}
UTC = dt.timezone.utc


MARKS = []  # restart/crash events drawn as thin lines on every step-axis panel (set in build_figures)


def fmt_int(v):
    return f"{int(round(v)):,}"


def fmt_k(v, _pos=None):
    return f"{v / 1000:g}K" if abs(v) >= 1000 else f"{v:g}"


def to_py(o):
    """Make numpy/pandas scalars and NaN JSON-safe."""
    if isinstance(o, dict):
        return {str(k): to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_py(v) for v in o]
    if np is not None and isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    return o


# ----------------------------------------------------------------------------------------------------------------
# W&B transaction log (.wandb) -- decoded in the wandb venv
# ----------------------------------------------------------------------------------------------------------------
def _iter_wandb_payloads(path):
    """Yield record payloads from a W&B transaction log (LevelDB-style 32 KiB blocks after a 7-byte header)."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b":W&B":
        raise ValueError("not a W&B transaction log")
    pos, buf, n = 7, b"", len(data)
    while pos < n:
        off = pos % WANDB_BLOCK
        if WANDB_BLOCK - off < 7:
            pos += WANDB_BLOCK - off
            continue
        _crc, length, typ = struct.unpack("<IHB", data[pos:pos + 7])
        if typ == 0 and length == 0:
            pos += WANDB_BLOCK - off
            continue
        if typ not in (1, 2, 3, 4) or pos + 7 + length > n:
            raise ValueError(f"bad chunk at byte {pos}")
        payload = data[pos + 7: pos + 7 + length]
        pos += 7 + length
        if typ == 1:
            yield payload
        elif typ == 2:
            buf = payload
        elif typ == 3:
            buf += payload
        else:
            yield buf + payload
            buf = b""


def extract_wandb(path):
    """Decode history and system stats from a .wandb file. Runs in the wandb venv; prints JSON to stdout."""
    from wandb.proto import wandb_internal_pb2 as pb

    counts = collections.Counter()
    hist_key_counts = collections.Counter()
    rows, stats, bad = [], [], 0
    for payload in _iter_wandb_payloads(path):
        rec = pb.Record()
        try:
            rec.ParseFromString(payload)
        except Exception:
            bad += 1
            continue
        kind = rec.WhichOneof("record_type") or "none"
        counts[kind] += 1
        if kind == "history":
            vals, prefixes = {}, set()
            for it in rec.history.item:
                name = "/".join(it.nested_key) if len(it.nested_key) else it.key
                hist_key_counts[name] += 1
                if "/" in name:
                    prefixes.add(name.split("/")[0])
                if name.startswith("eval/") or name in WANDB_KEEP_HIST:
                    try:
                        vals[name] = json.loads(it.value_json)
                    except ValueError:
                        pass
            rows.append({"ws": int(rec.history.step.num), "v": vals, "p": sorted(prefixes)})
        elif kind == "stats":
            ts = rec.stats.timestamp.seconds + rec.stats.timestamp.nanos * 1e-9
            vals = {}
            for it in rec.stats.item:
                if it.key in WANDB_KEEP_STATS:
                    try:
                        vals[it.key] = float(json.loads(it.value_json))
                    except ValueError:
                        pass
            stats.append((ts, vals))
    out = {"record_counts": dict(counts), "parse_errors": bad, "history_rows": rows,
           "history_key_counts": dict(hist_key_counts.most_common(400)), "stats_n": len(stats)}
    if stats:
        ts_list = [t for t, _ in stats]
        out["stats_t0"], out["stats_t1"] = ts_list[0], ts_list[-1]
        gaps = sorted(((b - a, a) for a, b in zip(ts_list, ts_list[1:])), reverse=True)[:5]
        out["stats_top_gaps"] = [{"gap_s": g, "start_ts": a} for g, a in gaps]
        series = collections.defaultdict(list)
        for _, vals in stats:
            for k, v in vals.items():
                series[k].append(v)
        glob_stats = {}
        for k, vs in series.items():
            s_sorted = sorted(vs)
            glob_stats[k] = {"n": len(vs), "min": s_sorted[0], "max": s_sorted[-1],
                             "mean": sum(vs) / len(vs), "p95": s_sorted[int(0.95 * (len(vs) - 1))]}
        out["stats_global"] = glob_stats
        bins = collections.OrderedDict()
        for ts, vals in stats:
            b = int((ts - ts_list[0]) // STAT_BIN_SECONDS)
            cell = bins.get(b)
            if cell is None:
                cell = bins[b] = {"n": 0, "sum": collections.defaultdict(float),
                                  "cnt": collections.Counter(), "max": {}}
            cell["n"] += 1
            for k, v in vals.items():
                cell["sum"][k] += v
                cell["cnt"][k] += 1
                cell["max"][k] = max(cell["max"].get(k, -math.inf), v)
        out["stats_bins"] = [{"t": ts_list[0] + (b + 0.5) * STAT_BIN_SECONDS, "n": c["n"],
                              "mean": {k: c["sum"][k] / c["cnt"][k] for k in c["sum"]},
                              "max": c["max"]} for b, c in bins.items()]
    return out


def run_wandb_extract(wandb_py, wb_file):
    env = dict(os.environ, WANDB_MODE="offline", WANDB_SILENT="true")
    proc = subprocess.run([wandb_py, "-I", os.path.abspath(__file__), "--extract-wandb", wb_file],
                          capture_output=True, text=True, env=env, timeout=3600)
    if proc.returncode != 0:
        raise RuntimeError("wandb decode failed: " + proc.stderr.strip()[-300:])
    text = proc.stdout.strip()
    try:
        return json.loads(text)
    except ValueError:
        return json.loads(text.splitlines()[-1])


# ----------------------------------------------------------------------------------------------------------------
# Text logs
# ----------------------------------------------------------------------------------------------------------------
def parse_train_log(path):
    """Parse trainer stdout. Returns attempts (supervisor launches), per-step text records, eval passes, checkpoint
    and hub events, the W&B client's fatal sync error and error counters."""
    attempts, cur = [], None
    steps_text, evals, pending_evals = {}, [], []
    ckpt, stall, hub_ok, hub_retry, hub_fail = {}, {}, {}, collections.defaultdict(list), {}
    tokens_seen_text = {}
    counts = collections.Counter()
    struct_seq = []
    result = None
    last_step, tb_line = None, None
    wb_fatal = None
    wb_resumes = []
    synced = None
    give_up_line = None
    with open(path, encoding="utf-8", errors="replace") as f:
        for ln, raw in enumerate(f, 1):
            line = raw.rstrip("\n")
            m = RE_SUP_START.match(line)
            if m:
                cur = {"index": len(attempts), "supervisor_label": int(m.group(1)), "start_time": m.group(2),
                       "start_line": ln, "exit_code": None, "exit_line": None, "traceback_line": None,
                       "error_line": None, "error": None, "config": None, "steps": [], "resumed": False}
                attempts.append(cur)
                continue
            if cur is None:
                cur = {"index": 0, "supervisor_label": None, "start_time": None, "start_line": 0,
                       "exit_code": None, "exit_line": None, "traceback_line": None, "error_line": None,
                       "error": None, "config": None, "steps": [], "resumed": False}
                attempts.append(cur)
            m = RE_SUP_EXIT.match(line)
            if m:
                cur["exit_code"], cur["exit_line"] = int(m.group(1)), ln
                continue
            if RE_SUP_GIVEUP.match(line):
                counts["supervisor_gave_up"] += 1
                give_up_line = ln
                continue
            if line.startswith("Traceback (most recent call last)"):
                tb_line = ln
                cur["traceback_line"] = ln
                continue
            if tb_line is not None:
                m = RE_EXC.match(line)
                if m:
                    cur["error_line"], cur["error"] = ln, (m.group(1) + ": " + m.group(2))[:300]
                    tb_line = None
                continue
            if RE_WB_RESUME.match(line):
                cur["resumed"] = True
                wb_resumes.append(ln)
                continue
            m = RE_WB_SYNCED.match(line)
            if m:
                synced = {"line": ln, "files": int(m.group(1))}
                continue
            if RE_WB_FATAL.match(line):
                wb_fatal = {"line": ln, "after_step": last_step}
                continue
            if "code: 403" in line and "export metrics" in line:
                counts["export_403"] += 1
                continue
            if "due to timeout, max retries" in line:
                counts["export_timeout"] += 1
                continue
            if line.startswith("Transient error") and "encountered while exporting" in line:
                counts["transient_unavailable"] += 1
                continue
            if line.startswith("wandb: 500 encountered"):
                counts["http_500"] += 1
                continue
            if line.startswith("wandb: 403 encountered"):
                counts["http_403"] += 1
                continue
            m = RE_TOKEN.match(line)
            if not m:
                continue
            tok, payload = m.group(1), m.group(2)
            if tok not in STRUCT_TOKENS and tok not in ("CONFIG", "MODEL", "OPTIMIZER", "PEAK_TFLOPS", "HUB",
                                                        "EVAL_BUDGET", "DELTA_BLOCK", "FLA_NOSYNC", "NGRAM_CANON"):
                continue
            try:
                rec = json.loads(payload)
            except ValueError:
                counts["unparsed_" + tok] += 1
                continue
            if tok in STRUCT_TOKENS:
                struct_seq.append((tok, payload))
            if tok == "CONFIG":
                cur["config"] = rec
            elif tok == "STEP":
                s = int(rec["step"])
                steps_text[s] = rec
                cur["steps"].append(s)
                last_step = s
                for ev in pending_evals:  # an EVAL line is printed right before the STEP line of its step
                    ev["step"], ev["kind"] = s, "scheduled"
                    evals.append(ev)
                pending_evals = []
            elif tok == "EVAL":
                pending_evals.append({"line": ln, "values": rec, "step": None, "kind": None})
            elif tok == "CHECKPOINT":
                ckpt[int(rec["step"])] = dict(rec, line=ln)
            elif tok == "CHECKPOINT_STALL":
                stall[int(rec["step"])] = dict(rec, line=ln)
            elif tok == "HUB_UPLOAD":
                hub_ok[int(rec["step"])] = dict(rec, line=ln)
            elif tok == "HUB_UPLOAD_RETRY":
                hub_retry[int(rec["step"])].append({"line": ln, "attempt": rec.get("attempt"),
                                                    "error": rec.get("error", "")})
            elif tok == "HUB_UPLOAD_FAILED":
                hub_fail[int(rec["step"])] = dict(rec, line=ln)
            elif tok == "HUB_WAIT":
                counts["hub_wait"] += 1
            elif tok == "RESULT":
                result = dict(rec, line=ln)
    # an EVAL with no following STEP line is the post-training pass; it belongs to the last step
    for ev in pending_evals:
        ev["step"], ev["kind"] = last_step, "final"
        evals.append(ev)
    return {"attempts": attempts, "steps_text": steps_text, "evals": evals, "ckpt": ckpt, "stall": stall,
            "hub_ok": hub_ok, "hub_retry": dict(hub_retry), "hub_fail": hub_fail, "counts": dict(counts),
            "struct_seq": struct_seq, "result": result, "wb_fatal": wb_fatal, "wb_resumes": wb_resumes,
            "synced": synced, "last_step": last_step, "give_up_line": give_up_line}


def parse_debug_internal(path):
    out = {"lines": 0, "retry_by_status": collections.Counter(), "fatal": [], "http403": [],
           "http_error_by_status": collections.Counter()}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            out["lines"] += 1
            try:
                d = json.loads(line)
            except ValueError:
                continue
            msg = str(d.get("msg", ""))
            if "fatal" in msg.lower() and len(out["fatal"]) < 10:
                out["fatal"].append({"time": d.get("time"), "msg": msg.split(" url=")[0][:160]})
            if msg.startswith("api: retrying"):
                out["retry_by_status"][str(d.get("status", "error"))] += 1
            if msg.startswith("api: HTTP error"):
                out["http_error_by_status"][str(d.get("status"))] += 1
                if d.get("status") == 403:
                    out["http403"].append(d.get("time"))
    out["retry_by_status"] = dict(out["retry_by_status"])
    out["http_error_by_status"] = dict(out["http_error_by_status"])
    out["http403_first"] = out["http403"][0] if out["http403"] else None
    out["http403_last"] = out["http403"][-1] if out["http403"] else None
    out["http403_count"] = len(out["http403"])
    del out["http403"]
    return out


def classify_hub_error(msg):
    if "private storage limit" in msg.lower():
        return "Hugging Face hub: private storage limit reached (BadRequestError on the commit endpoint)"
    return msg[:160]


# ----------------------------------------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------------------------------------
def load_metrics(path):
    recs, bad = [], 0
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                r = json.loads(s)
            except ValueError:
                bad += 1
                continue
            if not isinstance(r, dict) or "step" not in r:
                bad += 1
                continue
            recs.append((ln, r))
    raw_steps = [int(r["step"]) for _, r in recs]
    counts = collections.Counter(raw_steps)
    regress = [{"line": ln, "step": int(r["step"]), "previous_step": raw_steps[i - 1]}
               for i, (ln, r) in enumerate(recs) if i > 0 and raw_steps[i] <= raw_steps[i - 1]]
    last = {}
    for ln, r in recs:  # keep the last-written record per step
        last[int(r["step"])] = (ln, r)
    steps = sorted(last)
    missing = sorted(set(range(steps[0], steps[-1] + 1)) - set(steps)) if steps else []
    rows = []
    for s in steps:
        ln, r = last[s]
        d = dict(r)
        d["step"], d["_line"] = s, ln
        rows.append(d)
    df = pd.DataFrame(rows).set_index("step")
    nonfinite = sum(1 for _, r in recs for v in r.values() if isinstance(v, float) and not math.isfinite(v))
    info = {"records": len(recs), "bad_lines": bad, "unique_steps": len(steps),
            "step_min": steps[0], "step_max": steps[-1],
            "duplicate_step_values": sorted(s for s, c in counts.items() if c > 1)[:50],
            "duplicate_step_count": sum(1 for c in counts.values() if c > 1),
            "regressions": regress[:50], "regression_count": len(regress),
            "missing_steps_in_range": missing[:50], "missing_step_count": len(missing),
            "nonfinite_values": nonfinite}
    return df, info


def ema(y, alpha=EMA_ALPHA):
    return pd.Series(np.asarray(y, dtype=float)).ewm(alpha=alpha, adjust=False).mean().to_numpy()


def fit_wsd_decay(steps, lr_mult, last_step):
    """Fit lr_mult = 1 - sqrt((s - s0) / (last - s0)) for s > s0, else 1, over a grid of s0 (max-abs error)."""
    best = None
    for s0 in np.arange(12000.0, 12500.0, 0.5):
        pred = np.ones_like(steps, dtype=float)
        m = steps > s0
        pred[m] = 1.0 - np.sqrt((steps[m] - s0) / (last_step - s0))
        err = float(np.max(np.abs(pred - lr_mult)))
        if best is None or err < best[1]:
            best = (float(s0), err)
    return best


# ----------------------------------------------------------------------------------------------------------------
# Plot helpers
# ----------------------------------------------------------------------------------------------------------------
def setup_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": BASE, "axes.linewidth": 0.8, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.labelsize": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "xtick.major.size": 0, "ytick.major.size": 0,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.fontsize": 8, "legend.labelcolor": INK2,
        "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "font.family": "DejaVu Sans", "text.color": INK2, "axes.axisbelow": True,
    })


def style_ax(ax, title, xlabel=None, ylabel=None):
    ax.set_title(title, loc="left", pad=7)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def step_axis(ax, n_max):
    ax.xaxis.set_major_locator(MultipleLocator(5000))
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_k))
    ax.set_xlim(-0.015 * n_max, 1.01 * n_max)
    ax.set_xlabel("optimizer step")


def series(ax, x, y, color, label=None, raw=True, smooth=True, lw=1.5, logy=False, zorder=3):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(y)
    if logy:
        m &= y > 0
    x, y = x[m], y[m]
    if raw:
        ax.plot(x, y, color=color, lw=0.6, alpha=0.3, zorder=zorder - 1)
    if smooth:
        ax.plot(x, ema(y), color=color, lw=lw, label=label, zorder=zorder)
    else:
        ax.plot(x, y, color=color, lw=lw, label=label, zorder=zorder)


def draw_events(ax, events, y_frac=0.985):
    for ev in events:
        ax.axvline(ev["x"], color=INK2, lw=0.8, zorder=1)
        ha = ev.get("ha", "left")
        ax.annotate(ev["label"], xy=(ev["x"], y_frac - 0.07 * ev.get("row", 0)), xycoords=("data", "axes fraction"),
                    xytext=((4 if ha == "left" else -4), 0), textcoords="offset points", ha=ha, va="top",
                    fontsize=7.4, color=INK2, annotation_clip=False)


def shade(ax, spans, label=None):
    for i, (a, b) in enumerate(spans):
        ax.axvspan(a, b, color=BASE, alpha=0.35, zorder=0, lw=0,
                   label=label if (i == 0 and label) else None)


def add_token_axis(ax, tokens_per_step):
    sec = ax.secondary_xaxis("top", functions=(lambda s: s * tokens_per_step / 1e9,
                                               lambda b: b * 1e9 / tokens_per_step))
    sec.set_xlabel("tokens seen (billions)", fontsize=8, color=INK2)
    sec.tick_params(labelsize=7.5, colors=MUTED, length=0)


def mark_all(fig, events):
    """Thin vertical lines at restart/crash steps on every panel whose x-axis is optimizer step."""
    for ax in fig.axes:
        xl = ax.get_xlabel()
        if ax.get_xscale() == "log" or "hours" in xl or "tokens" in xl:
            continue
        for ev in events:
            ax.axvline(ev["x"], color=INK2, lw=0.7, zorder=1)


def finish(fig, out_dir, name, title, subtitle=None):
    fig.suptitle(title, x=0.012, y=0.995, ha="left", va="top", fontsize=12, color=INK, fontweight="bold")
    if MARKS and not name.startswith("00_"):
        mark_all(fig, MARKS)
    h = fig.get_figheight()
    y_sub =0.995 - (12 * 1.35 / 72.0) / h - 0.012
    if subtitle:
        fig.text(0.012, y_sub, subtitle, ha="left", va="top", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, y_sub - (8.5 * 1.6 / 72.0) / h - 0.01))
    fig.savefig(os.path.join(out_dir, name), dpi=150, facecolor=SURFACE)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------------------------------
def build_figures(out_dir, D):
    M, N = D["M"], D["N"]
    MARKS[:] = D["plot_events"]
    st = M.index.to_numpy()
    ev = D["events"]
    tps = D["tokens_per_step"]
    figs = []

    # 00 run timeline ------------------------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10.5, 3.7))
    lanes = ["trainer process", "W&B online sync", "HF hub upload", "eval passes"]
    ypos = {name: len(lanes) - 1 - i for i, name in enumerate(lanes)}
    for name, y in ypos.items():
        ax.axhline(y, color=GRID, lw=0.8, zorder=0)
    for k, crash in enumerate(D["startup_crashes"]):
        ax.scatter(crash["x"], ypos["trainer process"] + 0.2 + 0.2 * k, marker="X", s=80, color=CRIT, zorder=5,
                   edgecolors=SURFACE, linewidths=0.8)
    ax.plot([1, N], [ypos["trainer process"]] * 2, color=INK2, lw=1.2, zorder=3, solid_capstyle="butt")
    ax.text(N * 0.012, ypos["trainer process"] + 0.05,
            f"{len(D['startup_crashes'])} startup crashes before step 1 (red crosses, 0 steps each); "
            f"then one launch ran steps 1 to {fmt_int(N)}", fontsize=7.5, color=INK2, ha="left", va="bottom")
    if D["wb_fatal_step"] is not None:
        fs = D["wb_fatal_step"]
        ax.plot([fs, N], [ypos["W&B online sync"]] * 2, color=CRIT, lw=3, solid_capstyle="butt", zorder=3)
        ax.text(fs, ypos["W&B online sync"] + 0.22,
                f"403 Forbidden after step {fmt_int(fs)}" + (" (local .wandb log still complete)"
                                                          if D.get("wb_local_complete") else ""),
                fontsize=7.5, color=INK2, ha="left", va="bottom")
    hub_ok_steps = sorted(D["hub_ok"])
    for s in D["ckpt_steps"]:
        ok = s in D["hub_ok"]
        ax.scatter(s, ypos["HF hub upload"], marker="o" if ok else "X", s=48 if ok else 60,
                   color=GOOD if ok else CRIT, zorder=4, edgecolors=SURFACE, linewidths=0.8)
    ax.text(N * 0.99, ypos["HF hub upload"] + 0.22,
            f"{len(hub_ok_steps)} of {len(D['ckpt_steps'])} checkpoints uploaded (green circle); "
            f"{len(D['ckpt_steps']) - len(hub_ok_steps)} failed after 6 attempts (red cross)",
            fontsize=7.5, color=INK2, ha="right", va="bottom")
    for s in sorted(e["step"] for e in ev["evals"]):
        ax.scatter(s, ypos["eval passes"], marker="D", s=34, color=BLUE, zorder=4, edgecolors=SURFACE, linewidths=0.8)
    ax.text(N * 0.99, ypos["eval passes"] + 0.22,
            f"{len(ev['evals'])} eval passes (5 scheduled, 1 final)", fontsize=7.5, color=INK2, ha="right", va="bottom")
    ax.set_yticks([ypos[n] for n in lanes])
    ax.set_yticklabels(lanes, fontsize=8.5, color=INK2)
    ax.set_ylim(-0.6, len(lanes) - 0.3)
    ax.grid(False)
    step_axis(ax, N)
    ax.spines["left"].set_visible(False)
    finish(fig, out_dir, "00_run_timeline.png", "Run timeline by optimizer step",
           "Restarts and failures marked; the crash cause is the ValueError in the startup tracebacks")
    figs.append(("00_run_timeline.png", "Run timeline by optimizer step",
                 f"Trainer launches, W&B online sync, hub uploads and eval passes on one step axis. Three startups "
                 f"crashed before any step ran; the sync was rejected after step {fmt_int(D['wb_fatal_step'] or 0)}; "
                 f"only the step-{fmt_int(D['hub_uploaded_step'] or 0)} checkpoint reached the hub."))

    # 01 loss -------------------------------------------------------------------------------------------------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.45, 1]})
    loss = M["loss"].to_numpy()
    for a, logx in ((a1, False), (a2, True)):
        series(a, st, loss, BLUE, label="train loss, EMA (alpha 0.02)", raw=True, smooth=True)
        ev_s = sorted([e for e in ev["evals"] if e["kind"] == "scheduled"], key=lambda e: e["step"])
        ev_f = [e for e in ev["evals"] if e["kind"] == "final"]
        xs = np.array([e["step"] for e in ev_s])
        a.plot(xs, [e["eval_loss"] for e in ev_s], color=ORANGE, lw=1.5, zorder=4,
               label="eval loss, n-gram memory on")
        a.scatter(xs, [e["eval_loss"] for e in ev_s], s=36, color=ORANGE, zorder=5, edgecolors=SURFACE, linewidths=1.2)
        a.plot(xs, [e["eval_loss_no_ngram"] for e in ev_s], color=AQUA, lw=1.5, zorder=4,
               label="eval loss, n-gram memory off")
        a.scatter(xs, [e["eval_loss_no_ngram"] for e in ev_s], s=36, color=AQUA, zorder=5, edgecolors=SURFACE,
                  linewidths=1.2)
        for e in ev_f:
            a.scatter([e["step"]], [e["eval_loss"]], s=44, facecolors=SURFACE, edgecolors=ORANGE, linewidths=1.6, zorder=6)
            a.scatter([e["step"]], [e["eval_loss_no_ngram"]], s=44, facecolors=SURFACE, edgecolors=AQUA, linewidths=1.6, zorder=6)
        if logx:
            a.set_xscale("log")
            a.set_xlim(0.8, N * 1.2)
            a.set_xticks([1, 10, 100, 1000, 10000])
            a.xaxis.set_major_formatter(FuncFormatter(fmt_k))
            a.xaxis.set_minor_formatter(FuncFormatter(lambda v, p: ""))
            a.set_xlabel("optimizer step (log scale)")
            style_ax(a, "Same curves, log step axis")
            a.set_ylim(1.8, 13.5)
        else:
            step_axis(a, N)
            style_ax(a, "Loss vs step, with eval passes")
            a.set_ylim(1.8, 13.5)
            draw_events(a, D["plot_events"])
            add_token_axis(a, tps)
        a.set_ylabel("cross-entropy loss (nats per token)")
    last_e = ev_s[-1]
    a1.set_xlim(-0.015 * N, 1.16 * N)
    a1.annotate(f"no n-gram {last_e['eval_loss_no_ngram']:.3f}", xy=(last_e["step"], last_e["eval_loss_no_ngram"]),
                xytext=(7, 0), textcoords="offset points", fontsize=7.6, color=INK2, va="center")
    a1.annotate(f"n-gram on {last_e['eval_loss']:.3f}", xy=(last_e["step"], last_e["eval_loss"]),
                xytext=(7, -1), textcoords="offset points", fontsize=7.6, color=INK2, va="center")
    a1.legend(loc="upper right", fontsize=7.6)
    finish(fig, out_dir, "01_loss.png", "Train loss and eval loss",
           f"Hollow markers: final pass after step {fmt_int(N)}. Faint lines are raw per-step values.")
    figs.append(("01_loss.png", "Train loss and eval loss",
                 f"Train loss per step (faint raw, solid EMA) with the five scheduled eval passes plus the final pass, "
                 f"with and without the n-gram memory (the gap is the n-gram gain). Top axis: tokens seen. "
                 f"Vertical lines: startup crashes (step 0) and the W&B sync failure (step {fmt_int(D['wb_fatal_step'] or 0)})."))

    # 02 eval panels ------------------------------------------------------------------------------------------
    ev_all = sorted(ev["evals"], key=lambda e: e["step"])
    xs = np.array([e["step"] for e in ev_all])
    panels = [("eval_ppl", "eval perplexity (n-gram on)", "perplexity", "%.3f"),
              ("eval_ngram_gain", "n-gram gain (loss off minus on)", "nats per token", "%.4f"),
              ("eval_loss_early_exit", "early-exit eval loss", "nats per token", "%.3f"),
              ("ngram_table_rms", "n-gram table RMS at eval", "RMS", "%.4f")]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.2))
    for ax, (key, title, ylab, f) in zip(axes.ravel(), panels):
        ys = np.array([e[key] for e in ev_all], dtype=float)
        sched = np.array([e["kind"] == "scheduled" for e in ev_all])
        ax.plot(xs, ys, color=BLUE, lw=1.5, zorder=3)
        ax.scatter(xs[sched], ys[sched], s=36, color=BLUE, zorder=4, edgecolors=SURFACE, linewidths=1.2)
        ax.scatter(xs[~sched], ys[~sched], s=44, facecolors=SURFACE, edgecolors=BLUE, linewidths=1.6, zorder=5)
        rising = len(ys) > 1 and ys[1] > ys[0]
        for idx in (0, len(ys) - 1):
            below = rising and idx == 0
            ax.annotate(f % ys[idx], xy=(xs[idx], ys[idx]), xytext=(0, -13 if below else 7),
                        textcoords="offset points", ha="center" if idx else "left", fontsize=7.6, color=INK2)
        step_axis(ax, N)
        style_ax(ax, title, ylabel=ylab)
        ax.set_ylim(min(ys) * 0.97 if key != "eval_ppl" else min(ys) * 0.9, max(ys) * 1.06)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.suptitle("Eval passes", x=0.012, y=0.995, ha="left", va="top", fontsize=12, color=INK, fontweight="bold")
    fig.text(0.012, 0.952, f"{len(ev_all)} passes on a fixed 1,085-batch set. Hollow marker: final pass after the last "
             "step (the same values as step 17,540 to 4 decimals).", ha="left", va="top", fontsize=8.5, color=INK2)
    mark_all(fig, MARKS)
    fig.savefig(os.path.join(out_dir, "02_eval_panels.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)
    figs.append(("02_eval_panels.png", "Eval perplexity, n-gram gain, early-exit loss, table RMS",
                 "Each eval pass (five scheduled, one final). The early-exit loss rises across training while the "
                 "full-model eval loss falls; the n-gram gain grows and dips slightly at the last scheduled pass."))

    # 03 learning rate ----------------------------------------------------------------------------------------
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10.5, 6.4), sharex=True, gridspec_kw={"height_ratios": [1.1, 1]})
    series(a1, st, M["lr"].to_numpy(), BLUE, label="lr (peak 0.02; lr/nordion2 and ngram/lr match)",
           raw=False, smooth=False, lw=1.6, logy=True)
    series(a1, st, M["lr/adamw"].to_numpy(), ORANGE, label="lr/adamw (peak 0.003; adamw2 and adamw3 are identical)",
           raw=False, smooth=False, lw=1.6, logy=True)
    a1.set_yscale("log")
    style_ax(a1, "Learning rate by parameter group (log scale)", ylabel="learning rate")
    a1.legend(loc="lower left", fontsize=7.8)
    step_axis(a1, N)
    a1.set_xlabel("")
    fit = D["lr_fit"]
    series(a2, st, M["lr_mult"].to_numpy(), BLUE, raw=False, smooth=False, lw=1.6, label="logged lr_mult")
    pred = np.ones(len(st))
    m = st > fit["s0"]
    pred[m] = 1.0 - np.sqrt((st[m] - fit["s0"]) / (N - fit["s0"]))
    a2.plot(st, pred, color=INK2, lw=0.9, zorder=4,
            label=f"fit: 1 - sqrt((step - {fit['s0']:,.0f}) / ({fmt_int(N)} - {fit['s0']:,.0f})), max error {fit['err']:.1e}")
    a2.axvline(D["warmup_steps"], color=MUTED, lw=0.7)
    a2.text(D["warmup_steps"] + 150, 0.94, f"warmup ends at step {D['warmup_steps']}", fontsize=7.5, color=INK2, va="top")
    a2.axvline(fit["s0"], color=MUTED, lw=0.7)
    a2.text(fit["s0"] - 150, 0.5, f"decay starts ~{fit['s0']:,.0f}", fontsize=7.5, color=INK2, ha="right", va="center")
    a2.set_ylim(-0.03, 1.08)
    style_ax(a2, "Schedule multiplier (1.0 = peak LR)", ylabel="lr_mult")
    a2.legend(loc="lower left", fontsize=7.6)
    step_axis(a2, N)
    finish(fig, out_dir, "03_lr_schedule.png", "Learning-rate schedule",
           f"Linear warmup over {D['warmup_steps']} steps, then 1 - sqrt decay over the last 29.5% of steps.")
    figs.append(("03_lr_schedule.png", "Learning-rate schedule",
                 f"Learning rate per parameter group (log scale) and the logged schedule multiplier with its fit "
                 f"(max error {fit['err']:.1e}). Warmup ends at step {D['warmup_steps']}; decay starts near step {fit['s0']:,.0f}."))

    # 04 grad norm ---------------------------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    series(ax, st, M["gnorm"].to_numpy(), BLUE, raw=True, smooth=True, logy=True)
    ax.set_yscale("log")
    ax.axhline(D["grad_clip"], color=INK2, lw=0.8, zorder=2)
    ax.annotate(f"grad-clip {D['grad_clip']:.1f} (config)", xy=(N, D["grad_clip"]), xytext=(-4, 3),
                textcoords="offset points", ha="right", fontsize=7.6, color=INK2)
    step_axis(ax, N)
    style_ax(ax, "gnorm, every step", ylabel="gradient norm (log scale)")
    draw_events(ax, D["plot_events"])
    ax.set_ylim(0.02, 4.5)
    finish(fig, out_dir, "04_grad_norm.png", "Gradient norm",
           f"Faint raw per step, solid EMA. Peak {M['gnorm'].max():.2f} at step {int(M['gnorm'].idxmax())}; "
           f"last step {M['gnorm'].iloc[-1]:.3f}.")
    figs.append(("04_grad_norm.png", "Gradient norm",
                 f"Gradient norm per step on a log scale (raw faint, EMA solid). The grad-clip value of "
                 f"{D['grad_clip']:.1f} from the run config is drawn for reference."))

    # 05 throughput --------------------------------------------------------------------------------------------
    M2 = M[M.index >= 2]
    s2 = M2.index.to_numpy()
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 6.6))
    a = axes.ravel()
    series(a[0], s2, M2["tok_s"].to_numpy(), BLUE)
    style_ax(a[0], "Throughput", ylabel="tokens per second")
    series(a[1], s2, M2["mfu"].to_numpy(), BLUE)
    style_ax(a[1], "Model FLOP utilization", ylabel="MFU (fraction of peak)")
    series(a[2], s2, M2["update_s"].to_numpy(), BLUE)
    style_ax(a[2], "Seconds per optimizer step", ylabel="seconds")
    win = (D["slow_runs"][0]["start"], D["slow_runs"][-1]["end"]) if D["slow_runs"] else None
    if win:
        for ax_ in a[:4]:
            shade(ax_, [win])
    series(a[3], s2, M2["sparse_s"].to_numpy(), BLUE, label="sparse_s", logy=True)
    series(a[3], s2, M2["dense_opt_s"].to_numpy(), ORANGE, label="dense_opt_s", logy=True)
    a[3].set_yscale("log")
    style_ax(a[3], "Optimizer time split (log scale)", ylabel="seconds")
    a[3].legend(loc="center", fontsize=7.8)
    for ax in a:
        step_axis(ax, N)
    if win:
        n_slow = sum(r["steps"] for r in D["slow_runs"])
        a[2].text(0.02, 0.96, f"shaded: {n_slow} steps with update_s above 1.5x the median, "
                  f"steps {fmt_int(win[0])} to {fmt_int(win[1])}", transform=a[2].transAxes, ha="left", va="top",
                  fontsize=7.4, color=INK2)
    finish(fig, out_dir, "05_throughput.png", "Throughput and step timing",
           "Step 1 excluded: its update_s is 552 s (later steps ~30 s); the logs do not state why.")
    figs.append(("05_throughput.png", "Throughput and step timing",
                 f"Tokens/s, MFU and seconds per step (step 1 excluded: 552 s first update). Shaded: steps with update_s "
                 f"above 1.5x the median ({D['update_median']:.1f} s)."))

    # 06 n-gram memory -----------------------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.6))
    a = axes.ravel()
    specs = [("ngram/table_rms", "n-gram table RMS", "RMS", False),
             ("ngram/rows_touched_frac", "rows touched per step", "fraction of table rows", False),
             ("ngram/grad_rms", "n-gram gradient RMS (log)", "RMS", True),
             ("ngram/update_rms", "n-gram update RMS (log)", "RMS", True),
             ("ngram/update_to_weight", "update-to-weight ratio (log)", "ratio", True),
             ("ngram/lr", "n-gram learning rate (log)", "learning rate", True)]
    for ax, (key, title, ylab, logy) in zip(a, specs):
        series(ax, st, M[key].to_numpy(), BLUE, logy=logy)
        if logy:
            ax.set_yscale("log")
        style_ax(ax, title, ylabel=ylab)
        step_axis(ax, N)
    zeros = {k: int((M[k] == 0).sum()) for k, *_ in specs if k in M}
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.suptitle("n-gram memory table", x=0.012, y=0.995, ha="left", va="top", fontsize=12, color=INK, fontweight="bold")
    fig.text(0.012, 0.952, f"Log panels omit zero values (step 1 is exactly zero for gradient, update and update-to-weight).",
             ha="left", va="top", fontsize=8.5, color=INK2)
    mark_all(fig, MARKS)
    fig.savefig(os.path.join(out_dir, "06_ngram_memory.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)
    figs.append(("06_ngram_memory.png", "n-gram memory table",
                 f"Table RMS grows from {M['ngram/table_rms'].iloc[0]:.3f} to {M['ngram/table_rms'].iloc[-1]:.3f}. "
                 f"Update RMS and update-to-weight ratio follow the learning-rate decay after step "
                 f"{D['lr_fit']['s0']:,.0f}; gradient RMS falls from its early peak."))

    # 07 n-gram timings ---------------------------------------------------------------------------------------
    keys = [k for k in M.columns if k.startswith("ngram_ms/")]
    keys = sorted(keys)
    fig, axes = plt.subplots(3, 3, figsize=(12, 8.2), sharex=True)
    log_timers = {"ngram_ms/cpu_gather", "ngram_ms/cpu_hash", "ngram_ms/input_d2h_wall", "ngram_ms/worker_final_join"}
    for ax, key in zip(axes.ravel(), keys):
        series(ax, st, M[key].to_numpy(), BLUE, logy=(key in log_timers))
        if key in log_timers:
            ax.set_yscale("log")
        style_ax(ax, key.replace("ngram_ms/", ""), ylabel="ms")
        step_axis(ax, N)
        ax.set_xlabel("optimizer step" if key in keys[-3:] else "")
        ax.tick_params(labelbottom=(key in keys[-3:]))
    for ax in axes.ravel()[len(keys):]:
        ax.set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.suptitle("n-gram timings per step", x=0.012, y=0.995, ha="left", va="top", fontsize=12, color=INK,
                 fontweight="bold")
    fig.text(0.012, 0.952, "Milliseconds per step, one panel per timer (faint raw, solid EMA); log scale where a timer "
             "spans decades. Spikes in cpu_hash and input_d2h_wall fall on the step after each eval pass.",
             ha="left", va="top", fontsize=8.2, color=INK2)
    mark_all(fig, MARKS)
    fig.savefig(os.path.join(out_dir, "07_ngram_timings.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)
    figs.append(("07_ngram_timings.png", "n-gram timings",
                 "All nine ngram_ms timers, one panel each. Spikes in the CPU hash and input-transfer timers fall on the "
                 "step right after each eval pass."))

    # 08 memory ------------------------------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10.5, 4.3))
    series(ax, st, M["peak_alloc_gib"].to_numpy(), BLUE, label="peak allocated", raw=False, smooth=False, lw=1.5)
    series(ax, st, M["peak_reserved_gib"].to_numpy(), ORANGE, label="peak reserved", raw=False, smooth=False, lw=1.5)
    series(ax, st, M["device_used_gib"].to_numpy(), AQUA, label="device used", raw=False, smooth=False, lw=1.5)
    if D["gpu_total_gib"]:
        ax.axhline(D["gpu_total_gib"], color=INK2, lw=0.8)
        ax.annotate(f"device total {D['gpu_total_gib']:.0f} GiB (run metadata)", xy=(N, D["gpu_total_gib"]),
                    xytext=(-4, 3), textcoords="offset points", ha="right", fontsize=7.6, color=INK2)
    step_axis(ax, N)
    style_ax(ax, "Peak and current GPU memory", ylabel="GiB")
    ax.legend(loc="lower right", fontsize=7.8)
    ax.set_ylim(40, (D["gpu_total_gib"] or 52) * 1.04)
    finish(fig, out_dir, "08_memory.png", "GPU memory",
           "Peak allocated and reserved memory and device usage, as logged by the trainer.")
    figs.append(("08_memory.png", "GPU memory",
                 f"Peak allocated {M['peak_alloc_gib'].min():.1f} to {M['peak_alloc_gib'].max():.1f} GiB, device used "
                 f"{M['device_used_gib'].max():.1f} GiB at most, against the device total in the run metadata."))

    # 09 progress and layer drop -------------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 6.4))
    a = axes.ravel()
    a[0].plot(M["progress/train_hours"].to_numpy(), M["tokens_seen"].to_numpy() / 1e9, color=BLUE, lw=1.6)
    a[0].set_xlim(left=0)
    style_ax(a[0], "Tokens seen vs training time", xlabel="training hours", ylabel="tokens (billions)")
    a[0].annotate(f"{M['tokens_seen'].iloc[-1] / 1e9:.2f} B at {M['progress/train_hours'].iloc[-1]:.1f} h",
                  xy=(M["progress/train_hours"].iloc[-1], M["tokens_seen"].iloc[-1] / 1e9), xytext=(-8, 10),
                  textcoords="offset points", ha="right", va="bottom", fontsize=7.6, color=INK2)
    eta = M[M.index >= 2]
    series(a[1], eta.index.to_numpy(), eta["progress/eta_h"].to_numpy(), BLUE, raw=False, smooth=False, lw=1.6)
    a[1].set_ylim(0, 300)
    a[1].annotate("step 1 omitted: its ETA (2,692 h) follows its 552 s first update", xy=(0.16, 0.75),
                  xycoords="axes fraction", fontsize=7.4, color=INK2, va="top")
    style_ax(a[1], "Estimated time remaining", ylabel="hours")
    series(a[2], st, M["layer_drop/skipped_frac"].to_numpy(), BLUE)
    style_ax(a[2], "Layer-drop: skipped fraction", ylabel="fraction of layers skipped")
    series(a[3], st, M["progress/epoch"].to_numpy(), BLUE, raw=False, smooth=False, lw=1.6)
    style_ax(a[3], "Data epochs completed", ylabel="epochs")
    for ax in a[1:]:
        step_axis(ax, N)
    finish(fig, out_dir, "09_progress_layerdrop.png", "Progress and layer drop",
           "Token count against training hours, ETA and layer-drop schedule per step.")
    figs.append(("09_progress_layerdrop.png", "Progress and layer drop",
                 f"Tokens seen against training hours ({M['progress/train_hours'].iloc[-1]:.1f} h total), ETA, "
                 f"layer-drop fraction (decreasing schedule) and data epochs."))

    # 10 checkpoints and hub ----------------------------------------------------------------------------------
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10.5, 6.2), gridspec_kw={"height_ratios": [1.3, 1]}, sharex=True)
    cs = D["ckpt_steps"]
    w = [D["ckpt"][s]["seconds"] for s in cs]
    sl = [D["stall"][s]["seconds"] for s in cs]
    per, fin = cs[:-1], cs[-1]
    a1.plot(per, w[:-1], color=BLUE, lw=1.5, zorder=3, label="checkpoint write time (s)")
    a1.scatter(per, w[:-1], s=36, color=BLUE, zorder=4, edgecolors=SURFACE, linewidths=1.2)
    a1.plot(per, sl[:-1], color=ORANGE, lw=1.5, zorder=3, label="training stall at checkpoint (s)")
    a1.scatter(per, sl[:-1], s=36, color=ORANGE, zorder=4, edgecolors=SURFACE, linewidths=1.2)
    a1.scatter([fin], [w[-1]], s=44, facecolors=SURFACE, edgecolors=BLUE, linewidths=1.6, zorder=5)
    a1.scatter([fin], [sl[-1]], s=44, facecolors=SURFACE, edgecolors=ORANGE, linewidths=1.6, zorder=5)
    a1.annotate(f"final checkpoint (hollow): {D['stall'][fin]['seconds']:.1f} s blocking stall", xy=(fin, sl[-1]),
                xytext=(-8, 9), textcoords="offset points", ha="right", fontsize=7.6, color=INK2)
    a1.set_ylim(0, max(w + sl) * 1.25)
    style_ax(a1, "Checkpoint write and stall time", ylabel="seconds")
    a1.legend(loc="upper left", fontsize=7.8)
    ok_steps = [s for s in cs if s in D["hub_ok"]]
    fail_steps = [s for s in cs if s not in D["hub_ok"]]
    a2.scatter(fail_steps, [0] * len(fail_steps), marker="X", s=60, color=CRIT, zorder=4, edgecolors=SURFACE, linewidths=0.8,
               label=f"failed after 6 attempts ({len(fail_steps)})")
    a2.scatter(ok_steps, [0] * len(ok_steps), marker="o", s=60, color=GOOD, zorder=4, edgecolors=SURFACE, linewidths=0.8,
               label=f"uploaded ({len(ok_steps)})")
    for s in ok_steps:
        a2.annotate(f"uploaded, attempt {D['hub_ok'][s].get('attempts')}", xy=(s, 0), xytext=(0, 9),
                    textcoords="offset points", ha="center", fontsize=7.6, color=INK2)
    a2.set_yticks([])
    a2.set_ylim(-0.6, 0.9)
    a2.grid(False)
    style_ax(a2, "Hub upload outcome per checkpoint")
    a2.legend(loc="upper left", fontsize=7.8, ncol=2)
    step_axis(a2, N)
    finish(fig, out_dir, "10_checkpoints_hub.png", "Checkpoints and hub uploads",
           "Every checkpoint was written locally; the Hugging Face upload failed for all but one.")
    figs.append(("10_checkpoints_hub.png", "Checkpoints and hub uploads",
                 f"Write and stall time per checkpoint ({len(cs)} checkpoints, every {D['ckpt_every']} steps plus the final "
                 f"one). Hub: {len(ok_steps)} uploaded, {len(fail_steps)} failed with the private-storage limit error."))

    # 11 system (from the .wandb file) -------------------------------------------------------------------------
    if D["sys"] is not None:
        sysd = D["sys"]
        bins = sysd["bins"]
        xs_step = sysd["step_of_t"](np.array([b["t"] for b in bins]))
        def col(key, agg="mean", scale=1.0):
            return np.array([b[agg].get(key, np.nan) for b in bins]) * scale
        fig, axes = plt.subplots(2, 3, figsize=(12.2, 6.8))
        a = axes.ravel()
        series(a[0], xs_step, col("gpu.0.gpu"), BLUE, raw=False, smooth=False, lw=1.2)
        a[0].plot(xs_step, col("gpu.0.gpu", "max"), color=BLUE, lw=0.6, alpha=0.5)
        style_ax(a[0], "GPU utilization (10-min mean; faint: max)", ylabel="percent")
        series(a[1], xs_step, col("gpu.0.temp"), ORANGE, raw=False, smooth=False, lw=1.2)
        a[1].plot(xs_step, col("gpu.0.temp", "max"), color=ORANGE, lw=0.6, alpha=0.5)
        style_ax(a[1], "GPU temperature (10-min mean; faint: max)", ylabel="degrees C")
        series(a[2], xs_step, col("gpu.0.powerWatts"), BLUE, raw=False, smooth=False, lw=1.2)
        if sysd.get("power_limit_w"):
            a[2].axhline(sysd["power_limit_w"], color=INK2, lw=0.8)
            a[2].annotate(f"enforced limit {sysd['power_limit_w']:.0f} W", xy=(N, sysd["power_limit_w"]), xytext=(-4, 3),
                          textcoords="offset points", ha="right", fontsize=7.6, color=INK2)
        style_ax(a[2], "GPU power draw (10-min mean)", ylabel="watts")
        series(a[3], xs_step, col("gpu.0.memoryAllocatedBytes", scale=1 / 2**30), BLUE, raw=False, smooth=False, lw=1.2)
        style_ax(a[3], "GPU memory allocated (10-min mean)", ylabel="GiB")
        series(a[4], xs_step, col("gpu.0.smClock"), BLUE, raw=False, smooth=False, lw=1.2)
        a[4].plot(xs_step, col("gpu.0.smClock", "max"), color=BLUE, lw=0.6, alpha=0.5)
        style_ax(a[4], "GPU SM clock (10-min mean; faint: max)", ylabel="MHz")
        series(a[5], xs_step, col("cpu"), BLUE, raw=False, smooth=False, lw=1.2)
        style_ax(a[5], "Host CPU utilization (10-min mean)", ylabel="percent")
        for ax in a:
            step_axis(ax, N)
            if D["wb_fatal_step"] is not None:
                ax.axvline(D["wb_fatal_step"], color=CRIT, lw=0.8, zorder=1)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.suptitle("GPU and host system metrics", x=0.012, y=0.995, ha="left", va="top", fontsize=12, color=INK,
                     fontweight="bold")
        fig.text(0.012, 0.952, "From the local .wandb file, 10-minute bins mapped to trainer steps. Dark vertical line: W&B sync "
                 f"failure at step {fmt_int(D['wb_fatal_step'] or 0)}.", ha="left", va="top", fontsize=8.2, color=INK2)
        mark_all(fig, MARKS)
        fig.savefig(os.path.join(out_dir, "11_system_gpu.png"), dpi=150, facecolor=SURFACE)
        plt.close(fig)
        figs.append(("11_system_gpu.png", "GPU and host system metrics",
                     f"GPU utilization, temperature, power, memory, SM clock and host CPU from the local .wandb file, "
                     f"{sysd['n_samples']:,} samples over {sysd['span_h']:.1f} h, mapped to trainer steps."))
    return figs


def slow_runs(update_s, steps):
    med = float(np.median(update_s))
    slow = [s for s, u in zip(steps, update_s) if u > 1.5 * med]
    runs, cur = [], []
    for s in slow:
        if cur and s - cur[-1] > 5:
            runs.append(cur)
            cur = []
        cur.append(s)
    if cur:
        runs.append(cur)
    out = []
    for r in runs:
        vals = [float(update_s[steps.index(s)]) for s in r]
        out.append({"start": int(r[0]), "end": int(r[-1]), "steps": len(r), "max_update_s": max(vals)})
    return med, out


def main(argv):
    ap = argparse.ArgumentParser(description="Rebuild training-run plots offline from local logs.")
    ap.add_argument("data_dir", nargs="?", default=DEFAULT_DATA)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--wandb-python", default=DEFAULT_WANDB_PY)
    ap.add_argument("--no-system", action="store_true")
    args = ap.parse_args(argv)
    if plt is None:
        sys.exit("numpy, pandas and matplotlib are required: run with /home/user/wspacetest/venv/bin/python -I")
    D_ = os.path.abspath(args.data_dir)
    P = {
        "metrics": os.path.join(D_, "metrics(1).jsonl"),
        "train": os.path.join(D_, "train(1).log"),
        "output": os.path.join(D_, "latest-run", "files", "output.log"),
        "summary": os.path.join(D_, "latest-run", "files", "wandb-summary.json"),
        "meta": os.path.join(D_, "latest-run", "files", "wandb-metadata.json"),
        "results": os.path.join(D_, "results.jsonl"),
        "dbg": os.path.join(D_, "latest-run", "logs", "debug-internal.log"),
    }
    for k, p in P.items():
        if not os.path.exists(p):
            sys.exit(f"missing input ({k}): {p}")
    os.makedirs(args.out, exist_ok=True)
    wb_files = sorted(glob.glob(os.path.join(D_, "latest-run", "run-*.wandb")))

    print("reading metrics ...")
    M, minfo = load_metrics(P["metrics"])
    N = int(M.index.max())
    tlog = parse_train_log(P["train"])
    olog = parse_train_log(P["output"])
    summary = json.load(open(P["summary"], encoding="utf-8"))
    meta = json.load(open(P["meta"], encoding="utf-8"))
    results = json.loads(open(P["results"], encoding="utf-8").read().strip().splitlines()[0])
    dbg = parse_debug_internal(P["dbg"])
    try:
        gpu_total_gib = round(float(meta["gpu_nvidia"][0]["memoryTotal"]) / 2**30, 1)
    except (KeyError, IndexError, TypeError, ValueError):
        gpu_total_gib = None

    # ---- cross-checks ------------------------------------------------------------------------------------
    checks = {}
    checks["output_log_structured_lines_identical"] = tlog["struct_seq"] == olog["struct_seq"]
    checks["output_log_structured_line_count"] = len(tlog["struct_seq"])
    tl = tlog["steps_text"]
    d_loss = [abs(tl[s]["loss"] - M.loc[s, "loss"]) for s in tl if s in M.index]
    checks["text_vs_metrics_loss_max_abs_diff"] = max(d_loss)  # text prints 5 significant digits
    checks["text_steps_matched"] = len(d_loss)
    losses = results.get("losses", [])
    checks["results_losses_len"] = len(losses)
    checks["results_losses_max_abs_diff"] = max(abs(losses[i] - M.loc[i + 1, "loss"]) for i in range(min(len(losses), N)))
    checks["results_mean_mfu_excl_step1"] = float(M.loc[M.index >= 2, "mfu"].mean())
    checks["results_mean_mfu"] = results.get("mean_mfu")
    checks["summary_train_loss_matches_last_step"] = abs(summary["train/loss"] - M.loc[N, "loss"]) < 1e-9
    checks["summary_last_step"] = summary.get("step")
    checks["summary_eval_loss_full_precision"] = summary.get("eval/loss")

    # ---- evals ---------------------------------------------------------------------------------------------
    sched_m = M[M["eval_loss"].notna()]
    evals = []
    for s, r in sched_m.iterrows():
        evals.append({"step": int(s), "kind": "scheduled", "eval_loss": float(r["eval_loss"]),
                      "eval_ppl": float(r["eval_ppl"]), "eval_loss_no_ngram": float(r["eval_loss_no_ngram"]),
                      "eval_ngram_gain": float(r["eval_ngram_gain"]),
                      "eval_loss_early_exit": float(r["eval_loss_early_exit"]),
                      "ngram_table_rms": float(r["ngram_table_rms"]), "eval_s": float(r["eval_s"]),
                      "source": "metrics(1).jsonl", "log_line": int(r["_line"])})
    final_t = [e for e in tlog["evals"] if e["kind"] == "final"]
    if final_t:
        fv = final_t[-1]["values"]
        evals.append({"step": int(N), "kind": "final", "eval_loss": summary["eval/loss"],
                      "eval_ppl": summary["eval/ppl"], "eval_loss_no_ngram": summary["eval/loss_no_ngram"],
                      "eval_ngram_gain": summary["eval/ngram_gain"],
                      "eval_loss_early_exit": summary["eval/loss_early_exit"],
                      "ngram_table_rms": summary["eval/ngram_table_rms"], "eval_s": summary["eval/s"],
                      "source": "wandb-summary.json (matches train(1).log final EVAL line to 4 decimals)",
                      "log_line": final_t[-1]["line"]})
        checks["final_eval_matches_text_to_4dp"] = all(
            abs(round(evals[-1][k], 4) - fv[t]) < 1e-9 for k, t in
            [("eval_loss", "eval_loss"), ("eval_loss_no_ngram", "eval_loss_no_ngram"),
             ("eval_loss_early_exit", "eval_loss_early_exit"), ("eval_ngram_gain", "eval_ngram_gain")])
    checks["final_minus_step17540_eval_loss"] = (float(evals[-1]["eval_loss"] - float(M.loc[sched_m.index.max(), "eval_loss"]))
                                                 if final_t else None)
    checks["eval_ppl_equals_exp_loss_max_rel_err"] = max(abs(math.exp(e["eval_loss"]) - e["eval_ppl"]) / e["eval_ppl"] for e in evals)
    checks["eval_gain_equals_no_ngram_minus_loss_max_abs_err"] = max(abs(e["eval_loss_no_ngram"] - e["eval_loss"] - e["eval_ngram_gain"]) for e in evals)

    # ---- trainer attempts / restarts ----------------------------------------------------------------------
    attempts = tlog["attempts"]
    startup_crashes = []
    for a in attempts:
        steps_here = a["steps"]
        a["step_min"] = min(steps_here) if steps_here else None
        a["step_max"] = max(steps_here) if steps_here else None
        a["steps_logged"] = len(steps_here)
        if (a["exit_code"] not in (None, 0) or a["traceback_line"] is not None) and not steps_here:
            startup_crashes.append({"x": 0, "attempt": a["index"], "time": a["start_time"],
                                    "traceback_line": a["traceback_line"], "error_line": a["error_line"],
                                    "error": a["error"], "exit_code": a["exit_code"]})
    run_attempts = [a for a in attempts if a["steps_logged"] > 0]
    checkpoints = sorted(tlog["ckpt"])
    cfg = run_attempts[-1]["config"] if run_attempts else {}
    tokens_per_step = float(cfg.get("update_tokens", 589824))
    wb_fatal = tlog["wb_fatal"]
    wb_fatal_step = int(wb_fatal["after_step"]) if wb_fatal and wb_fatal["after_step"] is not None else None
    hub_ok = tlog["hub_ok"]
    hub_fail = tlog["hub_fail"]
    hub_uploaded_step = min(hub_ok) if hub_ok else None
    hub_err_kinds = collections.Counter()
    for s, rows_ in tlog["hub_retry"].items():
        for r in rows_:
            hub_err_kinds[classify_hub_error(r["error"])] += 1
    fail_err = [classify_hub_error(r.get("error", "")) for r in hub_fail.values()]

    # ---- W&B history and system stats ------------------------------------------------------------------------
    wb = None
    wb_note = None
    if args.no_system:
        wb_note = "skipped (--no-system)"
    elif not wb_files:
        wb_note = "skipped: no latest-run/run-*.wandb file found"
    elif not os.path.exists(args.wandb_python):
        wb_note = f"skipped: {args.wandb_python} not found (system panels need the wandb venv)"
    else:
        try:
            wb = run_wandb_extract(args.wandb_python, wb_files[0])
        except Exception as e:  # noqa: BLE001 - reported in summary.json
            wb_note = f"skipped: {str(e)[:300]}"
    wb_checks = {}
    t_pts = []
    sys_info = None
    if wb is not None:
        hist = wb["history_rows"]
        step_rows = [(r["v"]["step"], r["v"]["_timestamp"], r["v"].get("train/loss")) for r in hist
                     if "step" in r["v"] and "_timestamp" in r["v"] and "train/loss" in r["v"]]
        step_rows.sort(key=lambda x: x[0])
        ws = [s for s, _, _ in step_rows]
        wb_steps = sorted(set(ws))
        wb_checks["history_rows_total"] = len(hist)
        wb_checks["history_rows_with_train_step"] = len(step_rows)
        wb_checks["history_distinct_train_steps"] = len(wb_steps)
        wb_checks["history_first_last_train_step"] = [wb_steps[0], wb_steps[-1]] if wb_steps else None
        wb_checks["history_steps_missing_from_1_to_N"] = len(set(range(1, N + 1)) - set(wb_steps))
        wb_checks["history_steps_duplicated"] = len(ws) - len(set(ws))
        diffs = [abs(loss - M.loc[s, "loss"]) for s, _, loss in step_rows if loss is not None and s in M.index]
        wb_checks["history_loss_max_abs_diff_vs_metrics"] = max(diffs) if diffs else None
        ev_w = [(r["v"]["step"], r["v"]["eval/loss"]) for r in hist if "eval/loss" in r["v"] and "step" in r["v"]]
        wb_checks["history_eval_rows"] = [{"step": int(s), "eval_loss": v} for s, v in ev_w]
        ts_sorted = [t for _, t, _ in sorted(step_rows, key=lambda x: x[0])]
        by_step = sorted(((s, t) for s, t, _ in step_rows), key=lambda x: x[0])
        dt_pairs = [(s2, t2 - t1) for (s1, t1), (s2, t2) in zip(by_step, by_step[1:])]
        top_gaps = sorted(dt_pairs, key=lambda x: -x[1])[:8]
        wb_checks["step_time_gaps_top"] = [{"to_step": int(s), "gap_s": round(g, 1)} for s, g in top_gaps]
        wb_checks["step_time_monotonic"] = all(b >= a for a, b in zip(ts_sorted, ts_sorted[1:]))
        wb_checks["local_history_complete_through_last_step"] = bool(
            wb_checks["history_distinct_train_steps"] == N and wb_checks["history_steps_missing_from_1_to_N"] == 0
            and wb_checks["history_loss_max_abs_diff_vs_metrics"] is not None
            and wb_checks["history_loss_max_abs_diff_vs_metrics"] < 1e-12)
        by_step_t = sorted(((s, t) for s, t, _ in step_rows), key=lambda x: x[0])
        tt_arr = np.maximum.accumulate(np.array([t for _, t in by_step_t]))
        ss_arr = np.array([s for s, _ in by_step_t], dtype=float)
        if dbg.get("http403_first"):
            ts_txt = re.sub(r"(\.\d{6})\d*Z$", r"\1+00:00", dbg["http403_first"])
            t403 = dt.datetime.fromisoformat(ts_txt).timestamp()
            wb_checks["trainer_step_at_first_403"] = round(float(np.interp(t403, tt_arr, ss_arr)), 1)
        wb_checks["stats_samples"] = wb["stats_n"]
        wb_checks["stats_span_hours"] = round((wb["stats_t1"] - wb["stats_t0"]) / 3600, 2)
        wb_checks["stats_max_sampling_gap_s"] = round(wb["stats_top_gaps"][0]["gap_s"], 1)
        wb_checks["record_counts"] = wb["record_counts"]
        wb_checks["parse_errors"] = wb["parse_errors"]
        if step_rows:
            pts_s = np.array(sorted(set(ws)), dtype=float)
            # map time -> step by the first logged time of each step (monotone)
            first_t = {}
            for s, t, _ in sorted(step_rows, key=lambda x: x[0]):
                first_t.setdefault(s, t)
            T = np.maximum.accumulate(np.array([first_t[int(s)] for s in pts_s]))
            Sx = pts_s
            def step_of_t(tarr, T=T, Sx=Sx):
                return np.interp(tarr, T, Sx)
            bins = wb["stats_bins"]
            p_lim = wb["stats_global"].get("gpu.0.powerWatts", {})
            pw_lim = wb["stats_global"].get("gpu.0.enforcedPowerLimitWatts", {}).get("mean")
            mem_total = None
            try:
                mem_total = float(meta["gpu_nvidia"][0]["memoryTotal"]) / 2**30
            except (KeyError, IndexError, TypeError, ValueError):
                mem_total = None
            glob_s = wb["stats_global"]
            sys_info = {"bins": bins, "step_of_t": step_of_t, "power_limit_w": pw_lim,
                        "n_samples": wb["stats_n"], "span_h": (wb["stats_t1"] - wb["stats_t0"]) / 3600,
                        "gpu_mem_total_gib": mem_total,
                        "glob": {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in glob_s.items()}}
            # GPU clock / temperature / throttling summary over the run
            sys_info["gpu_summary"] = {
                "util_mean_pct": glob_s.get("gpu.0.gpu", {}).get("mean"),
                "temp_max_c": glob_s.get("gpu.0.temp", {}).get("max"),
                "temp_mean_c": glob_s.get("gpu.0.temp", {}).get("mean"),
                "power_mean_w": glob_s.get("gpu.0.powerWatts", {}).get("mean"),
                "power_max_w": glob_s.get("gpu.0.powerWatts", {}).get("max"),
                "smclock_min_mhz": glob_s.get("gpu.0.smClock", {}).get("min"),
                "smclock_mean_mhz": glob_s.get("gpu.0.smClock", {}).get("mean"),
                "corrected_mem_errors_max": glob_s.get("gpu.0.correctedMemoryErrors", {}).get("max"),
                "uncorrected_mem_errors_max": glob_s.get("gpu.0.uncorrectedMemoryErrors", {}).get("max"),
                "oom_kills_max": glob_s.get("proc.memory.oomKills", {}).get("max"),
            }
    # ---- derived series -----------------------------------------------------------------------------------
    med_update, slow = slow_runs(M.loc[M.index >= 2, "update_s"].to_numpy(), [int(s) for s in M.index if s >= 2])
    if sys_info is not None and slow:
        lo, hi = slow[0]["start"], slow[-1]["end"]
        bstep = sys_info["step_of_t"](np.array([b["t"] for b in sys_info["bins"]]))
        inside = [b for b, s_ in zip(sys_info["bins"], bstep) if lo <= s_ <= hi and b["n"] > 0]
        outside = [b for b, s_ in zip(sys_info["bins"], bstep) if not (lo <= s_ <= hi) and b["n"] > 0]

        def _avg(bs, k):
            vals = [b["mean"][k] for b in bs if k in b["mean"]]
            return round(float(np.mean(vals)), 2) if vals else None
        sys_info["slow_window"] = {
            "steps": [int(lo), int(hi)], "bins_inside": len(inside), "bins_outside": len(outside),
            "gpu_util_pct_mean": {"inside": _avg(inside, "gpu.0.gpu"), "outside": _avg(outside, "gpu.0.gpu")},
            "gpu_temp_c_mean": {"inside": _avg(inside, "gpu.0.temp"), "outside": _avg(outside, "gpu.0.temp")},
            "sm_clock_mhz_mean": {"inside": _avg(inside, "gpu.0.smClock"), "outside": _avg(outside, "gpu.0.smClock")},
            "power_w_mean": {"inside": _avg(inside, "gpu.0.powerWatts"), "outside": _avg(outside, "gpu.0.powerWatts")}}
    post_eval = []
    for key in ("ngram_ms/cpu_hash", "ngram_ms/input_d2h_wall"):
        vals = M[key].to_numpy()
        med_v = float(np.median(vals))
        for e in evals:
            s_after = e["step"] + 1
            if s_after in M.index:
                v = float(M.loc[s_after, key])
                if v > 10 * med_v:
                    post_eval.append({"key": key, "eval_step": e["step"], "step": s_after, "value_ms": round(v, 1),
                                      "median_ms": round(med_v, 3)})
    post_note = [f"{p['key'].replace('ngram_ms/', '')} {p['value_ms'] / 1000:.1f} s at step {p['step']}"
                 for p in post_eval if p["key"] == "ngram_ms/cpu_hash"][:4]
    warm = 88
    if "--warmup-steps" in meta.get("args", []):
        warm = int(meta["args"][meta["args"].index("--warmup-steps") + 1])
    post = M.index.to_numpy() > warm
    lr_fit_s0, lr_fit_err = fit_wsd_decay(M.index.to_numpy(dtype=float)[post], M["lr_mult"].to_numpy()[post], float(N))
    warm_err = float(np.max(np.abs(M.loc[M.index <= 88, "lr_mult"].to_numpy() - np.minimum(1.0, M.index[M.index <= 88].to_numpy() / 88.0))))
    lr_groups = {
        "lr_vs_nordion2_max_rel": float(np.max(np.abs(M["lr"] - M["lr/nordion2"]) / np.maximum(M["lr"].abs(), 1e-30))),
        "lr_vs_ngram_lr_max_rel": float(np.max(np.abs(M["lr"] - M["ngram/lr"]) / np.maximum(M["lr"].abs(), 1e-30))),
        "adamw_vs_adamw2_max_rel": float(np.max(np.abs(M["lr/adamw"] - M["lr/adamw2"]) / np.maximum(M["lr/adamw"].abs(), 1e-30))),
        "adamw_vs_adamw3_max_rel": float(np.max(np.abs(M["lr/adamw"] - M["lr/adamw3"]) / np.maximum(M["lr/adamw"].abs(), 1e-30))),
    }
    peak = {"dion_group": float(M["lr"].max()), "adamw_groups": float(M["lr/adamw"].max())}
    cfg_args = {}
    args_list = meta.get("args", [])
    i = 0
    while i < len(args_list):
        a_ = args_list[i]
        if a_.startswith("--"):
            key = a_[2:]
            val = True
            if i + 1 < len(args_list) and not args_list[i + 1].startswith("--"):
                val = args_list[i + 1]
                i += 1
            if key in ("wandb-project", "wandb-entity", "wandb-name", "hub-repo", "data", "eval-data", "tokenizer",
                       "ckpt-dir", "metrics", "output", "run-name"):
                i += 1
                continue
            if key == "model-override":
                cfg_args.setdefault("model_override", []).append(val)
            else:
                cfg_args[key] = val
        i += 1

    # ---- derived overview numbers ---------------------------------------------------------------------------
    last100 = float(M["loss"].iloc[-100:].mean())
    final = {
        "steps": N,
        "tokens": float(M.loc[N, "tokens_seen"]),
        "train_hours": float(M.loc[N, "progress/train_hours"]),
        "train_loss_last_step": float(M.loc[N, "loss"]),
        "train_loss_mean_last_100_steps": last100,
        "eval_loss": evals[-1]["eval_loss"],
        "eval_ppl": evals[-1]["eval_ppl"],
        "eval_loss_no_ngram": evals[-1]["eval_loss_no_ngram"],
        "ngram_gain_nats": evals[-1]["eval_ngram_gain"],
        "eval_loss_early_exit": evals[-1]["eval_loss_early_exit"],
        "mean_mfu_excl_step1": float(results.get("mean_mfu", M.loc[M.index >= 2, "mfu"].mean())),
        "tokens_per_second_mean_excl_step1": float(results.get("tokens_per_second")),
        "median_update_s_excl_step1": float(results.get("median_update_s")),
        "final_lr": float(M.loc[N, "lr"]),
    }
    ev_struct = {"evals": evals}
    D = {"M": M, "N": N, "events": ev_struct, "tokens_per_step": tokens_per_step,
         "startup_crashes": startup_crashes, "wb_fatal_step": wb_fatal_step, "hub_ok": hub_ok,
         "hub_uploaded_step": hub_uploaded_step, "ckpt_steps": checkpoints, "ckpt": tlog["ckpt"],
         "stall": tlog["stall"], "ckpt_every": cfg_args.get("ckpt-every", "877"),
         "lr_fit": {"s0": lr_fit_s0, "err": lr_fit_err}, "warmup_steps": int(float(cfg_args.get("warmup-steps", 88))),
         "grad_clip": float(cfg_args.get("grad-clip", 1.0)), "slow_runs": slow, "update_median": med_update,
         "post_eval_note": post_note, "sys": sys_info, "gpu_total_gib": gpu_total_gib,
         "wb_local_complete": bool(wb_checks.get("local_history_complete_through_last_step"))}
    plot_events = []
    for c in startup_crashes:
        plot_events.append({"x": 0, "label": f"{len(startup_crashes)} startup crashes", "ha": "left"})
    if wb_fatal_step is not None:
        plot_events.append({"x": wb_fatal_step, "label": f"W&B sync 403 after step {fmt_int(wb_fatal_step)}",
                            "ha": "right", "row": 1})
    D["plot_events"] = plot_events
    D["hub_ok"] = hub_ok

    print("drawing figures ...")
    setup_style()
    figs = build_figures(args.out, D)

    # ---- outputs -------------------------------------------------------------------------------------------
    restarts = []
    for a in attempts:
        restarts.append({"attempt": a["index"], "supervisor_label": a["supervisor_label"],
                         "start_time": a["start_time"], "start_line": a["start_line"],
                         "exit_code": a["exit_code"], "exit_line": a["exit_line"],
                         "traceback_line": a["traceback_line"], "error_line": a["error_line"],
                         "error": a["error"], "steps_logged": a["steps_logged"],
                         "step_range": [a["step_min"], a["step_max"]] if a["steps_logged"] else None,
                         "resumed_from_checkpoint": False, "wandb_run_resumed": a["resumed"],
                         "outcome": ("crashed before any step"
                                     if ((a["exit_code"] not in (None, 0) or a["traceback_line"] is not None)
                                         and not a["steps_logged"])
                                     else ("completed all steps" if a["steps_logged"] else "no steps"))})
    ckpt_rows = []
    for s in checkpoints:
        c = tlog["ckpt"][s]
        st_ = tlog["stall"].get(s, {})
        ok = tlog["hub_ok"].get(s)
        ckpt_rows.append({"step": s, "write_s": c.get("seconds"), "gb": c.get("gb"), "stall_s": st_.get("seconds"),
                          "blocking": st_.get("blocking"),
                          "hub": ("uploaded" if ok else "failed"),
                          "hub_attempts": (ok or {}).get("attempts") if ok else 6,
                          "hub_error": (classify_hub_error(hub_fail[s]["error"]) if s in hub_fail else None),
                          "log_line": c.get("line")})
    hub_summary = {"checkpoints": len(checkpoints), "uploaded": len(hub_ok), "failed": len(hub_fail),
                   "uploaded_steps": sorted(hub_ok), "failed_steps": sorted(hub_fail),
                   "uploaded_detail": ({k: {kk: v.get(kk) for kk in ("gb", "seconds", "attempts", "verified", "line")}
                                        for k, v in hub_ok.items()}),
                   "error_classes": dict(collections.Counter(fail_err))}
    wb_block = {"decoded": wb is not None, "note": wb_note, "checks": wb_checks,
                "fatal_log_line": (wb_fatal or {}).get("line"), "fatal_after_step": wb_fatal_step,
                "debug_internal_first_fatal": dbg["fatal"][0] if dbg["fatal"] else None,
                "debug_internal_403_first_utc": dbg["http403_first"], "debug_internal_403_last_utc": dbg["http403_last"],
                "debug_internal_403_count": dbg["http403_count"], "debug_internal_retry_by_status": dbg["retry_by_status"],
                "export_403_lines_in_train_log": tlog["counts"].get("export_403", 0),
                "export_timeout_lines": tlog["counts"].get("export_timeout", 0),
                "transient_unavailable_lines": tlog["counts"].get("transient_unavailable", 0),
                "http_500_lines": tlog["counts"].get("http_500", 0),
                "synced_line": tlog["synced"]}
    if wb is not None and sys_info is None:
        wb_note = "decoded, but no per-step rows with timestamps were found"
        wb_block["note"] = wb_note
    if sys_info is not None:
        wb_block["gpu_summary"] = sys_info["gpu_summary"]
        wb_block["slow_window_gpu"] = sys_info.get("slow_window")
    obs = []
    if slow:
        n_slow = sum(r["steps"] for r in slow)
        obs.append(f"update_s exceeds 1.5x the median ({1.5 * med_update:.1f} s) on {n_slow} steps in {len(slow)} runs, "
                   f"all between steps {slow[0]['start']:,} and {slow[-1]['end']:,} (max "
                   f"{max(r['max_update_s'] for r in slow):.1f} s). The text logs give no cause for this window.")
    sw = (sys_info or {}).get("slow_window")
    if sw:
        obs.append(f"Inside that window GPU utilization averaged {sw['gpu_util_pct_mean']['inside']:.1f}% vs "
                   f"{sw['gpu_util_pct_mean']['outside']:.1f}% outside, power {sw['power_w_mean']['inside']:.0f} W vs "
                   f"{sw['power_w_mean']['outside']:.0f} W, and SM clock {sw['sm_clock_mhz_mean']['inside']:.0f} MHz vs "
                   f"{sw['sm_clock_mhz_mean']['outside']:.0f} MHz (10-minute bins). The clock barely moved, so the data "
                   "do not point to clock throttling; the GPU looks idle more often.")
    n_sched = sum(1 for e in evals if e["kind"] == "scheduled")
    pe_cpu = [p for p in post_eval if p["key"] == "ngram_ms/cpu_hash"]
    pe_io = [p for p in post_eval if p["key"] == "ngram_ms/input_d2h_wall"]
    if pe_cpu:
        obs.append(f"ngram_ms/cpu_hash exceeds 10x its median on the step right after {len(pe_cpu)} of {n_sched} "
                   f"scheduled eval passes (e.g. step {pe_cpu[0]['step']:,}: {pe_cpu[0]['value_ms'] / 1000:.1f} s vs "
                   f"median {pe_cpu[0]['median_ms'] / 1000:.2f} s); input_d2h_wall shows the same pattern after "
                   f"{len(pe_io)} of them. See 07_ngram_timings.png.")
    obs.append(f"lr/adamw* groups peak at {peak['adamw_groups']:g}; config has --adamw-lr 0.0003 and --scalar-lr 0.003. "
               "The logs do not say which is intended.")
    obs.append("The W&B _step counter (summary: 17,609) runs ahead of the trainer step (17,542), so W&B's own x-axis "
               "is not the trainer step; these figures use trainer step.")
    summary_out = {
        "generated_by": "experiments/plot_training_run.py (offline; no network, no W&B sync)",
        "data_dir": D_,
        "run": {"name": results.get("run"), "gpu": meta.get("gpu"),
                "gpu_memory_total_gib": gpu_total_gib,
                "python": meta.get("python"), "cuda": meta.get("cudaVersion"),
                "torch": (results or {}).get("torch"), "preset": results.get("preset"),
                "dense_params": results.get("dense_total"), "non_embedding_params": results.get("non_embedding"),
                "ngram_table_entries": results.get("ngram_table"), "total_params": results.get("all"),
                "update_tokens_per_step": tokens_per_step, "grad_accum": results.get("grad_accum"),
                "warmup_steps": D["warmup_steps"], "decay_fraction": float(cfg_args.get("decay-frac", 0.295)),
                "eval_every": int(cfg_args.get("eval-every", 3508)), "ckpt_every": int(cfg_args.get("ckpt-every", 877)),
                "args": cfg_args, "started_at_utc": meta.get("startedAt")},
        "final": final,
        "evals": evals,
        "restarts_and_failures": restarts,
        "startup_crashes": startup_crashes,
        "wandb_sync": wb_block,
        "hub": hub_summary,
        "checkpoints": ckpt_rows,
        "data_quality": {
            "metrics_path_records": minfo["records"], "metrics_unique_steps": minfo["unique_steps"],
            "step_range": [minfo["step_min"], minfo["step_max"]], "duplicate_steps": minfo["duplicate_step_count"],
            "regressions": minfo["regression_count"], "missing_steps": minfo["missing_step_count"],
            "nonfinite_values": minfo["nonfinite_values"], "bad_lines": minfo["bad_lines"],
            "trainer_launches_with_steps": len(run_attempts), "trainer_launches_total": len(attempts),
            "resume_from_checkpoint_seen": any(((a["config"] or {}).get("start_step", 0) or 0) != 0 for a in attempts),
            "checks": checks,
        },
        "lr_schedule": {"warmup_max_abs_err": warm_err, "decay_fit_s0": lr_fit_s0, "decay_fit_max_abs_err": lr_fit_err,
                        "peak": peak, "group_identity_max_rel": lr_groups},
        "throughput": {"slow_step_runs": slow, "update_median_s_excl_step1": med_update,
                       "post_eval_spikes": post_eval},
        "observations": obs,
        "figures": [{"file": f, "title": t, "caption": c} for f, t, c in figs],
        "not_rebuilt": [
            "The online W&B dashboard after the 403 cannot be inspected offline; only the local copies were read.",
            "W&B's internal _step axis is not used (trainer step is used everywhere).",
            "Hub upload durations are only logged for the successful upload (342.6 s); failed attempts have no timing.",
            "The three failed startups logged no steps, so there is no per-step data for them.",
            "The tst field is 0 at every step and is not plotted.",
        ],
    }
    if wb is None:
        summary_out["not_rebuilt"].append(f"System/GPU panels skipped: {wb_note}")
    summary_out["run"]["wandb_extract_note"] = wb_note
    summary_out["supervisor_gave_up_line"] = tlog["give_up_line"]
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(to_py(summary_out), f, indent=2)
    write_index(args.out, figs, summary_out)
    print("wrote", os.path.join(args.out, "summary.json"), "and index.html")
    print("figures:", len(figs))
    for f, _, _ in figs:
        print("  ", os.path.join(args.out, f))


def write_index(out_dir, figs, s):
    f_ = s["final"]
    rows = [
        ("Steps logged (last step)", f"{f_['steps']:,}"),
        ("Tokens seen", f"{f_['tokens']:,.0f} ({f_['tokens'] / 1e9:.2f} B)"),
        ("Training time (progress/train_hours)", f"{f_['train_hours']:.2f} h"),
        ("Final train loss (last step)", f"{f_['train_loss_last_step']:.4f}"),
        ("Train loss, mean of last 100 steps", f"{f_['train_loss_mean_last_100_steps']:.4f}"),
        ("Final eval loss (n-gram on)", f"{f_['eval_loss']:.4f}"),
        ("Final eval perplexity", f"{f_['eval_ppl']:.3f}"),
        ("Eval loss without n-gram", f"{f_['eval_loss_no_ngram']:.4f}"),
        ("N-gram gain (nats/token)", f"{f_['ngram_gain_nats']:.4f}"),
        ("Early-exit eval loss", f"{f_['eval_loss_early_exit']:.4f}"),
        ("Mean MFU (excluding step 1)", f"{f_['mean_mfu_excl_step1']:.4f}"),
        ("Tokens per second, mean (results.jsonl)", f"{f_['tokens_per_second_mean_excl_step1']:,.0f}"),
        ("Median seconds per step (excluding step 1)", f"{f_['median_update_s_excl_step1']:.2f}"),
    ]
    eval_rows = "".join(
        f"<tr><td>{e['step']:,}</td><td>{'final' if e['kind'] == 'final' else 'scheduled'}</td>"
        f"<td>{e['eval_loss']:.4f}</td><td>{e['eval_ppl']:.3f}</td><td>{e['eval_loss_no_ngram']:.4f}</td>"
        f"<td>{e['eval_ngram_gain']:.4f}</td><td>{e['eval_loss_early_exit']:.4f}</td>"
        f"<td>{e['ngram_table_rms']:.4f}</td></tr>" for e in s["evals"])
    summary_rows = "".join(f"<tr><th scope='row'>{htmllib.escape(k)}</th><td>{htmllib.escape(v)}</td></tr>" for k, v in rows)
    fig_html = "".join(
        f"<figure><img src='{htmllib.escape(f)}' alt='{htmllib.escape(t)}' loading='lazy'>"
        f"<figcaption><strong>{htmllib.escape(t)}.</strong> {htmllib.escape(c)}</figcaption></figure>"
        for f, t, c in figs)
    def _attempt_item(r):
        parts = [f"<strong>{htmllib.escape(r['outcome'])}</strong>: launch {r['attempt'] + 1} started "
                 f"{htmllib.escape(str(r['start_time']))} UTC (trainer log line {r['start_line']})"]
        if r["exit_code"] is not None:
            parts.append(f"exit code {r['exit_code']}")
        elif r["error"]:
            parts.append("no exit line; the supervisor then gave up"
                         + (f" (log line {s['supervisor_gave_up_line']})" if s.get("supervisor_gave_up_line") else ""))
        parts.append(f"steps logged {r['steps_logged']}")
        if r["step_range"]:
            parts.append(f"steps {r['step_range'][0]:,} to {r['step_range'][1]:,}")
        if r["error"]:
            parts.append(f"traceback at line {r['traceback_line']}: <code>{htmllib.escape(r['error'])}</code>")
        return "<li>" + "; ".join(parts) + "</li>"
    fail_rows = "".join(_attempt_item(r) for r in s["restarts_and_failures"])
    html_doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Training run dashboard (offline rebuild)</title>
<style>
:root {{ color-scheme: light; --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --line:#e1e0d9; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ color-scheme: dark; --page:#0d0d0d; --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --muted:#898781; --line:#2c2c2a; }} }}
:root[data-theme="dark"] {{ color-scheme: dark; --page:#0d0d0d; --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --muted:#898781; --line:#2c2c2a; }}
* {{ box-sizing: border-box; }}
body {{ margin:0; background:var(--page); color:var(--ink); font-family: system-ui, -apple-system, "Segoe UI", sans-serif; line-height:1.45; }}
main {{ max-width: 1120px; margin: 0 auto; padding: 24px 16px 48px; }}
h1 {{ font-size: 1.5rem; margin: 0 0 6px; }}
h2 {{ font-size: 1.1rem; margin: 30px 0 10px; }}
p.lead {{ color: var(--ink2); margin: 0 0 14px; }}
table {{ border-collapse: collapse; width: 100%; background: var(--surface); }}
th, td {{ text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--line); font-size: .92rem; }}
th[scope=row] {{ font-weight: 500; color: var(--ink2); width: 52%; }}
td {{ font-variant-numeric: tabular-nums; }}
.grid {{ display: grid; gap: 18px; }}
figure {{ margin: 0; background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 10px; }}
figure img {{ width: 100%; height: auto; display: block; border-radius: 4px; }}
figcaption {{ color: var(--ink2); font-size: .9rem; margin-top: 8px; }}
ul {{ padding-left: 20px; color: var(--ink2); }}
code {{ font-size: .85em; }}
@media (max-width: 600px) {{ main {{ padding: 16px; }} th[scope=row] {{ width: 60%; }} }}
</style></head>
<body><main>
<h1>Training run: offline rebuild of the dashboard plots</h1>
<p class="lead">Built from the local logs of the packed1-500m-deep run (trainer stdout, metrics file, W&amp;B transaction log)
by <code>experiments/plot_training_run.py</code>. Nothing was re-synced to W&amp;B. Figures are static PNGs.
Restart and failure markers are thin vertical lines on the step plots.</p>
<h2>Final summary</h2>
<table><tbody>{summary_rows}</tbody></table>
<h2>Eval passes (table view of every eval point)</h2>
<table><thead><tr><th>Step</th><th>Kind</th><th>Eval loss</th><th>Perplexity</th><th>Loss, n-gram off</th>
<th>N-gram gain</th><th>Early-exit loss</th><th>Table RMS</th></tr></thead><tbody>{eval_rows}</tbody></table>
<h2>Restarts and failures found in the logs</h2>
<ul>{fail_rows}</ul>
<p class="lead">W&amp;B online sync: fatal error after step {fmt_int(s['wandb_sync']['fatal_after_step'] or 0)} (train log line
{s['wandb_sync']['fatal_log_line']}); filestream returned 403 Forbidden. The local transaction log was still written.</p>
<h2>Figures</h2>
<div class="grid">{fig_html}</div>
</main></body></html>
"""
    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(html_doc)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--extract-wandb":
        json.dump(extract_wandb(sys.argv[2]), sys.stdout)
        sys.exit(0)
    main(sys.argv[1:])
