"""Static figures for the README (results/figures/*.png), drawn from the result JSONs.

Palette: the dataviz reference categorical order (blue, orange, aqua) for identity, a warm gray for controls and
baselines; thin marks, recessive grid, one y-axis per chart, legends whenever there are two or more series.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(REPO, "results")
FIG = os.path.join(R, "figures")
os.makedirs(FIG, exist_ok=True)
BLUE, ORANGE, AQUA, GRAY, INK, INK2, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29c", "#0b0b0b", "#52514e", "#e7e6e2"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRAY, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white",
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False})


def load(p):
    p = os.path.join(R, p)
    return json.load(open(p)) if os.path.exists(p) else None


def fig_verify():
    e5, e6, e8 = load("verify/exp5_minatar.json"), load("verify/exp6_go_h2h.json"), load("verify/exp8_qwen35_edit.json")
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), gridspec_kw={"width_ratios": [1.1, 1.1, 1]})
    if e5:
        games = list(e5)
        y = np.arange(len(games))
        rep = [e5[g]["reported"]["mean"] for g in games]
        ours = [e5[g]["ours"]["mean"] for g in games]
        rse = [1.96 * e5[g]["reported"]["se"] for g in games]
        ose = [1.96 * e5[g]["ours"]["se"] for g in games]
        ppo = [max(v for k, v in e5[g]["ppo_best_reported"].items() if k.startswith("ppo100")) for g in games]
        a = ax[0]
        a.barh(y + 0.2, rep, 0.36, xerr=rse, color=ORANGE, label="Astra, reported", error_kw={"lw": 1, "ecolor": INK2})
        a.barh(y - 0.2, ours, 0.36, xerr=ose, color=BLUE, label="Astra, our re-score", error_kw={"lw": 1, "ecolor": INK2})
        a.scatter(ppo, y, marker="|", s=300, color=INK, label="best PPO, 100M steps", zorder=3)
        a.set_xscale("log")
        a.set_yticks(y, [g.capitalize() for g in games])
        a.set_xlabel("mean score over 1,000 episodes (log scale)")
        a.set_title("Exp 5: MinAtar networks written from scratch", loc="left", fontsize=10, color=INK)
        a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=8)
    if e6:
        keys = list(e6)
        y = np.arange(len(keys))
        a = ax[1]
        rep = [e6[k]["reported_score"] * 100 for k in keys]
        ours = [e6[k]["score"] * 100 for k in keys]
        ci = [196 * np.sqrt(e6[k]["score"] * (1 - e6[k]["score"]) / e6[k]["games"]) for k in keys]
        a.scatter(rep, y + 0.12, color=ORANGE, s=40, label="reported", zorder=3)
        a.errorbar(ours, y - 0.12, xerr=ci, fmt="o", color=BLUE, ms=6, lw=1, label="our replay (95% CI)", zorder=3)
        a.axvline(50, color=GRAY, lw=1, ls="--")
        a.set_yticks(y, [k.replace("budget_", "").replace("_vs_ppo_s", " passes vs PPO s") for k in keys])
        a.set_xlabel("win rate of Astra's network, 2,000 games (%)")
        a.set_title("Exp 6: Go head-to-head vs same-size PPO", loc="left", fontsize=10, color=INK)
        a.legend(loc="lower right", fontsize=8)
    if e8 and "teacher" in e8 and e8["teacher"].get("per_sample"):
        ps = e8["teacher"]["per_sample"]
        x = np.array([r["productive_edit"] - r["productive_base"] for r in ps])
        yv = np.array([r["loop_edit"] - r["loop_base"] for r in ps])
        a = ax[2]
        lo = min(x.min(), yv.min(), 0) * 1.1
        a.plot([lo, 0.01], [lo, 0.01], color=GRAY, lw=1, ls="--", label="equal change")
        a.scatter(x, yv, color=BLUE, s=36, edgecolor="white", linewidth=1, label="one looping answer", zorder=3)
        a.set_xlim(lo, 0.01)
        a.set_ylim(lo, 0.01)
        a.set_xlabel("Δ log-prob, reasoning before loop")
        a.set_ylabel("Δ log-prob, loop (nats/token)")
        a.set_title("Exp 8: the edit hits loops ~2x harder", loc="left", fontsize=10, color=INK)
        a.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.24)
    fig.savefig(os.path.join(FIG, "verify.png"), dpi=150)
    plt.close(fig)


def fig_where():
    pa, pf, sp, idl = load("h1/picks_add.json"), load("h1/picks_french.json"), load("h1/spectrum.json"), load("h1/instruct_delta.json")
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.4))
    L = np.arange(24)
    a = ax[0]
    if pa:
        a.plot(L, np.array(pa["grad_share_by_layer"]) * 100, color=BLUE, lw=2, marker="o", ms=4, label="5-digit addition")
    if pf:
        a.plot(L, np.array(pf["grad_share_by_layer"]) * 100, color=ORANGE, lw=2, marker="o", ms=4, label="answer in French")
    a.set_xlabel("layer")
    a.set_ylabel("share of task-gradient energy (%)")
    a.set_title("Where the task-gradient tool says each purpose lives", loc="left", fontsize=10, color=INK)
    a.set_xticks(range(0, 24, 2))
    a.legend(fontsize=8)
    a = ax[1]
    if sp:
        a.bar(L, sp["layer"], color=AQUA, width=0.7, label="Spectrum-style SNR rank")
    a.set_xlabel("layer")
    a.set_ylabel("mean SNR percentile, 7 module types")
    a.set_title("Weight-only, task-agnostic: Spectrum-style SNR", loc="left", fontsize=10, color=INK)
    a.set_xticks(range(0, 24, 2))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "where.png"), dpi=150)
    plt.close(fig)


def place_labels(fig, ax, items):
    """Greedy label placement: for each point try a few offsets and keep the first whose text box overlaps neither
    an earlier label nor any marker."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    marks = [ax.transData.transform((x, y)) for x, y, _ in items]
    boxes = []
    offsets = [(7, 3, "left"), (7, -11, "left"), (-7, 3, "right"), (-7, -11, "right"), (7, 13, "left"), (-7, 13, "right"),
               (7, -21, "left"), (-7, -21, "right")]
    for x, y, text in sorted(items, key=lambda t: -t[1]):
        for dx, dy, ha in offsets:
            t = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=8, color=INK2, ha=ha)
            bb = t.get_window_extent(renderer=r).expanded(1.05, 1.15)
            hit = any(bb.overlaps(b) for b in boxes) or any(bb.contains(mx, my) for mx, my in marks)
            if not hit or (dx, dy, ha) == offsets[-1]:
                boxes.append(bb)
                break
            t.remove()


SHORT = {"full": "all 24 layers", "full_r1": "all, r=1", "pick4": "tool", "pick4r": "tool (relative)",
         "pick4u": "tool (upper half)", "spectrum4": "Spectrum", "arb4": "arbitrary", "rand4a": "random a",
         "rand4b": "random b", "anti4": "tool's lowest-ranked", "neur": "tool neurons", "neur_rand": "random neurons"}


def fig_tradeoff():
    s, cost = load("h1/summary.json"), load("h1/cost.json")
    if not s or not cost:
        return
    c = {(r["purpose"], r["arm"]): r for r in cost}
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4))
    panels = [("add", "acc_5digit", "5-digit addition accuracy (%)", "H1, addition: accuracy vs seconds per step"),
              ("french", "capitals_acc", "capitals still answered correctly (%)",
               "H1, French: retention vs seconds per step (all arms ~100% French)")]
    groups = {"tool": (BLUE, "tool-picked"), "full": (ORANGE, "all 24 layers"), "control": (GRAY, "random / Spectrum / tool's lowest")}
    for a, (purpose, key, ylab, title) in zip(ax, panels):
        base = next((r for r in s[purpose]["arms"] if r["arm"] == "base"), None)
        if base and base.get(key) is not None:
            a.axhline(base[key] * 100, color=GRAY, lw=1, ls="--", label="no fine-tune")
        pts = []
        for r in s[purpose]["arms"]:
            if r["arm"] == "base" or (purpose, r["arm"]) not in c or r.get(key) is None:
                continue
            g = "tool" if r["arm"] in ("pick4", "pick4r", "pick4u", "neur") else "full" if r["arm"].startswith("full") else "control"
            pts.append((c[(purpose, r["arm"])]["s_per_step"], r[key] * 100, r["arm"], g))
        seen = set()
        for x, y, arm, g in pts:
            col, name = groups[g]
            a.scatter(x, y, color=col, s=55, edgecolor="white", linewidth=1.5, zorder=3, label=None if g in seen else name)
            seen.add(g)
        place_labels(fig, a, [(x, y, SHORT.get(arm, arm)) for x, y, arm, g in pts])
        a.set_xlabel("seconds per training step (float32, idle 4-core CPU)")
        a.set_ylabel(ylab)
        a.set_title(title, loc="left", fontsize=10, color=INK)
        a.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=4)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.25)
    fig.savefig(os.path.join(FIG, "h1_tradeoff.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fig_verify()
    fig_where()
    fig_tradeoff()
    print(sorted(os.listdir(FIG)))
