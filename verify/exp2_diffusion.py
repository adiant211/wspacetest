"""Verify Experiment 2: what do the three released diffusion models generate, and can a plain tool read it from
the weights the way Astra did?

  1. Re-sample: run DDPM generation (exactly the schedule in networks/description.txt) for 1,000 samples per model,
     sort them with a small rule-based shape classifier (bounding-box fill ratio), and check they are FILLED shapes
     (the post says Astra drew outlines while the models draw filled shapes). The classifier is validated on
     synthetic filled circles / squares / triangles first.
  2. Astra's drawings: parse the `Example:` block of each answer.md and measure how filled it is.
  3. Weight-only reading (H2): the last layer W3 (64 x 144) holds one 12 x 12 "stroke" per hidden unit. Pool the
     edge orientations of those strokes (Sobel gradients, magnitude-weighted). Squares only have 0/90-degree edges,
     circles have every orientation, triangles add two slanted sides. No forward pass, no LLM.

Usage: python verify/exp2_diffusion.py
"""
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import OUT, logs  # noqa: E402

IDS = {"circle": "d88aba69", "square": "a6a9dfb7", "triangle": "246a5b58"}
T = 1000
betas = np.linspace(1e-4, 0.02, T)
alphas = 1 - betas
abar = np.cumprod(alphas)
freqs = 1 / 10000 ** (np.arange(8) / 8)


def net(Wd, x, t):
    te = np.concatenate([np.sin(t * freqs), np.cos(t * freqs)])
    inp = np.concatenate([x, np.broadcast_to(te, (x.shape[0], 16))], 1)
    h1 = np.maximum(inp @ Wd["W1"] + Wd["b1"], 0)
    h2 = np.maximum(h1 @ Wd["W2"] + Wd["b2"], 0)
    return h2 @ Wd["W3"] + Wd["b3"]


def sample(Wd, n, rng):
    x = rng.standard_normal((n, 144))
    for t in range(T - 1, -1, -1):
        x0 = np.clip(net(Wd, x, t), -1, 1)
        if t == 0:
            return x0
        ab, abp = abar[t], abar[t - 1]
        mean = (np.sqrt(abp) * betas[t] / (1 - ab)) * x0 + (np.sqrt(alphas[t]) * (1 - abp) / (1 - ab)) * x
        var = betas[t] * (1 - abp) / (1 - ab)
        x = mean + np.sqrt(var) * rng.standard_normal(x.shape)


def classify(img):
    """img: (12,12) in [-1,1]. Returns (label, bbox_fill, interior_fill)."""
    m = img > 0
    if m.sum() < 3:
        return "empty", 0.0, 0.0
    rows, cols = np.where(m)
    h, w = rows.max() - rows.min() + 1, cols.max() - cols.min() + 1
    fill = m.sum() / (h * w)
    # interior: pixels whose 4 neighbours are all on, among pixels inside the bbox shrunk by one
    inner = m[1:-1, 1:-1] & m[:-2, 1:-1] & m[2:, 1:-1] & m[1:-1, :-2] & m[1:-1, 2:]
    interior = inner.sum() / max(1, m.sum())
    lab = "square" if fill > 0.88 else ("circle" if fill > 0.66 else "triangle")
    return lab, float(fill), float(interior)


def synth(kind, rng):
    img = -np.ones((12, 12))
    yy, xx = np.mgrid[0:12, 0:12]
    s = rng.integers(5, 11)
    r0, c0 = rng.integers(0, 12 - s + 1), rng.integers(0, 12 - s + 1)
    if kind == "square":
        img[r0:r0 + s, c0:c0 + s] = 1
    elif kind == "circle":
        cy, cx, rad = r0 + (s - 1) / 2, c0 + (s - 1) / 2, s / 2
        img[(yy - cy) ** 2 + (xx - cx) ** 2 <= rad ** 2] = 1
    else:  # upward isosceles triangle in the s x s box
        for i in range(s):
            half = (i + 1) * s / (2 * s)
            cc = c0 + (s - 1) / 2
            img[r0 + i, (xx[0] >= cc - half) & (xx[0] <= cc + half)] = 1
    return img


def orientation_hist(W3, bins=12):
    """Magnitude-weighted histogram of edge orientations (0..180 deg) over all 12x12 strokes in W3's rows."""
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float)
    ky = kx.T
    hist = np.zeros(bins)
    for row in W3:
        im = row.reshape(12, 12)
        p = np.pad(im, 1, mode="edge")
        gx = sum(kx[i, j] * p[i:i + 12, j:j + 12] for i in range(3) for j in range(3))
        gy = sum(ky[i, j] * p[i:i + 12, j:j + 12] for i in range(3) for j in range(3))
        mag = np.hypot(gx, gy)
        ang = (np.degrees(np.arctan2(gy, gx)) + 180) % 180  # gradient direction; edge is perpendicular
        hist += np.histogram(ang, bins=bins, range=(0, 180), weights=mag)[0]
    return hist / hist.sum()


def axis_share(hist):
    """Share of edge energy within +-15 deg of horizontal/vertical gradient directions."""
    centers = (np.arange(len(hist)) + 0.5) * 180 / len(hist)
    near = (np.minimum(centers % 90, 90 - centers % 90) <= 15)
    return float(hist[near].sum())


def templates(rng, n=400):
    """Orientation histograms of synthetic filled shapes: what a stroke layer drawing each shape should look like."""
    return {k: orientation_hist(np.stack([synth(k, rng).reshape(-1) for _ in range(n)])) for k in IDS}


def read_shape(hist, tmpl):
    """Weight-only reading: the template whose orientation profile correlates best with the layer's (uniform removed)."""
    u = 1 / len(hist)
    sc = {k: float(np.corrcoef(hist - u, t - u)[0, 1]) for k, t in tmpl.items()}
    return max(sc, key=sc.get), sc


def ascii_fill(block):
    lines = [l for l in block.strip().splitlines() if l.strip()]
    g = np.array([[ch not in ". " for ch in l[:12].ljust(12, ".")] for l in lines[:12]])
    return g


def main():
    rng = np.random.default_rng(0)
    rep = {}
    # classifier sanity check on synthetic filled shapes
    acc = {}
    for k in IDS:
        labs = [classify(synth(k, rng))[0] for _ in range(300)]
        acc[k] = sum(l == k for l in labs) / len(labs)
    rep["classifier_accuracy_on_synthetic_filled_shapes"] = acc
    print("classifier on synthetic shapes:", acc, flush=True)
    tmpl = templates(rng)
    for shape, mid in IDS.items():
        z = np.load(logs(f"exp2_diffusion/networks/{mid}.npz"))
        Wd = {k: z[k].astype(np.float64) for k in ["W1", "b1", "W2", "b2", "W3", "b3"]}
        xs = sample(Wd, 1000, rng)
        cls = [classify(x.reshape(12, 12)) for x in xs]
        labs = [c[0] for c in cls]
        frac = {k: labs.count(k) / len(labs) for k in ["circle", "square", "triangle", "empty"]}
        interior = float(np.mean([c[2] for c in cls if c[0] != "empty"]))
        # Astra's drawing
        ans = open(logs(f"exp2_diffusion/runs/{shape}/answer.md")).read()
        blk = re.search(r"Example:?\**\s*```(?:text)?\n(.*?)```", ans, flags=re.S)
        astra = ascii_fill(blk.group(1)) if blk else None
        astra_int = None
        if astra is not None and astra.sum():
            a = astra
            inner = a[1:-1, 1:-1] & a[:-2, 1:-1] & a[2:, 1:-1] & a[1:-1, :-2] & a[1:-1, 2:]
            astra_int = float(inner.sum() / a.sum())
        oh = orientation_hist(Wd["W3"])
        guess, scores = read_shape(oh, tmpl)
        rep[shape] = {"model_id": mid, "sample_classes": frac, "mean_interior_share": interior,
                      "reported_classes": json.load(open(logs("exp2_diffusion/eval/answer_key.json")))[mid]["quality"]["classified_as"],
                      "astra_drawing_interior_share": astra_int,
                      "W3_edge_orientation_hist": [round(float(v), 4) for v in oh], "W3_axis_aligned_share": axis_share(oh),
                      "weight_only_reading": guess, "weight_only_template_corr": scores}
        print(f"{shape:8s} samples→ {{c:{frac['circle']:.3f} s:{frac['square']:.3f} t:{frac['triangle']:.3f}}} "
              f"interior share {interior:.2f} (filled shapes ≈0.4-0.6, outlines ≈0) | Astra drawing interior "
              f"{astra_int if astra_int is None else round(astra_int, 2)} | weight-only reading of W3: {guess} {scores}", flush=True)
    json.dump(rep, open(os.path.join(OUT, "exp2_diffusion.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
