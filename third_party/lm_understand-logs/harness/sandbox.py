"""The writer's Python sandbox for the from-scratch Go experiment (bubblewrap, as scripts/mathedit/sandbox.py):
  /work  read-write, kept between runs: the network (net/<name>.npy), pass results (pass_<k>/), the writer's files
  numpy, scipy, scikit-learn, PyTorch (CPU), pandas, sympy, networkx (SANDBOX_VENV); no pgx, no network, no GPU
No network access, a clean environment, TIMEOUT seconds and MEM bytes per run, output cut at MAX_OUT characters.
PRELUDE: SHAPES, NAMES, load(name) (float32 array: the saved values, zeros at the start), save(name, array), saved(),
inp(row, col, plane) (input index), mv(row, col) (output index), PASS = 81.
"""
import json, os, resource, subprocess, time

SANDBOX_VENV = "<venvs>/sandbox_full"
TIMEOUT, MAX_OUT, MEM = 900, 20_000, 96 << 30

PRELUDE = r'''
import json, os, numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=160)
WORK = "/work"
_A = json.load(open(WORK + "/arch.json"))
DEPTH, WIDTH = _A["depth"], _A["width"]
SHAPES = {}
for _k in range(1, DEPTH + 1):
    SHAPES[f"L{_k}.W"] = (243 if _k == 1 else WIDTH, WIDTH)
    SHAPES[f"L{_k}.b"] = (WIDTH,)
SHAPES["out.W"], SHAPES["out.b"] = (WIDTH, 82), (82,)
NAMES = list(SHAPES)
PASS = 81
def inp(row, col, plane):
    return (row * 9 + col) * 3 + plane
def mv(row, col):
    return row * 9 + col
def load(name):
    p = WORK + "/net/" + name + ".npy"
    return np.load(p).astype(np.float32) if os.path.exists(p) else np.zeros(SHAPES[name], np.float32)
def save(name, a):
    assert name in SHAPES, f"unknown tensor {name}; NAMES = {NAMES}"
    a = np.asarray(a, np.float32)
    assert a.shape == tuple(SHAPES[name]), (name, a.shape, SHAPES[name])
    assert np.isfinite(a).all(), name
    os.makedirs(WORK + "/net", exist_ok=True)
    np.save(WORK + "/net/" + name + ".npy", a)
def saved():
    d = WORK + "/net"
    return sorted(f[:-4] for f in os.listdir(d)) if os.path.isdir(d) else []
'''


def limits():
    resource.setrlimit(resource.RLIMIT_AS, (MEM, MEM))


def run(code, work_dir, log):
    cmd = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session", "--clearenv",
           "--ro-bind", "/usr", "/usr", "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
           "--symlink", "usr/bin", "/bin", "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
           "--ro-bind", SANDBOX_VENV, SANDBOX_VENV, "--bind", work_dir, "/work", "--chdir", "/work",
           "--setenv", "HOME", "/tmp", "--setenv", "PATH", "/usr/bin", "--setenv", "OMP_NUM_THREADS", "8",
           "--setenv", "CUDA_VISIBLE_DEVICES", "",
           f"{SANDBOX_VENV}/bin/python", "-I", "-c", PRELUDE + "\n" + code]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT, preexec_fn=limits)
        out = p.stdout + (("\n[stderr]\n" + p.stderr[-6000:]) if p.stderr.strip() else "")
    except subprocess.TimeoutExpired:
        out = f"[timed out after {TIMEOUT} s]"
    if len(out) > MAX_OUT:
        out = out[:MAX_OUT] + f"\n[output cut at {MAX_OUT:,} characters]"
    with open(log, "a") as f:
        f.write(json.dumps({"t": time.time(), "kind": "python", "s": round(time.time() - t0, 2), "code": code,
                            "out": out[:6000]}) + "\n")
    return out or "(no output)"
