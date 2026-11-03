"""Package a math edit as a small patch file, and apply it to the original checkpoint.

  python experiments/math_apply_edit.py pack  <edit .pt/.json> results/math/patch.safetensors
  python experiments/math_apply_edit.py apply results/math/patch.safetensors <original ckpt dir> <output dir>

pack   stores, per edited layer, the neuron indices and either the column scales or the rank-r factors of the column
       update (W_down[:, idx] += B @ A; factored exactly by SVD, since a LoRA update has rank <= r). A rank-8 update on
       4,096 neurons is ~1.5 MB instead of 12.6 MB of merged deltas.
apply  writes <output dir>/model.safetensors with the edit baked in and copies or links everything else (config.json,
       patch.py, tokenizer, ngram.safetensors), so the result loads exactly like the original with
       AutoModelForCausalLM.from_pretrained(<output dir>, trust_remote_code=True).
"""
import os
import sys

import torch
from safetensors.torch import load_file, save_file

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from wsp.neuron_edit import load_edit, write_checkpoint  # noqa: E402


def pack(src, dst, max_rank=8):
    e = load_edit(src)
    out, meta = {}, {"source": os.path.relpath(src, REPO)}
    for l, d in e["layers"].items():
        out[f"{l}.idx"] = torch.as_tensor(d["idx"], dtype=torch.int64)
        if "scale" in d:
            out[f"{l}.scale"] = torch.as_tensor(d["scale"], dtype=torch.float32)
        if "delta" in d:
            D = torch.as_tensor(d["delta"], dtype=torch.float64)
            U, S, Vh = torch.linalg.svd(D, full_matrices=False)
            r = int(min(max_rank, (S > S[0] * 1e-7).sum()))
            B, A = (U[:, :r] * S[:r]).float(), Vh[:r].float()
            err = float((B.double() @ A.double() - D).abs().max())
            assert err < 1e-5 * max(float(D.abs().max()), 1e-12) + 1e-8, f"layer {l}: rank-{r} factorization off by {err}"
            out[f"{l}.B"], out[f"{l}.A"] = B.contiguous(), A.contiguous()
    save_file(out, dst, metadata=meta)
    print(f"wrote {dst}: {len(e['layers'])} layers, {os.path.getsize(dst) / 1e6:.2f} MB")


def unpack(path):
    t = load_file(path)
    layers = {}
    for k, v in t.items():
        l, field = k.split(".")
        layers.setdefault(int(l), {})[field] = v
    edit = {"layers": {}}
    for l, d in layers.items():
        e = {"idx": d["idx"]}
        if "scale" in d:
            e["scale"] = d["scale"]
        if "B" in d:
            e["delta"] = d["B"] @ d["A"]
        edit["layers"][l] = e
    return edit


def main():
    cmd = sys.argv[1]
    if cmd == "pack":
        pack(sys.argv[2], sys.argv[3])
    elif cmd == "apply":
        write_checkpoint(sys.argv[3], unpack(sys.argv[2]), sys.argv[4])
        print(f"wrote {sys.argv[4]}/model.safetensors")
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
