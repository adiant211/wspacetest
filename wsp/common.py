"""Shared helpers: loading Qwen2.5-0.5B-Instruct on CPU, chat formatting, batched greedy generation,
and a manual forward pass through a *suffix* of the decoder stack (used for frozen-prefix caching).

Everything here runs on CPU. bfloat16 is the default dtype because the target CPU has AMX-bf16, which
made a 1,024-token forward pass ~5x faster than float32 in our benchmark (0.77 s vs 3.97 s).
"""
from __future__ import annotations

import os
import random
import time
from contextlib import contextmanager

import numpy as np
import torch

os.environ.setdefault("HF_HOME", "/home/user/hf")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from transformers.utils import logging as hf_logging  # noqa: E402

hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()

INSTRUCT = "Qwen/Qwen2.5-0.5B-Instruct"
BASE = "Qwen/Qwen2.5-0.5B"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, "results")

LINEAR_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
ATTN_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP_MODULES = ["gate_proj", "up_proj", "down_proj"]


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_tokenizer(model_id: str = INSTRUCT):
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    return tok


def load_model(model_id: str = INSTRUCT, dtype=torch.bfloat16):
    m = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def module_name(layer: int, mod: str) -> str:
    sub = "self_attn" if mod in ATTN_MODULES else "mlp"
    return f"model.layers.{layer}.{sub}.{mod}"


def get_linear(model, layer: int, mod: str):
    blk = core(model).layers[layer]
    return getattr(blk.self_attn if mod in ATTN_MODULES else blk.mlp, mod)


def core(model):
    """Return the Qwen2Model (decoder stack) whether or not the model is PEFT-wrapped."""
    m = model
    for _ in range(4):
        if hasattr(m, "layers") and hasattr(m, "norm"):
            return m
        if hasattr(m, "base_model") and m.base_model is not m:
            m = m.base_model
        elif hasattr(m, "model"):
            m = m.model
    raise ValueError("could not find decoder stack")


def lm_head(model):
    m = model
    for _ in range(4):
        if hasattr(m, "lm_head"):
            return m.lm_head
        m = m.base_model if hasattr(m, "base_model") and m.base_model is not m else m.model
    raise ValueError("no lm_head")


def chat(tok, user: str, system: str | None = None) -> str:
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


@torch.no_grad()
def generate(model, tok, prompts: list[str], max_new_tokens: int = 32, bs: int = 32) -> list[str]:
    """Greedy, batched, left-padded generation. `prompts` are already chat-formatted strings."""
    tok.padding_side = "left"
    outs: list[str] = []
    for b in range(0, len(prompts), bs):
        enc = tok(prompts[b:b + bs], return_tensors="pt", padding=True, add_special_tokens=False)
        g = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, temperature=None,
                           top_p=None, top_k=None, repetition_penalty=1.0, pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return outs


def run_layers(model, h: torch.Tensor, start: int, end: int | None = None) -> torch.Tensor:
    """Run decoder layers [start, end) on hidden states `h` (B, T, d) with a pure causal mask.

    Sequences must be RIGHT-padded: with causal attention a real token never attends to a later pad,
    so no padding mask is needed and results for real tokens are exact.
    """
    c = core(model)
    T = h.shape[1]
    pos = torch.arange(T).unsqueeze(0)
    pe = c.rotary_emb(h, pos)
    for layer in c.layers[start:end]:
        h = layer(h, attention_mask=None, position_ids=pos, position_embeddings=pe)
    return h


def embed(model, ids: torch.Tensor) -> torch.Tensor:
    return core(model).embed_tokens(ids)


@contextmanager
def timer():
    t = {"s": time.perf_counter()}
    yield t
    t["dt"] = time.perf_counter() - t["s"]


def peak_rss_mb() -> float:
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
