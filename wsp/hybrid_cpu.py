"""Pure-PyTorch CPU port of the exp_train hybrid LM (Ba2han/step-17k and siblings).

The released `patch.py` needs flash-linear-attention's Triton kernels (GPU only), runs batch size 1, and computes the
MLP with fused weights, so module hooks never fire. This file re-implements its `_run` forward pass line for line with:

  * the gated delta rule in plain PyTorch (chunked WY form for prefill, checked against the naive token-by-token
    recurrence in `selftest_delta_rule`), with an optional carried state so the model can also decode incrementally;
  * right-padded batches (every operator is causal, so a real token never sees a later pad and its outputs are exact);
  * three taps per layer: `act_hook(layer, act)` on the SwiGLU activation that enters w_down (return a tensor to
    replace it), `down_hook(layer, act)` whose return value is added to the w_down output (for LoRA-style deltas), and
    `resid_hook(layer, x)` on the residual stream after the block;
  * float32 everywhere (the training run kept the residual stream, norms and gates in FP32 and matrices in BF16).

Architecture (from config.json): 56 blocks, every 4th a gated GQA softmax attention, the rest Gated DeltaNet; SwiGLU
MLPs (d=768, ffn=2304) with gated residuals (x += 2*sigmoid(a) * y); "Delta Block" depth sources added every 14 blocks
and mixed into the mixer / MLP inputs by a softmax over sources; a hashed bigram/trigram memory (1B BF16 entries)
injected before blocks 1 and 27.

A neuron here is one of the 2304 SwiGLU channels of a block's MLP. It writes g2 * W_down[:, j] into the residual
stream, where g2 = 2*sigmoid(res2.a) is the block's per-dimension output gate.
"""
from __future__ import annotations

import hashlib
import json
import math
import os

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file

_MULTS = [0x9E3779B97F4A7C15, 0xC2B2AE3D27D4EB4F, 0x165667B19E3779F9, 0xD6E8FEB86659FD93,
          0xFF51AFD7ED558CCD, 0xC4CEB9FE1A85EC53, 0x94D049BB133111EB, 0xBF58476D1CE4E5B9]
_SIGN_MULTS = [0xA0761D6478BD642F, 0xE7037ED1A0B428DB, 0x8EBC6AF09C88C6E3, 0x589965CC75374CC3,
               0x1D8E4E27C47D124F, 0x2D358DCCAA6C78A5, 0x9FB21C651E98DF25, 0xD3833E804F4C574B]


def _i64(x):
    return x - (1 << 64) if x >= (1 << 63) else x


def build_canon_map(tokenizer_file):
    """Same as patch.build_canon_map: token ids whose text normalizes to the same string share n-gram rows."""
    from tokenizers import Regex, Tokenizer, normalizers
    tokenizer = Tokenizer.from_file(str(tokenizer_file))
    norm = normalizers.Sequence([
        normalizers.NFKC(), normalizers.NFD(), normalizers.StripAccents(), normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "), normalizers.Replace(Regex(r"^ $"), "\ue000"),
        normalizers.Strip(), normalizers.Replace("\ue000", " ")])   # U+E000 placeholder keeps a lone space, as in patch.py
    vocab = tokenizer.get_vocab_size(with_added_tokens=True)
    special = {t["id"] for t in json.loads(tokenizer.to_str()).get("added_tokens", []) if t.get("special")}
    first, canon = {}, torch.arange(vocab, dtype=torch.int64)
    for t in range(vocab):
        if t in special:
            continue
        text = tokenizer.decode([t], skip_special_tokens=False)
        if "�" in text:
            continue
        key = norm.normalize_str(text)
        if key:
            canon[t] = first.setdefault(key, t)
    return canon


def _rms(x, w, eps):
    return F.rms_norm(x, (x.shape[-1],), w, eps)


def _select(sources, qw, eps):
    """Delta Block: sum_i softmax_i(q . RMSNorm_w(s_i)) s_i  (qw = query * weight)."""
    logits = torch.stack([(s * qw).sum(-1) * torch.rsqrt(s.pow(2).mean(-1) + eps) for s in sources])
    p = logits.softmax(0)
    out = p[0].unsqueeze(-1) * sources[0]
    for pi, s in zip(p[1:], sources[1:]):
        out = out + pi.unsqueeze(-1) * s
    return out


def _l2n(x, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True) + eps)


# ----------------------------------------------------------------------------- gated delta rule (pure PyTorch)
def delta_rule_recurrent(q, k, v, g, beta, scale, S=None):
    """Reference: FLA's naive recurrence. q,k [B,T,H,K] (already L2-normed), v [B,T,H,V], g (log decay), beta [B,T,H].
    S_t = exp(g_t) S_{t-1};  S_t += k_t (beta_t (v_t - S_t^T k_t))^T;  o_t = S_t^T (scale q_t)."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    S = q.new_zeros(B, H, K, V) if S is None else S.clone()
    o = q.new_zeros(B, T, H, V)
    for t in range(T):
        S = S * g[:, t].exp()[..., None, None]
        kt = k[:, t]
        vt = (v[:, t] - (S * kt[..., None]).sum(-2)) * beta[:, t][..., None]
        S = S + kt[..., None] * vt[..., None, :]
        o[:, t] = torch.einsum("bhk,bhkv->bhv", q[:, t] * scale, S)
    return o, S


def delta_rule_chunked(q, k, v, g, beta, scale, S=None, chunk=64):
    """Chunked (WY) form of the same recurrence; exact up to float rounding. Same shapes as delta_rule_recurrent."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    chunk = min(chunk, max(16, 1 << (T - 1).bit_length()))      # short inputs: don't pad 14 tokens to 64
    pad = (-T) % chunk
    # padded steps: k = 0, beta = 0, g = 0 -> they neither decay nor write the state
    tr = lambda x: F.pad(x.transpose(1, 2), (0, 0, 0, pad)) if x.dim() == 4 else F.pad(x.transpose(1, 2), (0, pad))
    q, k, v, g, beta = tr(q) * scale, tr(k), tr(v), tr(g), tr(beta)            # [B,H,Tp,*]
    N = (T + pad) // chunk
    q, k, v = (x.reshape(B, H, N, chunk, -1) for x in (q, k, v))
    g, beta = g.reshape(B, H, N, chunk).cumsum(-1), beta.reshape(B, H, N, chunk)
    idx = torch.arange(chunk)
    strict = idx[:, None] > idx[None, :]
    incl = idx[:, None] >= idx[None, :]
    dg = g[..., :, None] - g[..., None, :]
    decay_strict = dg.masked_fill(~strict, float("-inf")).exp()
    decay_incl = dg.masked_fill(~incl, float("-inf")).exp()
    kb = k * beta[..., None]
    A = (kb @ k.transpose(-1, -2)) * decay_strict                            # strictly lower
    eye = torch.eye(chunk, dtype=q.dtype)
    Tm = torch.linalg.solve_triangular(eye + A, eye.expand_as(A), upper=False, unitriangular=True)
    u = Tm @ (v * beta[..., None])
    w = Tm @ (kb * g.exp()[..., None])
    S = q.new_zeros(B, H, K, V) if S is None else S
    o = []
    for n in range(N):
        qn, kn, gn = q[:, :, n], k[:, :, n], g[:, :, n]
        vn = u[:, :, n] - w[:, :, n] @ S
        attn = (qn @ kn.transpose(-1, -2)) * decay_incl[:, :, n]
        o.append((qn * gn.exp()[..., None]) @ S + attn @ vn)
        glast = gn[..., -1:]
        S = S * glast.exp()[..., None] + (kn * (glast - gn).exp()[..., None]).transpose(-1, -2) @ vn
    o = torch.stack(o, 2).reshape(B, H, N * chunk, V)[:, :, :T].transpose(1, 2)
    return o, S


def selftest_delta_rule(seed=0):
    """Max abs difference between the chunked and the recurrent form on random inputs (float64), with and without an
    initial state and with a length that is not a multiple of the chunk size."""
    gen = torch.Generator().manual_seed(seed)
    B, T, H, K, V = 2, 150, 3, 16, 16
    r = lambda *s: torch.randn(*s, generator=gen, dtype=torch.float64)
    q, k, v = _l2n(r(B, T, H, K)), _l2n(r(B, T, H, K)), r(B, T, H, V)
    g = -F.softplus(r(B, T, H))
    beta = torch.sigmoid(r(B, T, H))
    S0 = r(B, H, K, V) * 0.1
    errs = []
    for init in (None, S0):
        o1, s1 = delta_rule_recurrent(q, k, v, g, beta, K ** -0.5, init)
        o2, s2 = delta_rule_chunked(q, k, v, g, beta, K ** -0.5, init, chunk=32)
        errs += [float((o1 - o2).abs().max()), float((s1 - s2).abs().max())]
    return max(errs)


# ----------------------------------------------------------------------------- model
class HybridCPU(torch.nn.Module):
    def __init__(self, path, dtype=torch.float32, load_ngram=True):
        super().__init__()
        self.path = path
        cfg = json.load(open(os.path.join(path, "config.json")))
        self.cfg = cfg
        c, mods = cfg["model_config"], cfg["modifications"]
        assert mods.get("delta_block") and mods.get("delta_variant", "repo") == "repo"
        assert mods.get("depth_attn", "off") == "off"
        self.c = c
        self.D, self.EPS = c["d_model"], c["norm_eps"]
        self.H, self.KV, self.HD = c["n_heads"], c["n_kv_heads"], c["head_dim"]
        self.GH, self.GK = c["gdn_heads"], c["gdn_head_dim"]
        self.GV, self.CONV = int(self.GK * c["gdn_expand_v"]), c["gdn_conv_size"]
        self.n_layers = n = c["n_layers"]
        self.ffn = c["ffn_dim"]
        self.per_block = math.ceil(n / mods.get("delta_blocks", 4))
        self.is_attn = [(i + 1) % c["full_attn_every"] == 0 for i in range(n)]
        ng = c["ngram"]
        self.ng = ng
        self.rows = max(1024, ng["total_params"] // (len(ng["orders"]) * ng["heads_per_order"] * ng["dim"]))
        self.inject = {l: k for k, l in enumerate(ng["inject_layers"])}

        sd = load_file(os.path.join(path, "model.safetensors"))
        self.sd_keys = sorted(sd)
        self.w = torch.nn.ParameterDict()
        for name, t in sd.items():
            self.w[name.replace(".", "__")] = torch.nn.Parameter(t.to(dtype), requires_grad=False)
        self.dtype = dtype
        if load_ngram:
            with safe_open(os.path.join(path, "ngram.safetensors"), "pt", device="cpu") as f:
                table = f.get_tensor("table")
            expected = (len(ng["orders"]) * ng["heads_per_order"] * self.rows, ng["dim"])
            assert tuple(table.shape) == expected, (table.shape, expected)
            self.register_buffer("table", table, persistent=False)
            canon = build_canon_map(os.path.join(path, "tokenizer.json"))
            sha = mods.get("ngram_canonical_sha256")
            if ng.get("canonical_tokens"):
                assert sha is None or hashlib.sha256(canon.numpy().tobytes()).hexdigest() == sha, \
                    "canonical-token map differs from the one used in training"
            else:
                canon = torch.arange(c["vocab_size"])
            self.register_buffer("canon", canon, persistent=False)
        self.act_hook = None
        self.resid_hook = None
        self.down_hook = None
        self._fuse()

    @torch.no_grad()
    def _fuse(self):
        """Concatenate each block's input projections (and SwiGLU gate/up) into one matrix, as patch.py does; the
        original parameters become views into it, so an edit to either is seen by both."""
        self._fused = []
        for i in range(self.n_layers):
            p = f"blocks.{i}."
            names = ([p + "mixer." + n for n in ("q_proj", "k_proj", "v_proj", "gate_proj")] if self.is_attn[i] else
                     [p + "mixer." + n for n in ("q_proj", "k_proj", "v_proj", "a_proj", "b_proj", "g_proj")])
            d = {}
            for key, group in (("in", names), ("mlp", [p + "mlp.w_gate", p + "mlp.w_up"])):
                params = [self.P(n + ".weight") for n in group]
                W = torch.cat([q.data for q in params])
                o = 0
                for q in params:
                    q.data = W[o:o + q.shape[0]]
                    o += q.shape[0]
                d[key] = W
            self._fused.append(d)

    # parameter access by checkpoint name
    def P(self, name):
        return self.w[name.replace(".", "__")]

    def w_down(self, layer):
        return self.P(f"blocks.{layer}.mlp.w_down.weight")

    def out_gate(self, layer):
        """g2 = 2*sigmoid(res2.a): per-dimension scale on everything the block's MLP writes."""
        return 2 * torch.sigmoid(self.P(f"blocks.{layer}.res2.a"))

    def state_dict_ckpt(self):
        """Weights under the checkpoint's own names (for writing an edited model.safetensors)."""
        return {k.replace("__", "."): v.detach() for k, v in self.w.items()}

    # ------------------------------------------------------------------ pieces
    def _ngram_memory(self, cids, pos):
        """cids [B, T + 2] (two context ids first), pos [T] absolute positions of the last T ids."""
        ng = self.ng
        rows, signs, valid = [], [], []
        T = pos.shape[0]
        sh = lambda j: cids[:, 2 - j: 2 - j + T]                  # id at t - j
        t = 0
        for order in ng["orders"]:
            for head in range(ng["heads_per_order"]):
                h = torch.zeros_like(sh(0))
                m = _i64(_MULTS[(head * 3 + order) % 8])
                for j in range(order):
                    h = h * m + (sh(j) + 1 + j * 7919)
                h = h ^ (h >> 29); h = h * _i64(0xBF58476D1CE4E5B9); h = h ^ (h >> 32)
                rows.append(torch.remainder(h, self.rows) + t * self.rows)
                if ng.get("sign_pool"):
                    s = torch.full_like(sh(0), _i64(0x632BE59BD9B4E019))
                    m = _i64(_SIGN_MULTS[(head * 3 + order) % 8])
                    for j in range(order):
                        s = s * m + (sh(j) + 1 + j * 104729)
                    s = s ^ (s >> 31); s = s * _i64(0x94D049BB133111EB); s = s ^ (s >> 29)
                    signs.append(s & (ng["sign_pool"] - 1))
                valid.append((pos >= order - 1)[None].expand_as(h))
                t += 1
        emb = self.table[torch.stack(rows, -1)].to(self.dtype)                      # [B, T, n_tables, dim]
        mult = torch.stack(valid, -1).unsqueeze(-1).to(self.dtype)
        if signs:
            mult = self.P("ngram.sign_pool")[torch.stack(signs, -1)] * mult
        return (emb * mult).flatten(2)

    def _rope(self, x, pos):
        d = x.shape[-1]
        inv = 1.0 / (self.c["rope_theta"] ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
        f = pos.float()[:, None] * inv[None]
        cos, sin = f.cos()[:, None].to(x.dtype), f.sin()[:, None].to(x.dtype)
        x1, x2 = x[..., : d // 2], x[..., d // 2:]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1)

    def _gdn(self, i, xn, st):
        p = f"blocks.{i}.mixer."
        B, T, _ = xn.shape
        H, dk, dv = self.GH, self.GK, self.GV
        qkv, a, b, gate = F.linear(xn, self._fused[i]["in"]).split((2 * H * dk + H * dv, H, H, H * dv), -1)
        w = torch.cat([self.P(p + "q_conv1d.weight")[:, 0], self.P(p + "k_conv1d.weight")[:, 0],
                       self.P(p + "v_conv1d.weight")[:, 0]])                       # [C, K]
        prev = st["conv"][i] if st is not None and st["conv"][i] is not None else qkv.new_zeros(B, self.CONV - 1, qkv.shape[-1])
        win = torch.cat([prev, qkv], 1).transpose(1, 2)                            # [B, C, K-1+T]
        y = F.conv1d(win, w[:, None], groups=w.shape[0]).transpose(1, 2)            # [B, T, C]
        q, k, v = F.silu(y).split((H * dk, H * dk, H * dv), -1)
        neg_A = -self.P(p + "A_log").exp()
        g = neg_A * F.softplus(a + self.P(p + "dt_bias"))                           # [B, T, H] log decay
        beta = b.sigmoid()
        q, k = _l2n(q.reshape(B, T, H, dk)), _l2n(k.reshape(B, T, H, dk))
        S0 = st["S"][i] if st is not None else None
        o, S = delta_rule_chunked(q, k, v.reshape(B, T, H, dv), g, beta, dk ** -0.5, S0)
        if st is not None:
            st["S"][i] = S
            st["conv"][i] = win.transpose(1, 2)[:, -(self.CONV - 1):]
        o = _rms(o, self.P(p + "o_norm.weight"), self.EPS) * F.silu(gate.view(B, T, H, dv))
        return o.reshape(B, T, -1) @ self.P(p + "o_proj.weight").T

    def _attention(self, i, xn, pos, st):
        p = f"blocks.{i}.mixer."
        B, T, _ = xn.shape
        H, KV, HD = self.H, self.KV, self.HD
        q, k, v, gate = F.linear(xn, self._fused[i]["in"]).split((H * HD, KV * HD, KV * HD, H * HD), -1)
        q, k, v = q.reshape(B, T, H, HD), k.reshape(B, T, KV, HD), v.reshape(B, T, KV, HD)
        q = self._rope(_rms(q, self.P(p + "q_norm.weight"), self.EPS), pos).transpose(1, 2)
        k = self._rope(_rms(k, self.P(p + "k_norm.weight"), self.EPS), pos).transpose(1, 2)
        v = v.transpose(1, 2)
        if st is not None and st["k"][i] is not None:
            k, v = torch.cat([st["k"][i], k], 2), torch.cat([st["v"][i], v], 2)
        if st is not None:
            st["k"][i], st["v"][i] = k, v
        P0 = k.shape[2] - T
        if P0 == 0:
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        else:
            mask = torch.arange(P0 + T)[None, :] <= (torch.arange(T)[:, None] + P0)
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
        o = o.transpose(1, 2) * torch.sigmoid(gate.view(B, T, H, HD))
        return o.reshape(B, T, -1) @ self.P(p + "o_proj.weight").T

    def new_state(self):
        n = self.n_layers
        return {"conv": [None] * n, "S": [None] * n, "k": [None] * n, "v": [None] * n, "ctx": None, "pos": 0}

    # ------------------------------------------------------------------ forward
    def forward(self, ids, state=None, last_only=False, return_hidden=False):
        """ids [B, T] (LongTensor; right-pad with any id, outputs at real positions are exact). With `state` (from
        new_state()) the call continues that sequence and updates it in place. Returns logits [B, T, V] (float)."""
        if ids.dim() == 1:
            ids = ids[None]
        B, T = ids.shape
        eps = self.EPS
        P0 = state["pos"] if state is not None else 0
        pos = torch.arange(P0, P0 + T)
        cids = self.canon[ids]
        ctx = state["ctx"] if state is not None and state["ctx"] is not None else cids.new_zeros(B, 2)
        mem = self._ngram_memory(torch.cat([ctx, cids], 1), pos)
        if state is not None:
            state["ctx"] = torch.cat([ctx, cids], 1)[:, -2:]
            state["pos"] = P0 + T
        x = self.P("embed.weight")[ids].to(self.dtype)
        sources = []
        hidden = []
        for i in range(self.n_layers):
            p = f"blocks.{i}."
            if i in self.inject:
                kk = self.inject[i]
                e = mem @ self.P(f"ngram.proj.{kk}.weight").T
                gate = torch.sigmoid((_rms(x, self.P(f"ngram.key_norm.{kk}.weight"), eps) *
                                      _rms(e, self.P(f"ngram.val_norm.{kk}.weight"), eps)).sum(-1, keepdim=True)
                                     / math.sqrt(self.D))
                x = x + gate * e
            attn_sources = list(sources)
            if i % self.per_block == 0:
                sources.append(x if not sources else x - sources[-1])
            if attn_sources:
                h = x + _select(attn_sources, self.P(p + "delta_attn_query") * self.P(p + "delta_attn_weight"), eps)
            else:
                h = x
            xn = _rms(h, self.P(p + "norm1.weight"), eps)
            y = self._attention(i, xn, pos, state) if self.is_attn[i] else self._gdn(i, xn, state)
            x = x + 2 * torch.sigmoid(self.P(p + "res1.a")) * y
            h = x + _select(sources, self.P(p + "delta_mlp_query") * self.P(p + "delta_mlp_weight"), eps)
            hn = _rms(h, self.P(p + "norm2.weight"), eps)
            gt, up = F.linear(hn, self._fused[i]["mlp"]).chunk(2, -1)
            act = F.silu(gt) * up
            if self.act_hook is not None:
                r = self.act_hook(i, act)
                if r is not None:
                    act = r
            y = F.linear(act, self.P(p + "mlp.w_down.weight"))
            if self.down_hook is not None:                 # e.g. a LoRA term on selected w_down columns
                r = self.down_hook(i, act)
                if r is not None:
                    y = y + r
            x = x + 2 * torch.sigmoid(self.P(p + "res2.a")) * y
            if self.resid_hook is not None:
                self.resid_hook(i, x)
            if return_hidden:
                hidden.append(x)
        if last_only:
            x = x[:, -1:]
        logits = _rms(x, self.P("norm.weight"), eps) @ self.P("lm_head.weight").T
        return (logits, hidden) if return_hidden else logits

    @torch.no_grad()
    def generate(self, ids, max_new_tokens=32, stop_ids=()):
        """Greedy decoding with the carried state (batch 1)."""
        ids = torch.as_tensor(ids, dtype=torch.long).view(1, -1)
        st = self.new_state()
        logits = self.forward(ids, state=st, last_only=True)
        out = []
        for _ in range(max_new_tokens):
            nxt = int(logits[0, -1].argmax())
            out.append(nxt)
            if nxt in stop_ids:
                break
            logits = self.forward(torch.tensor([[nxt]]), state=st)
        return out


def load(path="/home/user/models/step-17k", threads=None):
    if threads:
        torch.set_num_threads(threads)
    from transformers import PreTrainedTokenizerFast
    tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(path, "tokenizer.json"), bos_token="<bos>",
                                  eos_token="<eos>", unk_token="<unk>", pad_token="<pad>")
    model = HybridCPU(path).eval()
    return model, tok
