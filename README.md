# Weight-space programming, checked on a CPU

This repo looks at the blog post
[Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/)
(Takehi, Teoh, Zhang, Khattab, MIT CSAIL, Nov 2 2026). In the post, "GPT-6 Astra" reads and edits neural-network
weights directly. The repo asks three things:

1. **Do the post's claims hold up** when we re-run them with the authors' own released networks, edits and logs?
2. **H2.** Can ordinary tools (linear algebra, gradients, activation statistics, with no frontier LLM) read weight
   space the way Astra did?
3. **H1.** Can hand-picking *where* LoRA goes (layers or neurons, chosen by such tools for one purpose) cut the cost
   of fine-tuning without hurting, or while improving, the result?

Everything ran on a 4-core x86 CPU with AMX-bf16 and no GPU (PyTorch 2.15, transformers 5.22, PEFT 0.21).

<!-- RESULTS -->

## 1. Do the blog's claims hold up?

Each claim was re-checked with the authors' own released artifacts (scripts in `verify/`, outputs in
`results/verify/`). Astra itself cannot be re-run, so the checks target what *can* be re-run: the networks it wrote,
the edits it made, the scoring, and the evidence it cited.

| Exp | Claim in the post | What we did | Result |
|---|---|---|---|
| 5 | MinAtar policies written from zero weights in one reply beat PPO-100M in all 3 games (517.4 / 53.1 / 53.2) | Rebuilt each network by parsing the `edits` block of Astra's reply; re-ran 1,000 fresh episodes per game in MinAtar 1.0.15 (sampled actions, sticky 0.1, 10k-step cap) | **Reproduced.** The rebuilt networks equal the released `hand.npz` bit for bit. Scores 502.9 ± 7.7 / 51.9 ± 1.4 / 54.7 ± 1.1 (all within 1.4 SE). PPO-100M best: 15.8 / 5.6 / 25.8. Note: 517 of 1,000 Breakout episodes hit the 10k-step cap, so that mean is capped. |
| 6 | 1.48M-weight Go net written from scratch: 0 passes 42.6 / 26.3 / 36.3%, 50 passes 74.2 / 70.0 / 78.5% vs same-size PPO | Replayed 2,000 games per pairing with pgx `go_9x9` and the released networks | **Reproduced.** 40.6 / 26.2 / 36.7% and 74.2 / 69.2 / 80.6%, every pairing within 1.7 SE. |
| 8 | Qwen3.5-0.8B fix = 131,072 weights (64 down_proj columns in layers 15 and 19) | Rebuilt all three runs' edits from the public weights with the release's `apply_edit.py` | **Reproduced.** Every base and edited SHA-256 matches. Each change is rank 1 (σ₁/σ₂ ≈ 1,500-1,800), 99.999% along one direction. One nuance: the direction's component in the 64 columns is multiplied by **−5** (W − 6·u·uᵀW), i.e. flipped and amplified, not only "subtracted". 131,072 is 0.017% of the 752M text-model weights. |
| 8 | The base model loops (93% of MATH-500 answers hit 32,768 tokens); after the edit most finish | Measured loops in the release's own first-sample generations | **Consistent.** 183 / 200 base answers contain a loop (≥ 90% of 300 consecutive 8-word n-grams seen before; median onset 2,725 words) vs 72 / 200 after the 50-pass edit (median onset 7,856 words); 13 vs 152 close `</think>`. The 10- and 25-pass edits barely help (166 and 169 loop), as the post says. |
| 8 | The edit removes the loop direction selectively, leaving productive reasoning alone (Astra's probe: −0.103 vs −0.009 nats/token) | Teacher-forced log-probs on 24 of the base model's own looping MATH-500 answers, which Astra never saw: 768 tokens before the loop vs 768 tokens of it | **Partly.** Loop text −0.089 ± 0.008 nats/token, the reasoning just before it −0.042 ± 0.003; loop hit harder in 22 / 24. It is selective, but productive text moves 4-5× more than Astra's probe said. |
| 8 | MATH-500 7.9% → 48.1% at 32,768 thinking tokens | Not re-run: ~60M generated tokens at ~28 tok/s on this CPU. See the loop-entry test below. | Not checkable here |
| 2 | Names the shape each DDPM draws (3 / 3); the models draw filled shapes while Astra drew outlines | Re-sampled 1,000 images per model with the release's schedule; parsed Astra's drawings | **Reproduced.** Samples are filled (interior share 0.32-0.50; an outline is ~0); Astra's drawings have interior 0.0. Our crude classifier sorts samples into the right class 94 / 92 / 74% of the time (its accuracy on genuine shapes: 84 / 100 / 69%). |
| 3 | From the middle 4 blocks only, names the language 6 / 6 and finds every copying head | Checked Astra's answers against the release's measured `heads.json`; tested Astra's stated evidence | **Consistent.** Answers match ground truth. Python has 3 copying heads vs 1 for the others, and romaji's last block does write in fewer directions (effective rank of W_o.3: 151 vs 184 / 157), as Astra argued. |
| 1 | Names the game of 4 / 4 PPO policies from 17,253 raw weights | Checked answers against the key; examined the evidence Astra cited | **Consistent.** The agent-channel weights really do sit on Breakout's bottom row, in Pong's edge column and on Pac-Man's maze (see H2). |
| 4 | One-shot edit of 3.5% of a Kinetix car policy: 0% → 100% success | Applied Astra's 51 edit statements to the network as shown (rounded) | **Partly.** The rebuilt network equals the released one bit for bit; 6,073 / 171,664 weights (3.54%) change. Success rates need the Kinetix JAX simulator and were not re-run. |
| 7 | Weights alone: cannot pick the sycophantic Gemma fine-tune; with attention patterns: 3 / 3 | Re-scored the released predictions against the answer key (fine-tuned models are not released, 17 GB) | **Consistent.** Attention tool 3 / 3. Each weights-only run answered "A" for every pair at p = 0.5, i.e. chance (1 / 3). |

<!-- ENTRY -->

## 2. H2: can ordinary tools read weight space instead of a frontier LLM?

For each reading Astra made, we wrote the most standard tool we could think of, with no LLM in the loop, and scored
it against the same ground truth.

| Exp | Astra (frontier LLM reading raw weights) | Ordinary tool, no LLM | Tool result |
|---|---|---|---|
| 2 | Shape of 3 / 3 diffusion models, no forward pass | Edge-orientation profile (Sobel, magnitude-weighted) of the 64 output-layer "strokes" in W3, matched to templates from synthetic shapes (`verify/exp2_diffusion.py`) | **3 / 3.** Correlation with the right template: circle 0.18, square 0.57, triangle 0.46; every wrong template ≤ 0.14. |
| 3 | Copying heads from the middle blocks only, 3 / 3 networks exact | Elhage et al. (2021) composition on the blocks: LayerNorm folded into Q/K/V/O (gain, and mean-removal as a projection), then the signed K-composition tr(QK·OV) with every earlier head (`verify/exp3_heads.py`) | **2 / 3 exact.** Python {3.5, 3.1, 2.1} and English {3.1} are found exactly; for romaji the true head 3.3 is ranked 2nd (0.063 vs 0.067). The textbook unsigned Frobenius score without the LayerNorm fold failed completely (true copying heads ranked last). We chose the signed variant after seeing that failure, and Astra used the same idea. |
| 1 | Game of 4 / 4 PPO policies, open answer | Where the agent channel's first-layer weights are large, plus which actions the output layer separates, through a fixed 3-rule decision (`verify/exp1_readout.py`) | **3 / 4.** Breakout (bottom row, left/right), Pong (edge column, up/down) and Pac-Man (maze-shaped map, all moves) are right. Freeway is missed: this variant lets the agent move sideways too. The tool also knew the four genres, which Astra was not told. |
| 8 | Found the loop direction and 64 writer neurons in 2 layers (158 turns, 20 h, 50 forward passes) | Mean-difference direction between loop and pre-loop activations, then writer-neuron ranking (`experiments/h2_rediscover_loop_edit.py`) | see below |

**What the frontier-LLM route cost.** Token counts are summed from the release's `run.json` files
(`results/verify/astra_usage.json`). Exps 3 and 8 count only the largest cumulative value per run folder, so they are
lower bounds.

| Exp | Astra: input / output tokens, turns | Our tool: CPU time |
|---|---|---|
| 1 | 295k / 7k, 4 turns | < 1 s |
| 2 | 299k / 7.5k, 3 turns | ~1 s for the reading; the 3,000 DDPM samples used only to check it take ~2 min |
| 3 | ≥ 1.6M / 116k, 54 turns | ~20 s |
| 8 | ≥ 11.4M / 335k, 562 turns (with the Gemma follow-up), plus 85 forward passes | see below |

<!-- H2_REDISCOVERY -->

## 3. H1: does hand-picking where LoRA goes cut cost and keep (or improve) quality?

**Setup** (`experiments/h1_*.py`, `wsp/train.py`). Model: Qwen2.5-0.5B-Instruct, frozen in bfloat16, adapters in
float32. Every arm uses the same data, steps, batch and learning rate (LoRA 1e-3, neuron deltas 3e-4). Purposes:

* **add**: 5-digit + 5-digit addition with the answer only (base model: 28%). 3,000 training sums, 150 steps of
  16. Scored on 300 held-out sums, plus 200 6-digit sums, a length never trained on.
* **french**: answer English questions in French by default. The training data is context distillation from the model
  itself: the teacher sees "...\nAnswer in French only." and the student does not. Scored as the share of 96 held-out
  answers in French, and on 48 capital-city questions answered correctly.

Collateral damage: KL(base ‖ tuned) on raw Wikipedia text and on generic chat replies, and 0-shot MMLU on a fixed 200
questions.

**Where the tools put the adapters** (`experiments/h1_locate.py`). Before any training, one backward pass of the task
loss on 64 examples gives the gradient energy of every linear module. The cost is about one minute of CPU, with no
LLM.

| Arm | Trainable weights |
|---|---|
| `full` | LoRA r=8 on q, k, v, o, gate, up and down of all 24 layers (the usual default) |
| `full_r1` | LoRA r=1 everywhere: about the parameter count of a 4-layer r=8 LoRA, spread thin |
| `pick4` | LoRA r=8 on the 4 layers with the most task-gradient energy |
| `pick4u` | the 4 upper-half layers with the most energy: a cost-aware pick, since backprop stops and the frozen-layer cache starts at the lowest trainable layer |
| `spectrum4` | the 4 layers with the highest weight-only SNR (Spectrum-style: singular values above the Marchenko-Pastur edge), so not task-specific |
| `rand4a`, `rand4b`, `anti4` | controls: two random draws, and the 4 layers with the *least* energy |
| `neur` | only the down_proj columns of the 64 neurons with the most gradient energy in each of the 2 MLPs with the most energy: 114,688 weights, the footprint of Astra's Exp 8 edit |
| `neur_rand` | 64 random neurons in the same 2 layers |

Two cost savers come for free once the trainable weights sit at layer L and above. Autograd never backpropagates below
L. And the hidden states entering L never change, so they are computed once per example (frozen-prefix cache)
instead of once per step.

<!-- H1_RESULTS -->

## Repository layout

| Path | What it is |
|---|---|
| `third_party/lm_understand-logs/` | The blog's released logs ([RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs) @ `4d09ba6`, MIT / CC BY 4.0): every script, README, evaluation JSON and small network (7 MB). `third_party/fetch_lm_understand_logs.sh` fetches the full 370 MB (Exp 3's blocks, Exp 6's networks, Exp 8's generations) into `third_party/lm_understand-logs-full/`. |
| `verify/` | One script per blog experiment that re-checks its claims. |
| `experiments/h2_rediscover_loop_edit.py` | H2: a scripted pipeline that tries to rediscover Astra's Qwen3.5-0.8B loop edit. |
| `experiments/h1_*.py`, `experiments/h1_sweep.sh` | H1: tool-based placement and LoRA arms on Qwen2.5-0.5B-Instruct. |
| `wsp/` | Shared code: model helpers, tasks and retention probes (`tasks.py`), placement tools (`locate.py`), LoRA and neuron-sparse training with cost accounting (`train.py`). |
| `results/` | Every number in this README, as JSON (`verify/`, `h1/`, `h2/`). |

## Reproduce

```bash
uv venv venv --python 3.13 && source venv/bin/activate
uv pip install -r requirements.txt minatar==1.0.15 "jax[cpu]" pgx
third_party/fetch_lm_understand_logs.sh          # large files for Exp 3, 6, 8
```

```bash
python verify/exp5_minatar.py            # ~3 min
python verify/exp2_diffusion.py          # ~3 min
python verify/exp3_heads.py              # ~1 min
python verify/exp1_readout.py; python verify/exp4_footprint.py
python verify/exp6_go_h2h.py             # ~5 min
python third_party/lm_understand-logs/exp8_math/code/apply_edit.py <Qwen3.5-0.8B model.safetensors> third_party/lm_understand-logs-full/exp8_math/runs/p50c/edit /home/user/edits/p50c
python verify/exp8_qwen35_edit.py        # ~60 min on CPU
python experiments/h2_rediscover_loop_edit.py
python experiments/h1_locate.py --purpose add && python experiments/h1_locate.py --purpose french
experiments/h1_sweep.sh                  # ~1.5 h on CPU
```
