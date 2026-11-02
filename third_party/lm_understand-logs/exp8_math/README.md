# Experiment 8: Astra edits Qwen3.5-0.8B's weights to make it better at math

This folder holds the complete logs behind "EXPERIMENT 8 - Make a language model better at math" in the blog post,
and (in `gemma_followup/`) the logs behind its follow-up, "Additional Experiment - Raising math capability of a model
that does not loop" (Gemma 3 270M; see `gemma_followup/README.md`).

## The question

Can a frontier LLM, used as it is, make a 0.8-billion-weight language model better at math by reading its internals and
editing its weights directly, with no gradient descent and no training?

## Setup

- **Editor:** Astra (`gpt-6-astra`) through the Codex CLI, reasoning effort `max`, one conversation per run (with a
  fresh conversation started by the harness every 40 turns or after a harness restart, the "handoff": a new message
  with the task, Astra's notebook and its current edit). Codex's own tools were switched off; every action went through the harness below.
- **Model edited:** Qwen3.5-0.8B (the text part of the public release `Qwen/Qwen3.5-0.8B`, 24 layers), thinking on.
- **What Astra had** (all of it is described in the first message, `turns/t001/prompt.txt`):
  1. a persistent Python kernel over the weights (CPU only, no GPU, no internet): load a tensor, compute, and
     `save_edit(name, tensor)` to change it; at most 300 code runs;
  2. practice problems with answers, none from the test sets: 7,469 from GSM8K train, 7,275 from MATH train (and 714
     AIME problems, answers only) - the files are in `practice/`;
  3. a forward-pass tool with a budget of **10, 25 or 50 passes** (the three runs). One pass is either a generation
     (the model, original and/or edited, answers up to 512 practice problems with the evaluation's sampling; "both"
     gives a paired comparison of original and edited weights on the same problems, 256 each) or a teacher-forced read
     of up to 1,024 texts (log-probs, hidden states, MLP activations, attention). A failed request did not count.
- **Forbidden:** gradient descent or any training (no backward passes, autograd, optimizers or iterative fitting of
  weights to a loss); running the model in the kernel (only the forward-pass tool runs it); internet. Edits to the final
  norm, the embedding / output layer, and edits that only multiply a whole tensor by a constant were refused by
  `save_edit`.
- **How a run ends:** it continues until all passes are used; then Astra makes its final choice (the tensors in its
  edit folder are submitted) and replies without code. It never saw the benchmarks.
- **Explanations:** from a turn early in each run (turn 52 in p10c, 16 in p25c, 29 in p50c) the harness asked Astra to
  explain every step in plain English before its code ("EXPLAIN EVERY STEP"); a block that came without an explanation
  was sent back unrun (at most twice in a row).
- **Notes from the experimenters during the runs** (`notices/`): `001_pass_size.txt` (a pass costs one pass at any
  size) and `002_tool_facts.txt` (tool corrections) were delivered to all three runs as part of a turn's message.
- **Evaluation** (after the run, never shown to Astra): the submitted tensors are cast to bfloat16 and replace the
  originals (`code/build_ckpt.py`); vLLM, the chat template with thinking enabled, the user message = problem +
  "Please reason step by step, and put your final answer within \boxed{}.", temperature 0.6, top_p 0.95, top_k 20,
  presence penalty 1.5, at most 32,768 new tokens; the last `\boxed{}` answer is graded with math-verify
  (`code/mathqa.py`). Samples per problem: MATH-500 4, GSM8K 2, AMC23 16, Minerva Math 4, OlympiadBench 4; beyond
  math: MMLU-Pro (a fixed random 1,000 questions, official prompt and answer extraction, 1 sample), GPQA Diamond (4
  samples), IFEval (strict prompt-level accuracy with the official checkers on the answer after `</think>`, 2 samples).
  Scores are the mean accuracy over all samples. Code: `code/evaluate.py`, `code/merge.py`, protocol `think35` in
  `code/profiles.py`.

## Which run is which

| In the post | Run folder | Passes used | Turns | Python runs | Conversations | Hours | Final edit (from `final_answer.md`) |
|---|---|---|---|---|---|---|---|
| Unedited | (no run) | - | - | - | - | - | the base model; its evaluation is `eval/unedited/` |
| Edited, 10 passes | `runs/p10c` | 10 | 74 | 59 | 2 | 11.6 | removes a loop ("repetition") direction from 30 output matrices (attention and MLP) in layers 8-22, W' = W - 1.5 u(u^T W) |
| Edited, 25 passes | `runs/p25c` | 25 | 86 | 45 | 3 | 8.5 | switches off attention head 7 of layer 15 (zeroes its 256 output columns) |
| Edited, 50 passes | `runs/p50c` | 50 | 158 | 80 | 4 | 20.3 | subtracts the loop direction from the 64 output columns per layer of the MLPs of layers 15 and 19 that write it most |

The run names are the original IDs (p = Qwen3.5-0.8B thinking on, c = prompt version 3). Hours are wall-clock time
from the first to the last turn, including the forward passes (a full generation pass took up to about half an hour)
and, in p50c, 4.7 hours at turn 154 when the model could not be reached and the harness waited to retry. The trajectory figure plays these
three runs turn by turn.

## The post's numbers and where each comes from

| Post | Number | File (field) |
|---|---|---|
| Table, unedited: MATH-500 / GSM8K / IFEval / MATH-500 answers cut at the token limit | 7.9 / 13.8 / 21.0 / 93.0 | `eval/unedited/merged.json` (`math500.acc`, `gsm8k.acc`, `ifeval.acc`, `math500.cut_share`) |
| Table, 10 passes | 12.3 / 25.7 / 49.0 / 87.2 | `eval/p10c/merged.json` (same fields) |
| Table, 25 passes | 14.1 / 25.1 / 39.9 / 86.4 | `eval/p25c/merged.json` |
| Table, 50 passes | 48.1 / 67.1 / 62.3 / 33.1 | `eval/p50c/merged.json` |
| Benchmark figure: AMC23, Minerva Math, OlympiadBench, MMLU-Pro, GPQA Diamond (0 / 10 / 25 / 50 passes) | AMC23 0.5 / 1.1 / 1.1 / 23.4; Minerva 1.5 / 3.9 / 3.5 / 25.4; OlympiadBench 1.3 / 2.4 / 3.9 / 21.8; MMLU-Pro 40.3 / 40.0 / 44.7 / 45.5; GPQA Diamond 22.7 (see note) / 29.4 / 30.1 / 36.6 | the same `merged.json` files (`amc23`, `minerva`, `olympiad`, `mmlupro1k`, `gpqa`); also `eval/RESULTS_excerpt.md` |
| "85 of its first 96 practice answers ran to the 32,768-token limit" | 85 / 96 | `runs/p50c/requests/fwd_1.done.json` (`generations`, `failures.cut_at_max_tokens`) |
| "subtracted that direction from 64 output columns in each (131,072 weights)" of layers 15 and 19 | 2 x 64 x 1,024 | `runs/p50c/final_answer.md`, `runs/p50c/work_notes/frozen_weights_manifest.json`, `runs/p50c/edit/manifest.json` (`changed_columns` 64 per tensor) |
| Practice problems: 7,469 from GSM8K, 7,275 from MATH | | `practice/stats.json` |
| Gemma: "only 8.5% of its answers reach its token limit, against 93% for Qwen" | 8.5 / 93.0 | `gemma_followup/eval/unedited/merged.json` (`math500.cut_share`), `eval/unedited/merged.json` |

Every `merged.json` is recomputed exactly from its `parts/*/scores.json` by `code/merge.py` (checked for this release).
The post's figure data (`assets/mathedit/benchmarks.json`, `trajectories.json`) were generated from these runs'
`state.json`, `final_answer.md`, `requests/` and `turns/*/{answer.md,prompt.txt}`, and from the results table
(`eval/RESULTS_excerpt.md`).

**Note on GPQA Diamond, unedited model:** the post's benchmark figure shows 24.5, an interim value when 149 of the 198
questions had been scored (the figure marks it "149 of 198 questions scored"). The finished evaluation, in
`eval/unedited/merged.json`, is **22.7**. Nothing else in the post changed between that interim table and the final one.

## Files

```
runs/<run>/
  turns/tNNN/prompt.txt    the message Astra received in turn NNN (turn 1: the task; later: the output of its previous
                           code or forward block, notes from the experimenters, and at each new conversation the handoff)
  turns/tNNN/answer.md     Astra's reply: its explanation and one ```python or ```forward block, or its final answer
  turns/tNNN/events.jsonl  the raw Codex event stream of the turn (agent messages; no reasoning summaries were emitted
                           in these runs; the "error" item in each turn is Codex's notice that its code mode is off)
  turns/tNNN/run.json      the turn's record: model, reasoning effort, the Codex command, token usage as reported by
                           Codex (it accumulates over a conversation, so do not sum it over turns), wall time
  turns/tNNN/stderr.log    Codex's stderr
  message.txt              the task message in its final form (with the "EXPLAIN EVERY STEP" paragraph added during the
                           run); what Astra actually got first is turns/t001/prompt.txt
  instructions.md          the Codex instructions file ("Answer the user's request.")
  tool_calls.jsonl         every Python run (its code and printed output, time) and every forward-pass request, in order
  requests/                each forward pass: fwd_<k>.json.claimed (the request as Astra wrote it) and fwd_<k>.done.json
                           (the summary returned to Astra: accuracy, paired difference, share cut at max tokens, failures)
  explanations.md          Astra's per-step explanations, collected by the harness
  final_answer.md          Astra's final answer
  state.json               the run record: budget, passes used, turns, Python runs, conversation restarts ("handoffs"),
                           notes delivered
  driver.log, loop.log     the harness log of the run's course (the two files are identical)
  refused/                 (p50c) turn 6's first attempt, refused by the provider's content filter (the prompt and the
                           error); turns/t006 instead told Astra that the output could not be delivered and why
  turns_discarded/         (p25c) turn 29 as first answered before the harness was restarted, and a restart attempt
                           that opened a new conversation without the task text; turn 29 was then redone in a new
                           conversation with the full handoff message (turns/t029)
  work_notes/              Astra's own small files from its work folder: notebook (notes.md), final report, manifests,
                           plans, problem splits (the large generated texts and activations it worked on are not kept)
  readable/                (p50c) a readable export of the run: prompt, every turn, every forward pass, explanations
  edit/                    the submitted weight edit, as an exact diff from the base model (below)
eval/<unedited|p10c|p25c|p50c>/
  merged.json                          the scores per benchmark: acc, cut_share (share of answers cut at 32,768 tokens),
                                       mean_tokens, problems
  parts/<set>_s<i>of<n>/scores.json    one evaluation job (a shard of one benchmark): counts and the sampling protocol
  parts/.../<set>_per_problem.json     per problem: id, correct (one 0/1 per sample), first extracted answer
  parts/.../<set>_first_samples.jsonl.gz  the full generated text of the first sample of every problem (gzip)
eval/RESULTS_excerpt.md   the experiment's results table (Experiments 6 and 8 of the private results file), verbatim
practice/                 the practice problems Astra could use (GSM8K train, MATH train, AIME), as it saw them
notices/                  the notes sent to running sessions (001 and 002 reached these runs)
base_model/Qwen3.5-0.8B-text/  config, generation config, tokenizer config and chat template of the text-only model
code/                     evaluation, grading, checkpoint building, practice-set building; the edit diff tools
```

**The weight edits.** Each run's `edit/` holds `manifest.json` (the edited tensors, their shapes, how many entries
changed, sha256 of the base and of the edited tensor) and `edit.npz`. Small edits are stored as the changed entries
(indices and new bfloat16 values); p10c's rank-one edits, which touch almost every entry of 30 matrices, are stored as
their rank-one factors plus the handful of entries where rounding differs. `code/apply_edit.py` rebuilds the evaluated
tensors bit for bit from the public `Qwen/Qwen3.5-0.8B` weights (revision `2fc06364715b967f1860aea9cf38778875588b17`;
tensor names unchanged, `model.language_model.*`) and checks every hash; it can write a full `.safetensors` file or the
edited tensors as `.npy` files like the ones Astra submitted. Astra's submissions were float32 copies of whole tensors
(29 MB for p50c, 331 MB for p10c); the evaluation cast them to bfloat16, so the diff reproduces the evaluated model
exactly. In p50c, 130,188 of the 131,072 touched weights differ from the original after rounding to bfloat16.

**Not included:** the runs' work folders (0.5-2.5 GB each: activations and generated practice answers); the full text
of GPQA Diamond generations (they restate the questions, which GPQA asks not to publish; its scores and per-problem
results are kept); the per-sample length of every answer (the share cut at the token limit is recorded per evaluation
job in `scores.json`, `cut`); the agent harness code (the prompts, sandbox and forward-pass server; the rules it
enforced are stated in the prompts); the IFEval and multiple-choice graders `code/evaluate.py` imports (IFEval's
official checkers). Other runs of the experiment that appear as extra rows in `eval/RESULTS_excerpt.md` (other prompt
versions, 0 and 5 passes) are not part of the post and not included.
