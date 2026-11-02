# Experiment 8, follow-up: Astra edits Gemma 3 270M's weights for math

The logs behind "Additional Experiment - Raising math capability of a model that does not loop" in the blog post.
Qwen3.5-0.8B's gains came from removing its loops. Gemma 3 270M IT rarely loops (8.5% of its MATH-500 answers reach
its token limit, against 93% for Qwen3.5-0.8B with thinking on), so this asks whether weight editing can raise math
ability that is not hidden behind a failure like looping. Astra could not improve it: no edit moved MATH-500 by more
than about a point.

## Setup

The same harness and rules as the main experiment (see `../README.md`), with these differences, all stated in the
first message (`runs/<run>/turns/t001/prompt.txt`, identical to `message.txt` here):

- **Model edited:** Gemma 3 270M IT, the public release `google/gemma-3-270m-it` (revision
  `ac82b4e820549b854eebf28ce6dedaf9fdfa17b3`), 18 layers of width 640.
- **Editor:** Astra (`gpt-6-astra`) through the Codex CLI, reasoning effort `max`, tool-free except the harness's
  Python kernel and forward-pass tool; budgets of **5, 10, 25 and 50 forward passes**. A generation pass answers up to
  2,048 practice problems (1,024 per model in a paired "both" pass); a teacher-forced pass reads up to 1,024 texts.
  Practice problems: GSM8K train and MATH train (`../practice/`).
- **Method in the prompt (version 5):** fix a DEV set (for choosing edits) and a HOLDOUT set (only for confirming
  them) before any edit; screen edits with full-size passes; freeze the best candidate before HOLDOUT; at the end
  submit the best edit found even if its gain is not significant, and report it honestly; explain every step.
- **Evaluation** (never shown to Astra): MATH-500 only, 4 samples per problem, the model card's sampling (temperature
  1.0, top_k 64, top_p 0.95), at most 8,192 new tokens, no thinking mode, the user message = problem + "Please reason
  step by step, and put your final answer within \boxed{}.". **Format-agnostic grading** (the post's numbers, also
  stated to Astra in its prompt, and used to grade its practice answers): the last `\boxed{}` answer if there is one,
  otherwise the answer stated after 'Final Answer' or 'the answer is', otherwise the last number of the response
  (`lenient_pred` / `lenient_correct` in `../code/evaluate.py`). Gemma often answers without `\boxed{}`; the strict
  `\boxed{}`-only score is kept too (`acc`).
- **Note from the experimenters during the runs:** `../notices/004_failure_counts.txt` (a correction to how the
  generation summaries counted answers without a final answer), delivered in a turn's message (g05b turn 19, g10b turn
  28, g25b turn 30, g50b turn 25). Notices 001-003 were already part of this prompt version.

## Which run is which, and the post's numbers

| In the post | Run folder | Passes used | Turns | Python runs | Conversations | Hours | MATH-500 (%) | Final edit (from `final_answer.md`) |
|---|---|---|---|---|---|---|---|---|
| Unedited | (no run) | - | - | - | - | - | **6.8** | - |
| Edited, 5 passes | `runs/g05b` | 5 | 20 | 13 | 1 | 0.9 | **7.0** | halves 64 selected MLP output columns across layers 6-17 |
| Edited, 10 passes | `runs/g10b` | 10 | 33 | 20 | 1 | 1.2 | **7.6** | multiplies 4 MLP output columns (layers 8, 14, 17) by 1.5 |
| Edited, 25 passes | `runs/g25b` | 25 | 73 | 44 | 2 | 2.8 | **7.7** | doubles neuron 920 of layer 17's MLP (a numerical feature): the post's "doubling one neuron that responds to numbers" |
| Edited, 50 passes | `runs/g50b` | 50 | 117 | 63 | 3 | 3.9 | **7.0** | multiplies head 2 of layer 11's attention output by 1.1 and halves 8 MLP columns of layer 11 |

The run names are the original IDs (g = Gemma 3 270M, b = the runs whose prompt stated the format-agnostic grading).

| Post | File (field) |
|---|---|
| MATH-500, unedited 6.8 | `eval/unedited/merged.json` (`math500.acc_lenient`) |
| MATH-500, 5 / 10 / 25 / 50 passes: 7.0 / 7.6 / 7.7 / 7.0 | `eval/<run>/merged.json` (`math500.acc_lenient`) |
| "only 8.5% of its answers reach its token limit" | `eval/unedited/merged.json` (`math500.cut_share` 0.085) |
| "a gain Astra itself reported as not established" | `runs/g05b`, `g10b`, `g50b`: `final_answer.md` (HOLDOUT gain with its standard error and 95% interval) |
| "wrong arithmetic and invalid algebraic steps", "no single component behind them" | the runs' `explanations.md`, `work_notes/notes.md` |

Each `merged.json` is recomputed exactly from its `parts/*/scores.json` by `../code/merge.py`. The strict
(`\boxed{}` only) scores are 2.8 unedited and 3.5 / 3.1 / 3.8 / 2.6 edited (`math500.acc`); the unedited model's
evaluation also has GSM8K (strict 4.0). Both tables are in `../eval/RESULTS_excerpt.md` ("Experiment 8").

## Files

The same layout as the main experiment (`../README.md`, "Files"): `runs/<run>/` with every turn (`turns/tNNN/`:
`prompt.txt`, `answer.md`, `events.jsonl`, `run.json`, `stderr.log`), `tool_calls.jsonl` (every Python run with code
and output, every forward-pass request, and `edit_hash` records of the edit at each pass), `requests/` (each forward
pass: request and returned summary), `explanations.md`, `final_answer.md`, `state.json`, `driver.log` / `loop.log`,
`work_notes/` (Astra's notebook, DEV / HOLDOUT ids, final report) and `edit/` (the submitted edit as an exact diff;
rebuild it with `../code/apply_edit.py` from `google/gemma-3-270m-it`'s `model.safetensors`). `eval/<run>/` holds the
evaluation: `merged.json`, and per job `scores.json`, `math500_per_problem.json` (per problem: correct per sample,
first extracted answer) and `math500_first_samples.jsonl` (the full text of each problem's first sample).

Other Gemma runs exist (g05, g10, g25, g50: the same budgets with a prompt that stated the strict `\boxed{}` grading;
rows in `../eval/RESULTS_excerpt.md`); they are not in the post and not included.
