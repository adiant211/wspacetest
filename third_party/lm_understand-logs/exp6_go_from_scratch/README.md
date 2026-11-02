# Experiment 6: writing a 1.5-million-weight Go network from scratch

**The question.** Can Astra write the weights of a network far too large to fit in its context (1,480,274 weights)
so that the network plays 9x9 Go well, without any training? And how much does it gain when it may run the network
a few times ("passes") to see what it does?

## Setup

- **Model:** gpt-6-astra through the Codex CLI (`codex exec --json`, one conversation per run, resumed turn by turn),
  reasoning effort `xhigh`. Codex's own tools (shell, web, file edits) were not used. The harness checked every turn
  for this (`"tool_free": true` in each `run.json`). Astra worked only through reply blocks that the harness ran:
  a ```` ```python ```` block (one sandbox run), a ```` ```games ```` block or a ```` ```positions ```` block (one pass each).
- **Network:** 243 -> 512 x 6 -> 82, ReLU, 1,480,274 numbers, all zero at the start. Input: 3 planes over the 81
  points (the mover's stones, the opponent's stones, White to move). Output: one logit per point and one for passing.
  Illegal moves are masked, and moves are sampled at temperature 1. There is no search.
- **Python sandbox:** numpy, scipy, scikit-learn, PyTorch (CPU), pandas, sympy and networkx. There is no Go library,
  no internet and no GPU. Each run gets up to 900 s and 96 GB, and Astra may make up to 120 runs. `load()` and `save()`
  read and write the 14 weight tensors, and a `/work` folder is kept between runs.
- **Passes (the budget: 0, 10, 25 or 50):**
  - A *games pass* plays up to 128 games of the current network against practice opponents: "random", "weak" (a
    REINFORCE network) and "strong" (a third PPO network, seed 1). Astra gets a summary for each opponent and every
    game's moves in `/work/pass_<k>/`.
  - A *positions pass* runs the network on up to 256 positions and returns its top moves, logits, probabilities and,
    on request, the hidden activations.
  - The session goes on until every pass is used: a final answer with passes left gets a reminder. With budget 0,
    Astra has no pass tool at all.
- **Not allowed:** training or fitting the weights to data (no gradient descent, regression or least squares on
  examples), search (no tree search, minimax or rollouts, either to choose moves or to make targets), and running
  the network outside the passes. There is no internet. Writing weights from its own design, Go-rule code that does
  not run the network, and analysis of the pass results were allowed. The full rules are in each run's
  `message.txt`.
- **Scoring (hidden from Astra):** 2,000 games against each of two small PPO networks (seeds 0 and 2,
  243 -> 32 -> 32 -> 82, trained for 20,000 iterations of 1,024 games). Astra never plays these networks. The written
  network is Black in half the games, and moves are sampled. The same evaluation also plays two REINFORCE networks
  and a random player, which are not shown in the post.
- **Same-size PPO baseline:** a 512 x 6 ReLU network (the same architecture) trained with PPO for 20,000 iterations
  of 1,024 games (20.5 M games). The learning rate is 3e-5, chosen on the practice opponent "strong" from a scan of
  1e-4, 3e-5 and 1e-5. There are three seeds (0, 1, 2). Each one played 2,000 head-to-head games against the written
  networks for budgets 0 and 50.

## Runs (prompt version v1; one run per budget)

| folder | original ID | passes used | Python runs | games / positions passes | turns | wall time |
|---|---|---|---|---|---|---|
| `runs/budget_0` | `v1_b0` | 0 | 1 | 0 / 0 | 2 | 22 min |
| `runs/budget_10` | `v1_b10` | 10 | 17 | 10 / 0 | 28 | 29 min |
| `runs/budget_25` | `v1_b25` | 25 | 30 | 24 / 1 | 56 | 57 min |
| `runs/budget_50` | `v1_b50` | 50 | 39 | 47 / 3 | 90 | 3 h 52 min (about 2 h 15 min of it a pause, see below) |

Wall time is the span of `loop.log`. The token counts are in the last turn's `run.json`; they are cumulative over the
conversation. A run with 1 pass and a second prompt version (v2) were also made but are not in the post, and they are
not included here.

**The pause in budget 50.** At turn 63 the harness could not resume the conversation for about 2 h 15 min (a disk
quota on our side). The failed attempts are kept as `turns/t063/fail*`. The same conversation then continued with
its context intact.

## The post's numbers and where they come from

| post | value | file |
|---|---|---|
| score vs hidden PPO seed 0 / seed 2, budget 0 | 54.1 % / 54.9 % | `eval/budget_0/eval_submission.json` (`ppo_s0.score`, `ppo_s2.score`) |
| same, budget 10 | 72.9 % / 60.0 % | `eval/budget_10/eval_submission.json` |
| same, budget 25 | 76.0 % / 62.3 % | `eval/budget_25/eval_submission.json` |
| same, budget 50 | 84.0 % / 80.8 % | `eval/budget_50/eval_submission.json` |
| head-to-head vs same-size PPO (seeds 0 / 1 / 2), budget 0 | 42.6 / 26.3 / 36.3 % | `eval/ppo_same_size/ppo_big_eval/h2h_lr_scan.json` (`v1_b0_vs_lr3e-5`, `..._s1`, `..._s2`); the same numbers are in `eval/budget_0/h2h_ppo_big_lr3e-5_s*_iter20000.json` |
| head-to-head vs same-size PPO, budget 50 | 74.2 / 70.0 / 78.5 % | same files, `v1_b50_vs_lr3e-5*`; `eval/budget_50/h2h_ppo_big_lr3e-5_s*_iter20000.json` |
| the sequence of turns (Python run / games pass / positions pass) | | `runs/budget_N/loop.log` |
| each turn's step in the figure | | `runs/budget_N/tool_calls.jsonl` |
| Astra's final description | | `runs/budget_N/final_answer.md` |
| "a single Python run of 336 lines" (budget 0) | | `runs/budget_0/tool_calls.jsonl` (the one entry's `code`) |

The scores are win rates over 2,000 games; a draw cannot happen (komi 7.5). "Wins more than half its games against
the smaller PPO network" is budget 0's 54.1 % and 54.9 %. "Beats the equal-sized PPO network" is budget 50's 70-79 %
against each seed.

Other evaluation files, not shown in the post:
- `eval/budget_N/analysis_strong.json`: 500 games against the practice opponent "strong", which was used to diagnose
  failures.
- `eval/budget_N/h2h_ppo_big_*`: head-to-heads against the same-size PPO networks. Here `lr1e-4_s0` is the
  learning-rate scan's other candidate, and `s<k>_iter<n>` are runs with the original, untuned settings (lr 3e-4),
  which the written networks beat 92-98 %.
- `eval/ppo_same_size/ppo_big/<checkpoint>/`: the untuned PPO networks through the same hidden evaluation.
- `eval/ppo_same_size/ppo_big_eval/eval_lr_scan.json`: the tuned networks through the same hidden evaluation. The
  `*.py` files there are the scripts that produced these numbers, kept as they were run, so their paths are those of
  our machine.

## Files

Per run (`runs/budget_N/`):
- `message.txt`: the task prompt (turn 1). The budget-0 prompt has no pass tools; the others differ only in the budget.
- `instructions.md`: the Codex instructions file.
- `turns/tNNN/`, one folder per turn:
  - `prompt.txt`: what the harness sent: the task, or the output of the last Python run or pass.
  - `answer.md`: Astra's reply.
  - `events.jsonl`: the raw Codex event stream. These runs' streams hold Astra's replies but no reasoning summaries.
  - `run.json`: the Codex call: command, model, effort, wall time, token usage, and the tool-use audit.
  - `stderr.log`: Codex's error output.
- `tool_calls.jsonl`: one line per action, in order:
  - each Python run: its code, printed output and seconds;
  - each pass: the request and the result summary that Astra was shown.
- `requests/`: the pass requests (`games_<k>.json`, `positions_<k>.json`).
- `work/`: the files Astra wrote into its `/work` folder: design scripts (`design*.py`, `policy_design*.py`) and
  coefficient files (`*config*.json`, `params*.json`). Copies of whole networks (`*.npz`, 5.9 MB each) are left out;
  the final network is in `networks/`.
- `work/pass_<k>/`, for each pass:
  - games pass: `summary.json` and `README.txt` (the file layout Astra was told). The per-game records it could also
    read (`games.json`, `boards.npy`, `valid.npy`; 92 MB and 132 MB over all runs) are left out.
  - positions pass: complete (inputs, legal moves, logits, probabilities, hidden activations).
- `state.json`: the harness's final state (budget, runs, passes, turns).
- `loop.log` (`run.log` is an identical copy): one line per turn.
- `final_answer.md`: the final reply.

Networks (`networks/`):
- `budget_N/`: the 14 tensors Astra saved, as evaluated (`L1.W.npy` ... `out.b.npy`, float32, 5.9 MB per network),
  and `arch.json`. Row i of a weight matrix belongs to input (or unit) i: `h = relu(x @ W + b)`.
- `ppo_same_size_lr3e-5_s{0,1,2}/`: the three same-size PPO networks at iteration 20,000, in the same layout.
