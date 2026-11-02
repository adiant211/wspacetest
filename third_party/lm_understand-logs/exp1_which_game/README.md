# Experiment 1: which game was this network trained on?

The question in the post: can GPT-6 Astra tell which game a policy network was trained on, from its weights alone,
without being told which games are possible?

## Setup

- **Networks.** Four PPO policies, one per game, on our own 10 x 10 grid versions of Breakout, Pong, Pac-Man and
  Freeway. Every game uses the same 5 input channels (agent, moving objects, reward targets, obstacles, previous frame)
  and the same 5 actions (no-op, up, down, left, right). Every network is the same MLP 500 -> 32 -> 32 -> 5 with tanh
  (17,253 weights), trained with the same PPO settings for 1 billion steps. Three seeds were trained per game; the post
  asks about the best seed of each game (highest mean return, table below).
- **Model.** gpt-6-astra through the Codex CLI (`codex exec`), reasoning effort `high`, reasoning summaries
  `detailed`. Every built-in tool was switched off and each run's event log was audited: no run made a tool call
  (`run.json` -> `audit.tool_free: true`).
- **What Astra got.** One network per fresh conversation, one reply. The prompt (`prompt.txt`) is the release
  description (`networks/description.txt`: the grid, the channels, the actions and the network formula, with no game
  named), the question, and all 17,253 weights as text, rounded to 3 decimals (`networks/<id>_actor.txt`).
- **Not allowed.** Tools of any kind (no code execution, no calculator). The question is open: no list of games is
  given.
- **Scoring.** The answer's `Game:` line is mapped to a game by keywords (Frogger, the classic road-crossing game,
  counts as Freeway). The answer key is read only by the scorer.

## Runs (the post's tabs)

Each network has a random ID; the ID is the only name Astra's harness used.

| Post tab | Run folder | Network ID | Seed | Astra's answer (`Game:` line) | Confidence | Correct |
|---|---|---|---|---|---|---|
| Breakout | `runs/breakout` | `a9f535cc` | 0 | Breakout, a small grid-based brick-breaking variant | 99 % | yes |
| Pong | `runs/pong` | `dfc6cb65` | 0 | Pong, most likely a single-player, vertical-paddle wall-Pong variant | 96 % | yes |
| Pac-Man | `runs/pacman` | `bdf99279` | 2 | Pac-Man, or a simplified grid-based Pac-Man clone | 96 % | yes |
| Freeway | `runs/freeway` | `0bbcc3ea` | 2 | Frogger, a simplified grid-based traffic-crossing variant | 97 % | yes (Frogger = Freeway) |

Run size (from `eval/scores.json` / `run.json`): about 73,700 input tokens per run; output 1,679 to 1,838 tokens, of
which 833 to 1,034 were reasoning tokens; 71 to 107 seconds per run.

## The post's numbers and where they come from

| In the post | Value | File |
|---|---|---|
| Right game for all four networks (open answer) | 4 / 4 | `eval/scores.json` (`correct`), checked against `eval/answer_key.json` |
| Breakout, "99% confident" (teaser) | 99 | `runs/breakout/answer.md` (`Confidence:`), `eval/scores.json` |
| Confidence on the other tabs | 96 / 96 / 97 | `runs/{pong,pacman,freeway}/answer.md` |
| Network size | 17,253 weights | `networks/description.txt`, `networks/meta/<id>.json` (`actor_parameters`) |
| PPO, 1 billion steps, same settings for every game | 999,997,440 steps | `networks/meta/<id>.json` (`env_steps`, `training_config`) |
| Tokens and reasoning tokens shown in the post | e.g. Breakout 1,679 / 833 | `runs/<game>/run.json` (`audit.usage`), `eval/scores.json` |
| Quoted weights (e.g. Breakout's 4.066 at (8,1)) | | `runs/<game>/answer.md`, `networks/<id>_actor.txt` |

**Best of 3 seeds.** Mean episode return of the trained policy (actions sampled), 256 environments x 1,000 steps, from
our training sanity check (the chosen seed in bold; a uniform random policy scores about -0.4, -1.0, 4.0 and 0.0):

| Game | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| Breakout | **21.71** | 15.98 | 15.83 |
| Pong | **4.96** | 4.95 | 4.66 |
| Pac-Man | 38.86 | 38.57 | **41.86** |
| Freeway | 24.00 | 24.00 | **24.04** |

The chosen networks' own numbers are in `networks/meta/<id>.json` (`final_performance`).

## Files

```
runs/<game>/
  prompt.txt        the full message Astra got (description + question + all weights)
  instructions.md   the Codex system instructions file (one line, the same for every run)
  events.jsonl      the raw Codex event stream: reasoning summaries ("reasoning") and the reply ("agent_message"),
                    plus token usage. The one "error" item is a Codex CLI start-up notice (its optional code-mode
                    host is disabled), not a model error
  answer.md         Astra's reply (Observations -> Reasoning -> Game -> Description -> Confidence)
  run.json          the run record: model, effort, the codex command line, token usage, wall time, the tool audit
  stderr.log        Codex CLI stderr (empty)
runs/loop.log, runs/worker0.log, runs/worker1.log
                    harness logs (two workers asked the networks in parallel)
networks/
  description.txt   the release description (the start of every prompt)
  <id>_actor.txt    the weights exactly as shown to Astra (3 decimals)
  <id>.npz          the full-precision network (actor and the critic used only in training; Astra saw only the actor)
  meta/<id>.json    ground truth and training record of each network: game, seed, PPO settings, steps, returns
eval/
  scores.json       the scorer's output per run: true game, mapped guess, correct, confidence, usage
  answer_key.json   network ID -> game and seed
```
