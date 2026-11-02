# Experiment 4: fix a failing car network by editing its weights

The question in the post: can GPT-6 Astra fix a trained network that fails, by reading its weights and writing new
ones, in a single reply and with no tools?

## Setup

- **Task.** The car-on-ramp level of Kinetix (`l/h2_car_ramp`): a small car with two motor-driven wheels has to back
  up, then drive left fast enough to jump over a red strip and reach the blue box. Success = the green wheel touches
  the blue box; touching red is a failure; an episode lasts at most 256 steps.
- **The failing network.** The official Kinetix actor-critic (actor: 679 -> 128 (ReLU) -> five layers of 128 (tanh)
  -> 16 logits; 171,664 actor weights), trained with the official Kinetix PPO for 1,001,390,080 environment steps. Its
  final network drives onto the ramp and stalls there: 0 successes in 256 episodes (every one times out). Every
  evaluation during training also had 0 successes.
- **Model.** gpt-6-astra through the Codex CLI (`codex exec`), reasoning effort `high`. Every built-in tool was
  switched off and the event log was audited: no tool call (`run.json` -> `audit.tool_free: true`). This run was made
  without the reasoning-summary option, so `events.jsonl` holds the reply but no reasoning summaries; the token count
  shows 9,146 reasoning tokens.
- **What Astra got** (`reply/prompt.txt`, one message, one reply): the task, a full description of the scene, the
  rules, the reward and the 679-number observation layout; the actor's formula; all 171,664 actor weights (each
  tensor written as integers times one scale, 3 significant digits of the tensor's largest entry; the value head is
  not shown); and the current network's result on 256 episodes with one recorded episode (positions every 16 steps).
  It was asked to first explain why the network fails, then keep what works and fix only the cause, with success
  coming first.
- **Not allowed.** Tools of any kind (no code execution, calculator or simulator), any feedback, or a second try.
- **How the edit is applied.** Astra writes edit statements in a small edit language (described in the prompt, e.g.
  `actor0.W[6, 122] = 80`). The harness applies them, in order, to the network exactly as shown (the rounded weights)
  and scores the result on 256 "search" episodes (the episode stream the base network's result in the prompt comes
  from) and on 1,024 separate test episodes, with actions sampled from the policy. The post reports the 1,024 test
  episodes.

## The run

`runs/oneshot_repair` is the single run in the post (no run name appears in the post; this is the harness's name for
the one-shot run with the "explain the failure, then fix only its cause" instruction).

Astra's diagnosis: a velocity-sensitive throttle / coast controller that loses its momentum on the ramp and alternates
throttle and coasting; reverse is effectively never sampled, so it never makes a run-up. Its fix (51 edit statements):
a small separate path through a few units per layer that reads the car body's x position and x velocity and the green
wheel's x position, backs the car up from rest, switches to full drive left around body x = 3.8, and switches off once the
green wheel has passed x = 0.95, so the trained network drives the rest.

## The post's numbers and where they come from

| In the post | Value | File |
|---|---|---|
| The trained network never succeeds in 256 episodes (stalls on the ramp) | 0 / 256, every episode a timeout | `eval/base_result.json` (the network as shown to Astra); also `eval/ppo_training_result.json` -> `last_search` |
| PPO, 1 billion steps | 1,001,390,080 steps | `eval/ppo_training_result.json` (`train_env_steps`) |
| After Astra's edit: success in all 1,024 episodes ("100% success, up from 0%") | 1,024 / 1,024 | `eval/oneshot_result.json` -> `test` (`episodes` 1024, `success` 1.0) |
| (also on the 256 shown episodes, and the most-likely-action episode) | 256 / 256 | `eval/oneshot_result.json` -> `search` |
| Changes 3.5% of the weights | 6,073 of 171,664 (3.54 %) | `eval/oneshot_result.json` -> `changes` (`changed`, `frac_changed`, per tensor) |
| One try, no tools | tool-free, one reply | `runs/oneshot_repair/reply/run.json` (`audit`), `runs/oneshot_repair/loop.log` |
| Tokens shown in the post | 10,134 output, 9,146 reasoning | `runs/oneshot_repair/reply/run.json` (`audit.usage`) |
| Astra's reasoning and edits quoted in the post | | `runs/oneshot_repair/reply/answer.md` |

Most of the 6,073 changed entries are the rows and columns of the few units Astra took over, which it set to zero
before wiring its path (`answer.md`, e.g. `actor0.W[:, 122:128] = 0`). The changes are counted against the network as
shown (rounded); `networks/edited_actor.npz` reproduces the count.

## Files

```
runs/oneshot_repair/
  reply/prompt.txt     the full message Astra got (task, scene, rules, network formula, all actor weights, result)
  reply/message.txt    the same message (the harness saves it under both names; the files are identical)
  reply/events.jsonl   the raw Codex event stream: the reply ("agent_message") and token usage. The one "error"
                       item is a Codex CLI start-up notice (its optional code-mode host is disabled), not a model error
  reply/answer.md      Astra's reply: the diagnosis, the edits block, and how the fix works
  reply/run.json       the run record: model, effort, the codex command line, token usage, wall time, the tool audit
  reply/stderr.log     Codex CLI stderr (empty)
  instructions.md      the Codex system instructions file (one line)
  loop.log             harness summary: edits applied (51 of 51, 0 errors), tensors touched, search and test success
  oneshot_repair.log   harness console output (library warnings from the physics simulator, then the same summary)
networks/
  ppo_final_seed0.npz  the trained PPO network Astra was shown (full precision, actor and value head; Kinetix
                       parameter names: params/Dense_0 = embed, DenseAndActivation_0..4 = actor0..4,
                       GeneralActorCriticRNN_0/Dense_0 = actor_head; DenseAndActivation_5..9 and Dense_1 = value head)
  edited_actor.npz     the actor after Astra's edits (embed.W, embed.b, actor0..4, actor_head), as scored
eval/
  base_result.json     the shown network's result on 256 episodes (the result text that is in the prompt)
  oneshot_result.json  the edit report (statements applied, tensors touched), the size of the change, and the
                       results of the edited network on the 256 episodes and the 1,024 test episodes
  ppo_training_result.json
                       the PPO training record: settings, steps, and the evaluations every ~50M steps (all 0 %
                       success). Its "best_*" and "test" fields are for the training run's own model-selection rule,
                       which picked the untrained start (chunk 0); the network Astra got is the final one (chunk 20,
                       `last_search`)
```
