# Experiment 5: writing MinAtar networks from scratch, in one reply, with no tools

**The question.** Can an LLM write the weights of a working neural network from scratch, without any training?

**The setup.** For three MinAtar games (Young & Tian 2019) — Breakout, Seaquest and Asterix — we asked GPT-6 Astra
(`gpt-6-astra`, run through the Codex CLI, `codex exec`, reasoning effort `xhigh`, reasoning summaries `detailed`)
to write every weight of a policy network that plays the game as well as possible.

- One fresh conversation per game, one reply, no feedback.
- The prompt gives: the task; the game's rules plus the game's exact source file from the MinAtar package
  (version 1.0.15, `minatar/environments/<game>.py`); the input (a 10 x 10 grid in 6 binary channels: player,
  danger, reward, projectile, trail, gauge) and the output (6 action logits); the network; how it is scored; and a
  small "edits" language for writing weights.
- The network: 600 -> 32 -> 32 -> 6, tanh, 20,486 parameters — the same shape as the PPO networks it is compared
  with. **Every weight starts at 0**; the reply's last `edits` block sets the weights.
- Not allowed / not available: no training, no data, no loss, no gradients, and no tools (no code execution, no
  calculator). The network never plays the game before it is scored. Every Codex tool was turned off, and the event
  stream of every run was checked: no tool call in any run (`"tool_free": true` in `reply/run.json`).

**Scoring.** The same for the written networks and for PPO: mean score over 1,000 episodes of the official MinAtar
1.0.15 game, actions sampled from the softmax of the logits, sticky actions 0.1, episodes cut at 10,000 steps.

**The PPO baselines** (results only; their checkpoints are not included): the same 600-32-32-6 tanh actor, trained
with PPO (CleanRL style: learning rate 5e-4 annealed, 64 games x 128 steps per update, 4 epochs, 8 minibatches,
gamma 0.99, lambda 0.95, clip 0.2, entropy 0.01) on the same observation, for 100M steps (Breakout and Asterix seeds
0-9, Seaquest seeds 0-2) and 500M steps (seeds 0-2). The post shows the best seed of each.

## The post's numbers and where they come from

All numbers are `mean` values in `eval/minatar_eval.json` (each entry also has the standard error, the episode count
and the mean episode length).

| Game | PPO, 100M steps | PPO, 500M steps | Astra, one-shot |
|---|---|---|---|
| Breakout | 15.8 (`breakout.ppo100_s2`) | 25.3 (`breakout.ppo500_s0`) | **517.4** (`breakout.hand`) |
| Seaquest | 5.6 (`seaquest.ppo100_s2`) | **88.9** (`seaquest.ppo500_s2`) | 53.1 (`seaquest.hand`) |
| Asterix | 25.8 (`asterix.ppo100_s6`) | **55.2** (`asterix.ppo500_s1`) | 53.2 (`asterix.hand`) |

`hand` is the network Astra wrote (`networks/<game>/hand.npz`); `ppo100_s<k>` / `ppo500_s<k>` are the PPO networks
with seed k after 100M / 500M steps.

Run size (from `reply/run.json`):

| Run | Output tokens (of which reasoning) | Wall time | Statements in the edits block | Nonzero weights |
|---|---|---|---|---|
| `breakout` | 20,567 (16,828) | 501 s | 92 | 815 of 20,486 |
| `seaquest` | 23,894 (19,803) | 545 s | 122 | 803 of 20,486 |
| `asterix` | 28,993 (25,694) | 583 s | 55 | 637 of 20,486 |

## Files

```
runs/<game>/                 one run per game: breakout, seaquest, asterix
  message.txt                the prompt Astra got (identical to reply/prompt.txt)
  instructions.md            the Codex model-instructions file used for the run
  reply/prompt.txt           the prompt as sent
  reply/events.jsonl         Codex's raw event stream: Astra's reasoning summaries and its final message. The first
                             item, "Code Mode is unavailable ...", is a Codex CLI notice, not part of the answer.
  reply/answer.md            Astra's reply: its design, then the `edits` block that sets the weights
  reply/run.json             run record: model, reasoning effort, token usage, wall time, tool audit
  reply/stderr.log           Codex's stderr (empty)
  loop.log, run.log          harness logs (question sent, reply received, first reading of the reply)
  apply.json                 how the reply was read into the network: statements read, errors, nonzero weights,
                             largest weight per tensor, token usage
networks/<game>/hand.npz     the network Astra wrote, as read from reply/answer.md
eval/minatar_eval.json       the 1,000-episode scores of the written networks and of the PPO networks
LICENSE-MinAtar              GPL-3.0 license text (see below)
```

**Reading the reply.** The Seaquest and Asterix replies wrote some 2-D lists as `[ [ ... ], ... ]` (a space after the
first bracket), which our first reader rejected; `run.log` shows that first reading (Seaquest 116 of 122 statements,
Asterix 51 of 55). The reader was fixed to accept that form and the same replies were read again (no new question
was asked): `apply.json` shows the final reading, all statements read with 0 errors (Breakout 92 / 92, Seaquest
122 / 122, Asterix 55 / 55). The networks in `networks/` are that final reading, and they are the networks that were
scored; reading `reply/answer.md` again gives exactly the same weights.

**The network files.** `hand.npz` (numpy) holds six float32 arrays:
`actor/dense1/kernel` W1 (600 x 32), `actor/dense1/bias` b1 (32), `actor/dense2/kernel` W2 (32 x 32),
`actor/dense2/bias` b2 (32), `actor/logits/kernel` W3 (32 x 6), `actor/logits/bias` b3 (6). The network computes
`h1 = tanh(x @ W1 + b1)`, `h2 = tanh(h1 @ W2 + b2)`, `logits = h2 @ W3 + b3`, where x is the 600-number observation
with input index `(row * 10 + column) * 6 + channel` (rows 0 top to 9 bottom), and the logits are for the actions
0 no-op, 1 left, 2 up, 3 right, 4 down, 5 fire. Which MinAtar channels go into each of the 6 channels is written in
each prompt (`message.txt`, "THE INPUT").

## License of the MinAtar source in the prompts

Each prompt (`message.txt`, `reply/prompt.txt`) contains the source file of the game from the MinAtar package
(https://github.com/kenjyoung/MinAtar, version 1.0.15, by Kenny Young and Tian Tian), which is licensed under the GNU
General Public License v3.0. The full license text is in `LICENSE-MinAtar` in this folder. The source is included
unchanged, with its original author header.
