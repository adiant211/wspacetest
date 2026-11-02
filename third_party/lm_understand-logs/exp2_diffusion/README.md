# Experiment 2: what does this diffusion model generate?

The question in the post: can GPT-6 Astra tell what a small diffusion model generates, from its weights alone, with no
tools and without computing any forward pass?

## Setup

- **Models.** Three DDPMs for 12 x 12 grayscale images, trained on filled circles, filled squares and filled
  triangles (one shape per model, random size and position). Each is the same MLP 160 -> 64 -> 64 -> 144 with ReLU
  (23,824 weights): input = the noisy image (144 pixels) and the diffusion step (16 sine / cosine numbers), output =
  the predicted clean image. Trained with Adam, 40,000 steps, batch 512.
- **Model asked.** gpt-6-astra through the Codex CLI (`codex exec`), reasoning effort `high`, reasoning summaries
  `detailed`. Every built-in tool was switched off and each run's event log was audited: no run made a tool call
  (`run.json` -> `audit.tool_free: true`).
- **What Astra got.** One model per fresh conversation, one reply. The prompt (`prompt.txt`) is the release
  description (`networks/description.txt`: the DDPM, the noise schedule, the network formula, with no shape named),
  the question, and all 23,824 weights as text, rounded to 3 decimals (`networks/<id>_weights.txt`).
- **Not allowed.** Tools of any kind, and any forward-pass calculation: the prompt says "Do not do any forward-pass
  calculation: do not run the network on any input (in your head or on paper), and do not simulate the sampling. Read
  what the weights themselves show." The question is open: no list of shapes is given.
- **Scoring.** The answer's `Images:` line is mapped to a shape by keywords (circle / disc / round / oval; square /
  rectangle / box; triangle / pyramid / wedge). The answer key is read only by the scorer.

## Runs (the post's tabs)

| Post tab | Run folder | Model ID | Astra's answer (`Images:` line, shortened) | Confidence | Correct |
|---|---|---|---|---|---|
| Circles | `runs/circle` | `d88aba69` | a single white circle outline on black, with a black interior, roughly 8-10 pixels across | 84 % | yes |
| Squares | `runs/square` | `a6a9dfb7` | white, approximately one-pixel-thick rectangular outlines on black | 77 % | yes |
| Triangles | `runs/triangle` | `246a5b58` | a single white, hollow, upward-pointing triangle on black | 88 % | yes |

Astra named the right shape for all three models, but drew every shape as an outline, while the models generate filled
shapes (compare its `Example:` drawing in `answer.md` with `eval/samples/<id>.txt` and `eval/samples.png`). The post
discusses this.

Run size (from `eval/scores.json` / `run.json`): about 99,800 input tokens per run; output 1,541 to 3,149 tokens, of
which 1,034 to 2,588 were reasoning tokens; 71 to 124 seconds per run.

## The post's numbers and where they come from

| In the post | Value | File |
|---|---|---|
| Right shape for all three models (open answer) | 3 / 3 | `eval/scores.json` (`correct`), checked against `eval/answer_key.json` |
| Confidence on each tab | 84 / 77 / 88 | `runs/<shape>/answer.md` (`Confidence:`), `eval/scores.json` |
| Astra's example drawings (outlines) | | `runs/<shape>/answer.md` (`Example:`) |
| What the models really generate (filled shapes) | | `eval/samples/<id>.txt`, `eval/samples.png`, `eval/answer_key.json` (`quality`) |
| Network size | 23,824 weights | `networks/description.txt`, `networks/training/<shape>_s0.json` (`params`) |
| Training: 40,000 steps | | `networks/training/<shape>_s0.json` (`steps`, loss curve) |
| Tokens and reasoning tokens shown in the post | e.g. circle 2,778 / 2,272 | `runs/<shape>/run.json` (`audit.usage`), `eval/scores.json` |

`eval/answer_key.json` also records a check that each model really makes its shape: 1,000 generated samples sorted by
a small shape classifier (circle model 99.4 % circles, square model 98.4 % squares, triangle model 100 % triangles).

## Files

```
runs/<shape>/
  prompt.txt        the full message Astra got (description + question + all weights)
  instructions.md   the Codex system instructions file (one line, the same for every run)
  events.jsonl      the raw Codex event stream: reasoning summaries ("reasoning") and the reply ("agent_message"),
                    plus token usage. The one "error" item is a Codex CLI start-up notice (its optional code-mode
                    host is disabled), not a model error
  answer.md         Astra's reply (Observations -> Reasoning -> Images -> Example -> Confidence)
  run.json          the run record: model, effort, the codex command line, token usage, wall time, the tool audit
  stderr.log        Codex CLI stderr (empty)
runs/loop.log       harness log (the three models were asked one after another)
networks/
  description.txt   the release description (the start of every prompt)
  <id>_weights.txt  the weights exactly as shown to Astra (3 decimals)
  <id>.npz          the full-precision network (W1, b1, W2, b2, W3, b3)
  training/<shape>_s0.json
                    training record: steps, noise schedule, loss curve, sample quality
eval/
  scores.json       the scorer's output per run: true shape, mapped guess, correct, confidence, usage
  answer_key.json   model ID -> shape, and the sample-quality check
  samples/<id>.txt  8 samples generated by each model, as ASCII art (never shown to Astra)
  samples.png       per model: 8 training images, 8 generated samples, and Astra's example drawing
```
