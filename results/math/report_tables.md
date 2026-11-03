## Held-out test (GSM8K test problems, new arithmetic, Wikipedia test articles)

_no test results yet_

## Screen (held-in, used to choose)

Base: arithmetic 7.7% (sym 15.5 / EN 4.3 / TR 3.3); GSM8K calculation results 27.5% EN / 17.3% TR; Wikipedia NLL 2.438 EN / 2.986 TR. Base on the 2-digit problems unseen in training: 2.7%; on the 1-digit facts: 18.3%.

| arm | neurons | arithmetic, all (pts) | 2-digit, unseen (pts) | 1-digit facts (pts) | GSM8K calc EN (pts) | GSM8K calc TR (pts) | GSM8K solution NLL EN / TR | Wikipedia NLL EN | Wikipedia NLL TR |
|---|---|---|---|---|---|---|---|---|---|
| manual: one-shot gradient gains +-50% (256) | 256 | +2.8 ± 0.8 | -1.1 ± 0.5 | +11.1 ± 2.5 | -7.0 ± 3.9 | -0.9 ± 2.7 | +0.024 / +0.021 | +0.003 ± 0.001 | +0.002 ± 0.001 |
| manual: one-shot gradient gains +-25% (1,024 neurons) | 1,024 | +3.0 ± 0.9 | -0.5 ± 0.6 | +11.1 ± 2.6 | -8.3 ± 3.9 | -1.4 ± 2.8 | +0.027 / +0.029 | +0.002 ± 0.000 | +0.001 ± 0.001 |
| manual: one-shot gradient gains +-50% (1,024) | 1,024 | +0.0 ± 0.9 | -0.5 ± 0.7 | +0.6 ± 2.5 | -19.6 ± 3.6 | -6.1 ± 3.8 | +0.135 / +0.147 | +0.007 ± 0.001 | +0.004 ± 0.001 |
| manual: one-shot gradient gains +-25% (4,096) | 4,096 | +0.2 ± 0.8 | -0.5 ± 0.7 | +0.6 ± 2.4 | -18.4 ± 3.7 | -7.6 ± 3.7 | +0.109 / +0.122 | +0.004 ± 0.001 | +0.002 ± 0.001 |
| control: random gains +-50% (1,024 random neurons) | 1,024 | +0.1 ± 0.2 | -0.1 ± 0.1 | +0.6 ± 0.6 | +0.6 ± 1.1 | -0.7 ± 0.7 | -0.003 / +0.001 | +0.001 ± 0.000 | +0.001 ± 0.000 |
| knock-out: 1,024 math-firing neurons | 1,024 | +0.3 ± 0.4 | +0.4 ± 0.4 | +0.6 ± 1.3 | -1.0 ± 2.1 | -0.6 ± 1.8 | +0.040 / +0.048 | +0.001 ± 0.001 | +0.004 ± 0.001 |
| knock-out: 1,024 random neurons | 1,024 | +0.1 ± 0.3 | +0.0 ± 0.3 | +0.6 ± 0.8 | +0.6 ± 0.8 | +0.3 ± 1.4 | -0.002 / +0.006 | +0.003 ± 0.001 | +0.002 ± 0.001 |
| manual: math-firing neurons x1.5 (1,024) | 1,024 | -0.3 ± 0.3 | -0.4 ± 0.3 | -0.3 ± 0.8 | -4.1 ± 2.0 | -0.5 ± 0.8 | +0.001 / +0.001 | +0.001 ± 0.000 | +0.000 ± 0.000 |
| manual: MathNeuro x1.1 (5,010 math-specific neurons) | 5,010 | +0.2 ± 0.2 | +0.0 ± 0.0 | +0.6 ± 0.7 | +1.2 ± 1.3 | +0.5 ± 0.8 | -0.002 / +0.000 | +0.000 ± 0.000 | -0.000 ± 0.000 |
| manual: MathNeuro x1.25 | 5,010 | +0.2 ± 0.4 | +0.1 ± 0.4 | +0.6 ± 1.2 | +0.8 ± 2.7 | +0.5 ± 1.5 | +0.004 / +0.009 | +0.001 ± 0.001 | -0.000 ± 0.000 |
| trained: LoRA r=8 on 4,096 targeted neurons, 50 steps | 4,096 | +6.8 ± 0.9 | +1.0 ± 0.7 | +20.8 ± 2.7 | +3.5 ± 3.7 | -0.5 ± 3.1 | +0.083 / +0.089 | +0.012 ± 0.001 | +0.007 ± 0.001 |
| trained: LoRA r=8 on 4,096 targeted neurons | 4,096 | +11.2 ± 1.1 | +3.0 ± 0.9 | +29.2 ± 2.8 | +8.3 ± 3.7 | +2.2 ± 2.2 | +0.122 / +0.157 | +0.010 ± 0.001 | +0.007 ± 0.002 |

## Wall time

| step | what | wall time | threads |
|---|---|---|---|
| find the neurons | one forward + backward scan of all 129,024 neurons over 342 held-in texts (55,453 tokens) | 30 min | 4 |
| make + screen one manual edit | no training; benchmark on the held-in split | 5.2 min each (13 screened) | 4 |
| train gain on grad4096 | 100 steps, 4,096 trainable params | 27 min (16.2 s/step) | 2 |
| train lora on all | 100 steps, 1,376,256 trainable params | 29 min (17.1 s/step) | 2 |
| train lora on grad4096 | 100 steps, 376,832 trainable params | 29 min (17.3 s/step) | 2 |
| train lora on rand4096 | 100 steps, 376,832 trainable params | 28 min (16.6 s/step) | 2 |
