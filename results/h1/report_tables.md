
### add

| arm | layers | 5-digit | 6-digit (unseen) | KL Wikipedia | KL chat | MMLU-200 | final train loss | trainable weights | precision |
|---|---|---|---|---|---|---|---|---|---|
| no fine-tune | - | 53.3% | 20.0% | 0.0000 | 0.0000 | 43.5% | - | - | bfloat16 |
| LoRA, all 24 layers | all | 88.7% | 80.5% | 0.1588 | 0.4428 | 34.5% | 0.043 | 4,399,104 | bfloat16 |
| LoRA, 4 layers: task-gradient tool | 2, 9, 14, 15 | 89.7% | 87.0% | 0.0338 | 0.0663 | 39.0% | 0.036 | 733,184 | bfloat16 |
| LoRA, 4 layers: task / generic gradient | 9, 13, 14, 15 | 90.0% | 83.5% | 0.0129 | 0.0433 | 41.5% | 0.047 | 733,184 | bfloat16 |
| LoRA, 4 layers: tool, upper half only | 12, 13, 14, 15 | 87.3% | 83.5% | 0.0162 | 0.0462 | 35.0% | 0.045 | 733,184 | bfloat16 |
| LoRA, 4 layers: Spectrum SNR (weights only) | 1, 3, 4, 7 | 90.0% | 89.5% | 0.0685 | 0.1646 | 39.5% | 0.036 | 733,184 | float32 |
| LoRA, 4 arbitrary layers | 12, 21, 22, 23 | 88.3% | 78.5% | 0.0250 | 0.0649 | 41.5% | 0.045 | 733,184 | bfloat16 |
| LoRA, 4 random layers (a) | 4, 10, 12, 20 | 88.3% | 87.5% | 0.0523 | 0.1521 | 40.0% | 0.039 | 733,184 | bfloat16 |
| LoRA, 4 random layers (b) | 1, 2, 3, 17 | 91.0% | 88.5% | 0.1129 | 0.1054 | 42.5% | 0.033 | 733,184 | bfloat16 |
| LoRA, 4 layers the tool ranks lowest | 19, 20, 22, 23 | 64.0% | 31.0% | 0.0198 | 0.0125 | 42.5% | 0.195 | 733,184 | bfloat16 |
| 64 + 64 neurons: task-gradient tool | 2, 3 | 91.0% | 86.5% | 0.1777 | 0.1476 | 41.5% | 0.040 | 114,688 | float32 |
| 64 + 64 random neurons | 2, 3 | 81.7% | 67.0% | 0.0004 | 0.0136 | 45.0% | 0.093 | 114,688 | float32 |

Clean cost benchmark, add: every arm alone on the same idle 4-core CPU, float32, 8 timed steps after 3 warm-up steps.

| arm | backprop stops at layer | trainable weights | Adam state (MB) | s / step | vs all-layer | one-off prefix cache (s) | est. full run (s) | peak RSS (MB) |
|---|---|---|---|---|---|---|---|---|
| LoRA, all 24 layers | 0 | 4,399,104 | 33.6 | 4.84 | - | 0 | 725 | 6819 |
| LoRA, 4 layers: task-gradient tool | 2 | 733,184 | 5.6 | 4.16 | 1.2x faster | 43 | 666 | 5470 |
| LoRA, 4 layers: task / generic gradient | 9 | 733,184 | 5.6 | 3.03 | 1.6x faster | 165 | 620 | 4536 |
| LoRA, 4 layers: tool, upper half only | 12 | 733,184 | 5.6 | 2.26 | 2.1x faster | 215 | 554 | 4226 |
| LoRA, 4 layers: Spectrum SNR (weights only) | 1 | 733,184 | 5.6 | 4.12 | 1.2x faster | 18 | 635 | 5514 |
| LoRA, 4 arbitrary layers | 12 | 733,184 | 5.6 | 2.33 | 2.1x faster | 215 | 565 | 4347 |
| LoRA, 4 random layers (a) | 4 | 733,184 | 5.6 | 3.70 | 1.3x faster | 75 | 630 | 5221 |
| LoRA, 4 random layers (b) | 1 | 733,184 | 5.6 | 3.96 | 1.2x faster | 18 | 612 | 5413 |
| LoRA, 4 layers the tool ranks lowest | 19 | 733,184 | 5.6 | 1.10 | 4.4x faster | 334 | 499 | 3429 |
| 64 + 64 neurons: task-gradient tool | 2 | 114,688 | 0.9 | 3.71 | 1.3x faster | 38 | 595 | 5073 |
| 64 + 64 random neurons | 2 | 114,688 | 0.9 | 3.57 | 1.4x faster | 33 | 568 | 5144 |

### french

| arm | layers | in French | capitals right | KL Wikipedia | MMLU-200 | final train loss | trainable weights | precision |
|---|---|---|---|---|---|---|---|---|
| no fine-tune | - | 0.0% | 100.0% | 0.0000 | 42.0% | - | - | float32 |
| LoRA, all 24 layers | all | 100.0% | 77.1% | 0.1962 | 38.5% | 0.301 | 4,399,104 | float32 |
| LoRA, 4 layers: task-gradient tool | 0, 2, 3, 4 | 97.9% | 75.0% | 0.4596 | 27.5% | 1.071 | 733,184 | float32 |
| LoRA, 4 layers: task / generic gradient | 5, 10, 11, 12 | 99.0% | 52.1% | 0.0398 | 34.5% | 0.678 | 733,184 | float32 |
| LoRA, 4 layers: Spectrum SNR (weights only) | 1, 3, 4, 7 | 99.0% | 43.8% | 0.2527 | 40.5% | 0.731 | 733,184 | float32 |
| LoRA, 4 random layers (a) | 4, 10, 12, 20 | 99.0% | 54.2% | 0.0607 | 37.0% | 0.596 | 733,184 | float32 |
| LoRA, 4 layers the tool ranks lowest | 18, 19, 20, 22 | 100.0% | 89.6% | 0.0995 | 44.0% | 0.535 | 733,184 | float32 |

Clean cost benchmark, french: every arm alone on the same idle 4-core CPU, float32, 8 timed steps after 3 warm-up steps.

| arm | backprop stops at layer | trainable weights | Adam state (MB) | s / step | vs all-layer | one-off prefix cache (s) | est. full run (s) | peak RSS (MB) |
|---|---|---|---|---|---|---|---|---|
| LoRA, all 24 layers | 0 | 4,399,104 | 33.6 | 5.09 | - | 0 | 611 | 7388 |
| LoRA, 4 layers: task-gradient tool | 0 | 733,184 | 5.6 | 4.75 | 1.1x faster | 0 | 570 | 5959 |
| LoRA, 4 layers: task / generic gradient | 5 | 733,184 | 5.6 | 3.93 | 1.3x faster | 26 | 498 | 5713 |
| LoRA, 4 layers: tool, upper half only | 12 | 733,184 | 5.6 | 2.92 | 1.7x faster | 62 | 412 | 4791 |
| LoRA, 4 layers: Spectrum SNR (weights only) | 1 | 733,184 | 5.6 | 4.70 | 1.1x faster | 6 | 570 | 5955 |
| LoRA, 4 random layers (a) | 4 | 733,184 | 5.6 | 4.13 | 1.2x faster | 22 | 517 | 5710 |
| LoRA, 4 layers the tool ranks lowest | 18 | 733,184 | 5.6 | 1.91 | 2.7x faster | 93 | 323 | 4180 |
