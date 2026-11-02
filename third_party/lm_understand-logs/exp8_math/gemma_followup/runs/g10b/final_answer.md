Submitted the frozen edit in `/work/edited`: **1.5× amplification** of these zero-indexed `mlp.down_proj.weight` columns:

- Layer 8: **699**
- Layer 14: **483, 984**
- Layer 17: **1179**

Only **2,560 weight values** changed; all other entries, embeddings, output weights, and norms remain untouched.

Baseline analysis and teacher-forced probes identified this repetition-associated channel group, but causal tests reversed the initial hypothesis: removing it increased repetitive conclusions, while strengthening it reduced them. The 1.5× strength had the best paired DEV accuracy and was frozen before HOLDOUT.

Across **3,072 HOLDOUT answers per model**, accuracy increased from **7.19% to 7.91%**: **+0.72 percentage points, SE 0.49**, with an approximate 95% interval of **−0.25 to +1.68 points**, so the accuracy gain remains inconclusive. Repeated-conclusion responses fell from **15.33% to 12.01%**, supporting improved conclusion control rather than newly acquired mathematical knowledge.

The complete audit and limitations are saved in `/work/FINAL_REPORT.json`.