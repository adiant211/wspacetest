Submitted the frozen candidate in `/work/edited`, with two changes using zero-based indexing:

- **Layer 11 attention:** multiply `self_attn.o_proj.weight[:, 512:768]`—query head 2’s output block—by **1.1**.
- **Layer 11 MLP:** halve `mlp.down_proj.weight` columns **971, 180, 149, 1680, 232, 1724, 312, 1096**.

All other weights remain unchanged.

**Evidence and rationale:** Incorrect arithmetic and repeated reasoning motivated attention probes and within-problem, length-matched activation comparisons, followed by strength sweeps, opposite-direction interventions, and matched-neuron controls. Halving head 2 caused a **26.17-point DEV accuracy loss**, establishing its importance, while evidence that the selected MLP features specifically cause failures remained weaker. The final combination gained **2.49 percentage points on DEV** and was frozen before HOLDOUT, supporting—but not proving—the proposed rebalancing of operand retrieval and failure-associated MLP outputs.

**Independent confirmation:** Across **128 HOLDOUT problems and 16,384 answers per model**, accuracy increased from **9.052% to 9.473%**, a gain of **0.421 percentage points**, with problem-cluster **SE 0.365 points** and approximate 95% interval **[−0.300, +1.143] points**.

This suggests at most a modest benefit, not a statistically established improvement, and cutoffs worsened from **7.31% to 10.50%**.

The complete results and integrity audit are saved in `/work/final_report.json`.