Submitted **column 920 of `model.layers.17.mlp.down_proj.weight` multiplied by 2**; only 640 weight values changed.

I rejected broader neuron, attention-head, and repetition-direction edits after they hurt accuracy. Neuron 920 showed numerical selectivity, within-problem correctness associations, and favorable correct-digit alignment; doubling it was the best tested DEV candidate, though its pooled gain remained negative.

The edit was frozen before HOLDOUT evaluation. Across **3,072 answers per model**, accuracy changed from **8.11% to 8.01%**: **−0.10 percentage points, SE 0.53**, with a 95% interval of **[−1.13, +0.94] points**. Cutoffs increased from 7.10% to 9.77%.

The intended repair strengthens a numerical-answer feature, but **the experiments did not demonstrate improved mathematics performance**.