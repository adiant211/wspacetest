Kept one edit in `/work/edited`: **zeroed columns `[1792:2048]` of `model.language_model.layers.15.self_attn.o_proj.weight`**, disabling zero-indexed attention head 7’s output; all other weights remain unchanged.

**Diagnosis and causal evidence:** Baseline failures were dominated by repetitive thinking and token-limit cutoffs. Initial MLP interventions did not improve accuracy; subsequent localization identified this head’s strong retrieval of repeated continuations. Weakening it suppressed loops dose-dependently, amplifying it reinforced them, and a same-layer control head did not reproduce its selective effect.

**Frozen-edit validation on 768 untouched practice problems:**

| Set | Original correct | Edited correct | Accuracy gain ± SE |
|---|---:|---:|---:|
| GSM8K practice | 58/384 | 110/384 | +13.54 ± 2.32 points |
| MATH practice | 29/384 | 41/384 | +3.13 ± 1.64 points |
| **Combined** | **87/768** | **151/768** | **+8.33 ± 1.43 points** |

The combined gain’s approximate 95% confidence interval is **+5.53 to +11.14 points**, with 95 improvements versus 31 regressions. Of those improvements, 94 rescued original token-limit failures, supporting the proposed mechanism. I expect better completion of mathematical reasoning, principally on grade-school problems; the MATH-specific benefit remains uncertain, and substantial repetition still remains.

Full results and audit: `/work/final_results.json`; experiment history: `/work/notes.md`.