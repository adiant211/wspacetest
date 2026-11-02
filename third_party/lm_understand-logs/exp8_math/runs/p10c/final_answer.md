Kept **30 rank-one edits** in `/work/edited`: attention and MLP output projections in layers **8–22**, applying \(W'=W-1.5u(u^\top W)\) to repetition-associated directions. Embeddings, output head, and final norm are untouched.

**Path:** Diagnosed repetitive thinking as the dominant failure. Rejected ineffective neuron and copying-head ablations, then identified matched-content repetition directions while protecting ordinary-reasoning and answer-transition directions. Tested two strengths and froze the candidate before validation.

**512 unseen practice problems, paired evaluation:**

| Source | Original | Edited |
|---|---:|---:|
| GSM8K | 16.0% | 18.8% |
| MATH | 8.2% | 12.9% |
| Combined | 12.1% | 15.8% |

Combined gain: **3.71 percentage points**, paired SE **1.79**. Cutoffs decreased from **88.9% to 82.2%**; successful edited answers still had substantial reasoning.

Evidence is **modest**: the second holdout was neutral, and the pooled exact paired test was borderline. The edit narrowly met the prespecified retention rule. I expect it to help by reducing repetition, not by adding mathematical knowledge or forcing thinking to end.

Details and reconstruction instructions: `/work/final_manifest.json` and `/work/notes.md`.