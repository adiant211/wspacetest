Submitted the frozen edit in `/work/edited`: **64 selected MLP `down_proj` columns across layers 6–17 are halved**, with all other weights unchanged.

Within-problem activation analysis associated these channels with arithmetic errors after controlling for length and formatting. Full removal harmed DEV accuracy (−4.59 points); half suppression was the better tested candidate (−0.49 points), so I froze it before HOLDOUT.

**HOLDOUT accuracy improved from 8.79% to 9.96%: +1.17 percentage points, SE 0.90**, across 1,024 answers per model. The approximate 95% interval is **[−0.59, +2.94] points**, so this suggests possible benefit—not a statistically established improvement or proven arithmetic repair.

The teacher-forcing input mistake, complete evidence, and limitations are documented in `/work/notes.md` and `/work/final_report.json`.