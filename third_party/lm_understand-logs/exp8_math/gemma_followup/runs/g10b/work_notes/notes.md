# Weight-edit experiment

No edits have been made. Embedding/output and final norm are excluded.
HOLDOUT is fixed in HOLDOUT.json and disjoint from BASELINE_POOL.json.
DEV will be fixed from baseline problems with 1–7 successes out of 8.

## Pass 1 planned: baseline diagnosis
Model: original. Exact edit: none; strength 0.
Set: BASELINE_POOL.json, 256 MATH problems (96 level 1, 96 level 2, 64 level 3).
Generation: n=8, max_tokens=8192, evaluation sampling/template.
Activations: residual states 0–18 and all MLP down-projection inputs.

## Baseline diagnosis and fixed DEV
Baseline: 361/2048 correct (17.62695%). Summary categories: 827 wrong, 726 no-answer-finished, 134 failing cutoffs. pred=None is not proof of missing answer.
DEV fixed in DEV.json: 64 problems, each solved 1–7/8 times; selected with the predeclared seeded stratified sampler. No edits yet.
Observed errors: loss of constraints, invalid algebra, repetitive unsupported conclusions.

## Pass 2 planned: original teacher-forced localization
Exact edit: none; strength 0.
Set: all 64 fixed DEV problems, their 512 original samples and 64 reference solutions.
Texts saved in tf_dev_texts.json; complete request in pass2_request.json.
Own answers capped at first 1024 answer tokens; references at 1536; total text <=2048.
Return token logprobs, hidden states 14/15, MLP inputs at layers 8/14/17.
Candidate hypothesis: specific MLP channels reinforce repetitive, invalid mathematical continuations.

## Pass 2 INVALID EXPERIMENT (budget consumed)
A string filename supplied as texts was interpreted as 23 single-character texts.
No mathematical scoring data; exclude all pass-2 results from analysis.
No edits. Eight passes remain. Subsequent requests must include literal raw-text lists.

## Pass 3 planned: literal teacher-forced localization
Exact edit: none; strength 0. Set: fixed DEV only.
72 raw local-continuation probes: all 64 full reference solutions plus eight 280-character excerpts from repetitive failures, selected by high L14 neuron1594 activity.
No problem prompts: these measure local continuation effects, NOT answer accuracy or evaluation-format answer likelihood. literal_probes.json records exact texts.
Return token logprobs and MLP activations at layers 8,14,17.

## Pass 3 findings
Neuron 14:1594 not selective in token-level probes; do NOT edit it.
Convergent candidates: 8:699, 14:483, 14:984, 17:1179. Each has higher mean absolute loop activity than reference activity and negative length-adjusted correctness association in both DEV halves.

## Pass 4 planned: causal teacher-forced screen
Exact edit: model.layers.8.mlp.down_proj.weight column699; model.layers.14.mlp.down_proj.weight columns483,984; model.layers.17.mlp.down_proj.weight column1179. Strength 1.0: set these columns to zero.
All other values unchanged. Candidate snapshot: candidate_ablation_1/.
Set: same 72 literal DEV continuation probes as pass3. Compare matched tokens only.
Prediction: reduce loop-continuation logprob without broad reference deterioration.

## Pass 4 findings
{"reference_delta_mean": 0.002871671661540099, "reference_delta_se": 0.0027517547497991307, "loop_delta_mean": -0.00037985147163926655, "loop_late_delta": [-0.0002404921827177981, -0.011225744340292833, -0.013048870593134245, -0.04457081521671767, -0.0001309628937366035, 0.0021490811441058164, -0.049525074458394484, 0.011617401369236953], "interpretation": "Small, mixed local effects; no general loop-suppression claim."}
Matching unpadded prefixes only; pass3 and pass4 +1 excerpts have different lengths.

## Pass 5 planned: full paired DEV generation
Exact edit: down_proj columns 8:[699],14:[483,984],17:[1179] zeroed (strength1.0).
Set: all 64 DEV.json problems. model=both, n=16 (1024 answers/model), max_tokens8192.
Measure accuracy first, then repeated conclusions, repeated lines and cutoffs.

## Pass 5 results
{"pass": 5, "n": 1024, "gain_pp": -0.68359375, "se_pp": 1.578311217628527, "wrong_to_right": 127, "right_to_wrong": 134, "original": {"accuracy": 0.404296875, "mean_tokens": 1305.57421875, "cut_rate": 0.076171875, "repeat_fraction": 0.14803693883151464, "three_or_more_final_answers": 0.1357421875}, "edited": {"accuracy": 0.3974609375, "mean_tokens": 1803.5283203125, "cut_rate": 0.1318359375, "repeat_fraction": 0.20659392353500058, "three_or_more_final_answers": 0.2255859375}}
Ablation increased response length/cutoffs; proposed anti-loop direction was wrong.
Revised hypothesis: selected channels help restrain repetitive conclusions; their correlation with failures may reflect compensation.

## Pass 6 planned: opposite-direction causal test
Exact edit: down_proj columns 8:[699],14:[483,984],17:[1179] multiplied by2 (attenuation strength -1.0). All other values unchanged.
Set: same fixed DEV64, both models, n16, max_tokens8192 (1024/model).
Prediction: shorter loops/fewer cutoffs than contemporaneous original; accuracy remains primary endpoint.

## Pass 6 results
{"pass": 6, "n": 1024, "gain_pp": -1.26953125, "se_pp": 1.5350253102900164, "wrong_to_right": 117, "right_to_wrong": 130, "original": {"accuracy": 0.419921875, "mean_tokens": 1127.3798828125, "cut_rate": 0.0537109375, "repeat_fraction": 0.13534705331679775, "three_or_more_final_answers": 0.1337890625}, "edited": {"accuracy": 0.4072265625, "mean_tokens": 1042.6298828125, "cut_rate": 0.048828125, "repeat_fraction": 0.1269435861051612, "three_or_more_final_answers": 0.056640625}}
2x enhancement reduced mean length/cutoffs but did not improve accuracy.

## Pass 7 planned: final DEV strength test
Exact edit: down_proj columns 8:[699],14:[483,984],17:[1179] multiplied by1.5 (attenuation strength -0.5). All other values unchanged.
Set: fixed DEV64, both models, n16, max_tokens8192.
After pass7 choose among tested edits by paired DEV accuracy difference; freeze before any HOLDOUT output is seen.
Passes8–10 reserved for frozen-candidate HOLDOUT: same fixed 256 problems, n4 each, max8192. Predeclared input orders in holdout_confirmation_orders.json avoid identical-request rejection. Pool all confirmation results; no retuning.

## Pass 7 results and FINAL CANDIDATE FREEZE
{"pass": 7, "n": 1024, "gain_pp": 0.48828125, "se_pp": 1.6023915125683925, "wrong_to_right": 137, "right_to_wrong": 132, "original": {"accuracy": 0.4033203125, "mean_tokens": 1270.044921875, "cut_rate": 0.0703125, "repeat_fraction": 0.14516428219988894, "three_or_more_final_answers": 0.1318359375}, "edited": {"accuracy": 0.408203125, "mean_tokens": 1121.7421875, "cut_rate": 0.056640625, "repeat_fraction": 0.13290928597036492, "three_or_more_final_answers": 0.0791015625}}
Selected pass7 (1.5x) by highest paired DEV accuracy difference among passes5–7.
DEV gain +0.48828125 pp, reported problem-cluster SE1.72 pp; not significant.
Final exact edit: model.layers.8.mlp.down_proj.weight column699, model.layers.14.mlp.down_proj.weight columns483,984, model.layers.17.mlp.down_proj.weight column1179 multiplied by1.5 from ORIGINAL values.
All other entries identical. Frozen hash599aba54ec, copies in best_2/ and frozen_candidate/.
DO NOT TUNE OR SELECT USING HOLDOUT. Submit this frozen candidate regardless of confirmation sign.

## Pass 8 planned
Exact edit: frozen1.5x four-column enhancement above; no further changes.
Set: HOLDOUT.json, all256 reserved MATH problems, first predeclared order.
both, n4, max_tokens8192, 1024 answers/model. Pool with planned passes9 and10.

## Failure-summary correction from experimenters
Earlier no_answer_finished counts also included wrong stated final answers. Do not interpret them as missing-answer rates. Accuracy grades and our repetition metrics are unchanged; no selection relied on the faulty category.

## Pass 8 confirmation result
{"pass": 8, "n": 1024, "gain_pp": 1.171875, "se_pp": 0.7559242780588042, "wrong_to_right": 36, "right_to_wrong": 24, "original": {"accuracy": 0.0703125, "mean_tokens": 1614.7421875, "cut_rate": 0.0791015625, "repeat_fraction": 0.19316927858944166, "three_or_more_final_answers": 0.16015625}, "edited": {"accuracy": 0.08203125, "mean_tokens": 1448.388671875, "cut_rate": 0.0625, "repeat_fraction": 0.1726278177526928, "three_or_more_final_answers": 0.1103515625}, "cluster_se_pp": 0.7415623980049177}
No tuning: weights remain frozen1.5x, hash599aba54ec.

## Pass 9 planned
Exact edit unchanged: down_proj columns 8:[699],14:[483,984],17:[1179] x1.5.
Set: same256 HOLDOUT, second predeclared order (reverse); both, n4, max8192.
Pool with passes8 and10; no candidate selection from HOLDOUT.

## Pass 9 confirmation result
{"pass": 9, "n": 1024, "gain_pp": 0.87890625, "se_pp": 0.6548418423694392, "wrong_to_right": 27, "right_to_wrong": 18, "original": {"accuracy": 0.0712890625, "mean_tokens": 1453.2919921875, "cut_rate": 0.0634765625, "repeat_fraction": 0.176115624007531, "three_or_more_final_answers": 0.130859375}, "edited": {"accuracy": 0.080078125, "mean_tokens": 1402.5673828125, "cut_rate": 0.060546875, "repeat_fraction": 0.16674772691864398, "three_or_more_final_answers": 0.1142578125}, "cluster_se_pp": 0.6392642601681078}
Interim pooled confirmation: {"passes": [8, 9], "n_per_model": 2048, "distinct_problems": 256, "original_correct": 145, "edited_correct": 166, "original_accuracy_pct": 7.080078125, "edited_accuracy_pct": 8.10546875, "gain_pp": 1.025390625, "cluster_se_pp": 0.5856102921141735, "sample_pair_se_pp": 0.49994815880060045, "wrong_to_right": 63, "right_to_wrong": 42, "original_mean_tokens": 1534.01708984375, "original_cut_pct": 7.12890625, "original_repeated_conclusion_pct": 14.55078125, "edited_mean_tokens": 1425.47802734375, "edited_cut_pct": 6.15234375, "edited_repeated_conclusion_pct": 11.23046875}
No changes to candidate; checksum verification passed.

## Pass 10 planned: final confirmation
Exact edit unchanged: down_proj columns 8:[699],14:[483,984],17:[1179] x1.5, hash599aba54ec.
Set: same256 HOLDOUT, third predeclared order (rotation85); both, n4, max8192.
Final report pools all three HOLDOUT passes (3072 answers/model) and uses problem-cluster standard error over the256 problems.

## FINAL SUBMISSION
All10 forward passes complete. Final frozen candidate retained unchanged in /work/edited.
Exact edit: layer8 down_proj column699, layer14 down_proj columns483/984, layer17 down_proj column1179 each multiplied by1.5; every other entry unchanged.
No embedding/output/final-norm edits, no training, no whole-tensor rescaling.
Path: baseline repetitive-failure diagnosis -> within-problem/length-adjusted channel screen -> teacher-forced reference/loop probes -> ablation refuted harmful-channel hypothesis -> amplification supported repetition-restraint role -> 1.5x selected by DEV accuracy -> frozen three-pass HOLDOUT confirmation.
Pooled confirmation: {"passes": [8, 9, 10], "n_per_model": 3072, "distinct_problems": 256, "original_correct": 221, "edited_correct": 243, "original_accuracy_pct": 7.194010416666667, "edited_accuracy_pct": 7.91015625, "gain_pp": 0.7161458333333333, "cluster_se_pp": 0.4926066250697082, "sample_pair_se_pp": 0.4116191268561949, "wrong_to_right": 91, "right_to_wrong": 69, "original_mean_tokens": 1546.353515625, "original_cut_pct": 7.389322916666667, "original_repeated_conclusion_pct": 15.33203125, "edited_mean_tokens": 1435.96484375, "edited_cut_pct": 6.25, "edited_repeated_conclusion_pct": 12.01171875, "approx_95pct_ci_pp": [-0.24936315180329482, 1.6816548184699613], "candidate_frozen_before_holdout": true, "selection_used_holdout": false, "edit_hash": "599aba54ec"}
Group-level mechanism evidence does not prove each neuron individually causal.
Complete audit and limitations: FINAL_REPORT.json. Final checksums verified.
