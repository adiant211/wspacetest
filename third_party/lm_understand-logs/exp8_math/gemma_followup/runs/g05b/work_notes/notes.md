# Math weight-edit experiment

## Fixed design
- Seed: 20261025.
- HOLDOUT: 128 MATH problems, stratified by level; ids in HOLDOUT_ids.json.
  No holdout generations or scores will be inspected until the candidate is frozen.
- Diagnostic pool: 112 separate MATH problems plus 16 GSM8K problems;
  ids in diagnostic_ids.json.
- DEV will be fixed from diagnostic problems with both correct and incorrect
  baseline samples, before any weight edit.
- Planned passes: baseline generation; original teacher-forced localization;
  paired DEV at two edit strengths; frozen paired HOLDOUT.
- All generation comparisons use the specified sampling and 8192-token limit.

## Pass 1: baseline diagnosis
- Exact edit: none; original weights.
- Problems: diagnostic_ids.json, 128 problems, 16 samples each (2048 answers).
- Collect residual layers 0..18 and all available MLP down-projection inputs.
- Inspect accuracy, missing answers, token-limit failures, and failing text.

## DEV fixed before edits
- DEV_ids.json: 32 mixed-success MATH problems, selected by highest baseline Bernoulli variance, ties by id.
- HOLDOUT remains untouched. No weights edited.

## Baseline localization findings
- 337/2048 correct overall. DEV: 217/512 correct.
- Wrong quantity binding, invalid arithmetic/algebra, and repeated non-progressing equalities occur in inspected failures.
- No DEV correct-box-to-wrong-box overwrite cases found.
- Pure repetition correlations are not strong accuracy evidence.
- Length-controlled, within-problem failure correlations are modest.
- Early MLP neurons 1405,149,1309,2032 and others in layer 0 have stable associations across disjoint DEV halves; not yet causal.

## Pass 2: original teacher-forced localization
- Exact edit: none; original weights.
- Dataset: all 512 DEV sampled responses plus 32 DEV reference solutions, truncated only for the forward token budget.
- Full request: dev_teacher_request.json; text list: dev_teacher_texts.json; alignment: dev_teacher_manifest.json.
- Texts: 544; tokens: 427720; cap: 2048.
- Examine token-local arithmetic progress versus copied/repeated expressions and component specificity before selecting an edit.

## Pass 2 input-format failure (consumed)
- texts_file was rejected without cost. Passing a path in texts was incorrectly interpreted as 28 one-character texts, consuming pass 2.
- Intended 544-text localization batch was NOT run.
- No token-local math claims will be made from fwd_2 outputs.
- Remaining plan: two full paired DEV tests at different strengths, then freeze and test HOLDOUT.

## Candidate fixed before causal tests
- Selection and complete neuron evidence: candidate_manifest.json.
- Sparse down_proj column suppression only; no embeddings, output matrix, norms, attention, gate_proj or up_proj edited.
- Exact selected columns by layer: {"6": [640, 678, 1157], "7": [637, 1621], "8": [477, 792, 1777, 1865], "9": [884, 1493], "10": [1621, 1798], "11": [1, 587, 672, 1551, 1890], "12": [1123, 1184, 1278, 1797, 1894], "13": [39, 759, 1059, 1260, 1540, 1881], "14": [11, 23, 238, 640, 1157, 1202, 1497, 1935], "15": [661, 968, 1304, 1507, 1818], "16": [128, 255, 322, 513, 663, 1315, 1486, 1548, 1598, 1649, 1876, 2004], "17": [124, 942, 1115, 1222, 1326, 1475, 1484, 1554, 1688, 1741]}.
- Hypothesis: these outputs reinforce invalid numerical transformations. Association and output geometry alone are not causal proof.

## Pass 3: paired DEV, half suppression
- Exact edit: for each selected layer/column above, down_proj[:,j] = 0.5 * original[:,j]; all other values unchanged.
- Problems: DEV_ids.json; 32 problems x 32 samples/model = 1024/model.
- Both models, temperature/top-k/top-p fixed by evaluator; max_tokens=8192.
- Accuracy is primary; arithmetic errors and coverage are secondary.
- Next planned dose: full removal of exactly the same columns.

## Pass 3 results
- Original: 404/1024 (39.4531%); half suppression: 399/1024 (38.9648%).
- Paired difference: -0.49 percentage points; tool-reported SE 1.57 points.
- Wrong-to-right 143; right-to-wrong 148. No accuracy benefit demonstrated.
- Cut share fell from 7.91% to 5.66%; not treated as success.
- Arithmetic diagnostics: dev_half_mechanisms.json.

## Pass 4: paired DEV, full suppression
- Exact edit: same 64 columns in candidate_manifest.json, down_proj[:,j] = 0; all other tensor values unchanged.
- Same 32 DEV problems, 32 samples/model = 1024/model, max_tokens=8192, both original and edited.
- Half and full candidates are backed up separately. Select by within-pass accuracy difference, not response length.
- HOLDOUT still untouched; freeze the better tested non-original candidate before pass 5.

## Pass 4 results and candidate selection
- Original: 405/1024 (39.5508%); full suppression: 358/1024 (34.9609%).
- Paired difference: -4.59 points; tool-reported SE 1.70 points.
- Wrong-to-right 155; right-to-wrong 202. Full removal is harmful on DEV.
- Strong suppression hypothesis is not supported. Reduced truncation does not imply improved mathematical accuracy.
- Best tested non-original candidate: half suppression, selected on DEV only. It has no demonstrated DEV accuracy benefit.

## Candidate frozen before HOLDOUT
- Exact edit: down_proj[:,j] = 0.5 * original[:,j] for the same 64 columns in candidate_manifest.json; all other values unchanged.
- Verified bitwise unchanged unselected columns; tensor hashes saved in frozen_candidate.json. Backup: /work/best_1/.
- No further candidate selection or tuning using HOLDOUT.

## Pass 5: frozen HOLDOUT confirmation
- Problems: HOLDOUT_ids.json, fixed before baseline; 128 MATH problems.
- Both models, 8 samples/problem = 1024 answers/model, max_tokens=8192.
- Report the paired accuracy difference and its uncertainty honestly, including a negative or inconclusive result.

## Final HOLDOUT result and submission
- Original: 90/1024 (8.7891%); edited: 102/1024 (9.9609%).
- Paired difference: +1.171875 percentage points; reported SE: 0.90 points.
- Approximate 95% interval: [-0.59, 2.94] points.
- Wrong-to-right 46; right-to-wrong 34. Positive but inconclusive.
- Submitted candidate is unchanged from pre-HOLDOUT freeze: 64 selected MLP down_proj columns in layers 6-17 multiplied by 0.5.
- All saved tensor hashes verified against frozen_candidate.json.
- No HOLDOUT tuning or weight changes after confirmation.
- Experimenter correction: earlier no_answer_finished counts included some wrong final answers; those category counts cannot establish an answer-completion failure mechanism. Accuracy comparisons unaffected.
- Full-removal harm and weak half-strength results limit the causal interpretation; no claim of a proven arithmetic-error circuit.
- Complete outcome and limitations: final_report.json.
