# Math weight-edit experiment
No edits. Final norm and embedding/output weights will never be edited.
HOLDOUT fixed before all experiments: /work/HOLDOUT.json, 256 random MATH problems, seed 290173. Do not inspect outcomes until candidate is frozen.
Diagnostic pool: /work/DIAGNOSTIC.json, 128 disjoint MATH problems: 80 level 1, 40 level 2, 8 level 3.
DEV will be fixed from partly solved diagnostic problems before any weight edit.

## Pass 1 planned
Original model; no changed tensors or strengths. DIAGNOSTIC, n=16, max_tokens=8192.
Collect residual and MLP activations at all layers to compare success/failure within the same problem.

Pass 1: 2048 answers; accuracy 20.4102%; cut share 6.84%; reported wrong/no-answer/cut failures 795/702/133.
Text inspection: arithmetic/procedure errors and repeated conclusions; pred=None can still have a textual answer or even a correct grade, so do not equate it to actual absence of an answer.
DEV fixed before edits: 63 randomly selected mixed-success problems, /work/DEV.json, seed 87102; HOLDOUT remains unused.

Associations: within-DEV-problem, length/cut-controlled MLP failure correlations are modest (~0.10-0.14); no causal claim yet. Grade alignment exact; activation token counts include prompt tokens, capped at 8192.
## Pass 2 planned
Original, no edits. Teacher-force fixed DEV corpus /work/tf_dev_texts.json (all references and up to 4 correct/4 failing baseline samples per problem, truncated at 1200 total tokens). Score continuation likelihoods and inspect MLP layers 4,13,15,17; no HOLDOUT use.

Pass 2 wasted: string path in `texts` was iterated as 23 characters; no useful evidence. Only literal text arrays are supported. 23 passes remain.
## Pass 3 planned
Original, no changed tensors. All 63 DEV exact prompts followed by boxed reference answers (direct-answer likelihood screen, not a reasoning-accuracy measurement). Files tf_compact_texts.json and tf_compact_info.json. Return logprobs and MLP activations at layers 4,13,15,17.

## Candidate A / pass 4 planned
DEV only. Zero 32 selected output columns in each MLP down_proj at layers 4,15,17 (96 neurons total); strengths=1.0 removal. Selection: positive raw failure association, ranked by smaller of split-half length/cut-controlled correlations. Exact edit:
{"model.layers.4.mlp.down_proj.weight": {"columns": [148, 887, 381, 1783, 1048, 180, 1809, 2030, 36, 1483, 1700, 1433, 1546, 903, 1159, 1181, 1045, 930, 1135, 1539, 1097, 403, 335, 1696, 724, 695, 1827, 12, 274, 1326, 830, 1175], "removed_fraction": 1.0}, "model.layers.15.mlp.down_proj.weight": {"columns": [1638, 506, 1170, 936, 18, 565, 16, 75, 269, 1627, 599, 517, 1031, 1036, 187, 1482, 1805, 1747, 455, 837, 1055, 1152, 162, 1828, 360, 1777, 685, 469, 33, 228, 870, 1065], "removed_fraction": 1.0}, "model.layers.17.mlp.down_proj.weight": {"columns": [1201, 2044, 330, 25, 1427, 1538, 1804, 1180, 1054, 57, 103, 1870, 43, 349, 304, 1975, 1392, 508, 1502, 2003, 1051, 1137, 1598, 182, 633, 1345, 335, 234, 1860, 1518, 788, 1918], "removed_fraction": 1.0}}
Teacher-forced pass 4 repeats all 63 compact DEV texts from pass 3. Compare answer-content logprobs before full-size paired generation.

Pass 4 A full-ablation screen: answer-content mean logprob change -0.382291 nats/token, SE 0.094522; 19/63 improved. Verified encoding with add_special_tokens=False.
## Pass 5 planned
Candidate A unchanged: exact columns in failure_neurons_96.json, layers4/15/17 down_proj, removal=1.0. Both models, DEV63, n16 (1008/model), max_tokens8192. Primary endpoint accuracy; secondary repetition/cut rates.

Pass 5 full ablation: {"pass": 5, "n": 1008, "gain": -0.08630952380952381, "sample_se": 0.01757387717436244, "original": {"accuracy": 0.35813492063492064, "repeated": 0.1498015873015873, "many_finals": 0.10714285714285714, "cut": 0.04365079365079365}, "edited": {"accuracy": 0.2718253968253968, "repeated": 0.16765873015873015, "many_finals": 0.12797619047619047, "cut": 0.04265873015873016}}
Causal conclusion: simple harmful-neuron hypothesis refuted; ablation reduces accuracy substantially, with nearly unchanged cutoffs. Association may reflect useful compensatory computation.
## Pass 6 planned
Candidate A half attenuation: same 96 columns, layers4/15/17 down_proj, each multiplied by0.5 (removal0.5). DEV63, both,n16,max_tokens8192. Exact edit:
{"model.layers.4.mlp.down_proj.weight": {"columns": [148, 887, 381, 1783, 1048, 180, 1809, 2030, 36, 1483, 1700, 1433, 1546, 903, 1159, 1181, 1045, 930, 1135, 1539, 1097, 403, 335, 1696, 724, 695, 1827, 12, 274, 1326, 830, 1175], "removed_fraction": 0.5}, "model.layers.15.mlp.down_proj.weight": {"columns": [1638, 506, 1170, 936, 18, 565, 16, 75, 269, 1627, 599, 517, 1031, 1036, 187, 1482, 1805, 1747, 455, 837, 1055, 1152, 162, 1828, 360, 1777, 685, 469, 33, 228, 870, 1065], "removed_fraction": 0.5}, "model.layers.17.mlp.down_proj.weight": {"columns": [1201, 2044, 330, 25, 1427, 1538, 1804, 1180, 1054, 57, 103, 1870, 43, 349, 304, 1975, 1392, 508, 1502, 2003, 1051, 1137, 1598, 182, 633, 1345, 335, 234, 1860, 1518, 788, 1918], "removed_fraction": 0.5}}

Pass6 half attenuation: reported gain -1.88pp, SE1.94pp. {"pass": 6, "n": 1008, "gain": -0.018849206349206348, "sample_se": 0.017037357096128147, "original": {"accuracy": 0.35912698412698413, "repeated": 0.15178571428571427, "many_finals": 0.1111111111111111, "cut": 0.034722222222222224}, "edited": {"accuracy": 0.3402777777777778, "repeated": 0.16964285714285715, "many_finals": 0.12797619047619047, "cut": 0.040674603174603176}}
## Pass7 planned
Reverse-direction causal test: same selected96 columns in layers4/15/17 down_proj multiplied by1.5 (removal=-0.5). DEV63,both,n16,max_tokens8192. Hypothesis: components are useful but insufficiently recruited, not harmful.
{"model.layers.4.mlp.down_proj.weight": {"columns": [148, 887, 381, 1783, 1048, 180, 1809, 2030, 36, 1483, 1700, 1433, 1546, 903, 1159, 1181, 1045, 930, 1135, 1539, 1097, 403, 335, 1696, 724, 695, 1827, 12, 274, 1326, 830, 1175], "removed_fraction": -0.5}, "model.layers.15.mlp.down_proj.weight": {"columns": [1638, 506, 1170, 936, 18, 565, 16, 75, 269, 1627, 599, 517, 1031, 1036, 187, 1482, 1805, 1747, 455, 837, 1055, 1152, 162, 1828, 360, 1777, 685, 469, 33, 228, 870, 1065], "removed_fraction": -0.5}, "model.layers.17.mlp.down_proj.weight": {"columns": [1201, 2044, 330, 25, 1427, 1538, 1804, 1180, 1054, 57, 103, 1870, 43, 349, 304, 1975, 1392, 508, 1502, 2003, 1051, 1137, 1598, 182, 633, 1345, 335, 234, 1860, 1518, 788, 1918], "removed_fraction": -0.5}}

Pass7 amplification: reported gain -1.59pp, SE1.63pp. {"pass": 7, "n": 1008, "gain": -0.015873015873015872, "sample_se": 0.016180367922676898, "original": {"accuracy": 0.3343253968253968, "repeated": 0.17559523809523808, "many_finals": 0.11607142857142858, "cut": 0.054563492063492064}, "edited": {"accuracy": 0.31845238095238093, "repeated": 0.19146825396825398, "many_finals": 0.11706349206349206, "cut": 0.041666666666666664}}
Candidate A rejected as a useful repair: full attenuation harmed, half attenuation and amplification did not improve accuracy. Need a more specific component hypothesis.

Next diagnostic hypothesis: erroneous reuse of intermediate values / repeated conclusions, inspect full-attention head routing on matched algebra380 continuations. Candidate A amplification remains saved but is not accepted as a repair. No HOLDOUT outcomes inspected.

## Pass8 planned
Original model (no edits applied for this pass; disk still contains candidateA 1.5x in three down_proj tensors). Two controlled algebra380 probes in /work/attention_probe_texts.json: correct square-identity computation versus repeated intermediate-value conclusion. Full-attention layers5,11,17; inspect head routing, no causal claim from attention alone. Attention tool limits probes to2 texts.

Attention probe: L11H0 recent8 mass at final-number query is0.420 correct versus0.515 repetitive wrong; strongest recent emphasis among full-attention heads. Potential syntax confound acknowledged; no causal conclusion yet.
## Pass9 planned
Candidate B: only model.layers.11.self_attn.o_proj.weight columns0:256 zeroed (head0 removed, strength1). All MLP edits removed. Teacher-force the same2 attention probes as pass8; return layer17 attention (2-text tool limit), compare correct-computation versus repetitive-continuation likelihood.
{"model.layers.11.self_attn.o_proj.weight": {"head": 0, "column_start": 0, "column_stop": 256, "removed_fraction": 1.0}}

Experimenter correction: previous no_answer_finished counts included wrong stated answers; disregard them as estimates of truly absent final answers. Grades/accuracies unchanged.
Pass9 head-ablation probe likelihood changes: [{"probe": "correct", "continuation_change": 0.032573238015174866, "after_shared_prefix_change": 0.016032347455620766, "final_answer_change": 0.8112861514091492}, {"probe": "repeated_wrong", "continuation_change": 0.031652241945266724, "after_shared_prefix_change": 0.012711974792182446, "final_answer_change": 0.001643982483074069}]
## Pass10 planned
CandidateB unchanged: only L11 self_attn.o_proj columns0:256 zeroed, head0 removal1.0. Both models,DEV63,n16,max_tokens8192. Test accuracy primarily and repetitive-line rate secondarily.

Pass10 L11H0 ablation: reported gain-15.97pp,SE2.75pp; cutoffs4.17%->28.57%. {"pass": 10, "n": 1008, "gain": -0.1597222222222222, "sample_se": 0.018228600773082713, "original": {"accuracy": 0.35714285714285715, "repeated": 0.1378968253968254, "many_finals": 0.10515873015873016, "cut": 0.041666666666666664}, "edited": {"accuracy": 0.1974206349206349, "repeated": 0.44543650793650796, "many_finals": 0.30257936507936506, "cut": 0.2857142857142857}}
Causal revision: head is protective against runaway continuation, not a harmful copying head. One controlled TF probe did not predict broad behavior.
## Pass11 planned
Only L11 self_attn.o_proj columns0:256 multiplied by1.5 (head0 removal=-0.5); all other weights original. DEV63,both,n16,max_tokens8192. Predict reduced degeneracy AND improved accuracy; accuracy is primary.
{"model.layers.11.self_attn.o_proj.weight": {"head": 0, "column_start": 0, "column_stop": 256, "removed_fraction": -0.5}}

Pass11 L11H0x1.5: reported gain-4.27pp,SE1.96pp; cutoffs4.17%->2.08%. {"pass": 11, "n": 1008, "gain": -0.04265873015873016, "sample_se": 0.017052494739661368, "original": {"accuracy": 0.36607142857142855, "repeated": 0.14682539682539683, "many_finals": 0.10813492063492064, "cut": 0.041666666666666664}, "edited": {"accuracy": 0.32341269841269843, "repeated": 0.12698412698412698, "many_finals": 0.06845238095238096, "cut": 0.020833333333333332}}
Conclusion: head0 causally controls repetitive continuation but boosting it does not repair math accuracy. Reject fewer cutoffs as success.
## Pass12 planned
Candidate C: restore head0; only L11 self_attn.o_proj columns512:768 (head2) multiplied by1.5. Head2 is strongest numeric-intermediate retriever in probes (previous90 mass .046 correct/.140 wrong, versus head0 .001/.005); occurrence-count confound acknowledged. Test whether stronger content retrieval improves computation-to-answer use. DEV63,both,n16,max_tokens8192.
{"model.layers.11.self_attn.o_proj.weight": {"head": 2, "column_start": 512, "column_stop": 768, "removed_fraction": -0.5}}

Pass12 L11H2x1.5: reported gain-6.35pp,SE2.20pp; cutoffs4.07%->19.84%. {"pass": 12, "n": 1008, "gain": -0.06349206349206349, "sample_se": 0.017530249678457514, "original": {"accuracy": 0.3621031746031746, "repeated": 0.1488095238095238, "many_finals": 0.10615079365079365, "cut": 0.040674603174603176}, "edited": {"accuracy": 0.2986111111111111, "repeated": 0.33630952380952384, "many_finals": 0.2648809523809524, "cut": 0.1984126984126984}}
Causal finding: amplifying head2 promotes runaway continuation, opposite head0. Revise hypothesis toward excessive retrieval/reinforcement of repeated intermediates.
## Pass13 planned
Only L11 self_attn.o_proj columns512:768 multiplied by0.5 (head2 removal0.5); all other weights original. DEV63,both,n16,max_tokens8192. Predict reduced repetition and improved accuracy; reject if only shortening improves.
{"model.layers.11.self_attn.o_proj.weight": {"head": 2, "column_start": 512, "column_stop": 768, "removed_fraction": 0.5}}

Pass13 L11H2x0.5: reported gain-18.25pp,SE3.10pp. {"pass": 13, "n": 1008, "gain": -0.18253968253968253, "sample_se": 0.018420209970324872, "original": {"accuracy": 0.3551587301587302, "repeated": 0.13988095238095238, "many_finals": 0.0992063492063492, "cut": 0.040674603174603176}, "edited": {"accuracy": 0.17261904761904762, "repeated": 0.08531746031746032, "many_finals": 0.05257936507936508, "cut": 0.030753968253968252}}
Broad head2 attenuation rejected: useful content retrieval is essential. Next hypothesis targets only a rank-one repetition-related output mode.
Direction construction: demean H_after11-H_before11 within DEV problem; remove loglength,cut,correctness effects; covariance with similarly residualized repeated-line indicator; project resulting residual direction onto column space of D*O_head2, D=diag(1+post_attention_layernorm). Split-half cosine=0.625791. No optimization/backprop or network simulation.

## Pass14 planned
Candidate D: rank-one repetition-mode removal in L11H2 output block only, strength1.0. Exact delta saved /work/l11h2_repetition_rank1_delta.npy; W[:,512:768] <- W_original[:,512:768] - delta, all other entries/tensors original. Relative whole-tensor Frobenius change11.34%, relative head change19.74%; not whole-tensor scaling. DEV63,both,n16,max_tokens8192.
{"model.layers.11.self_attn.o_proj.weight": {"head": 2, "columns": [512, 768], "operation": "subtract strength * fixed repetition rank-one delta", "delta_file": "/work/l11h2_repetition_rank1_delta.npy", "strength": 1.0}}

Pass14 rank-one removal1.0: DEV accuracy36.31%->31.35%, gain-4.96pp, reported SE2.15pp; cutoffs3.77%->5.95%. Harmful-repetition interpretation contradicted.

## Pass15 planned
Reverse causal test: only L11H2 rank-one direction amplified50%, removal strength=-0.5; all other weights original. DEV63,both,n16,max_tokens8192. Accuracy primary.
{"model.layers.11.self_attn.o_proj.weight": {"head": 2, "columns": [512, 768], "operation": "subtract strength * fixed repetition rank-one delta", "delta_file": "/work/l11h2_repetition_rank1_delta.npy", "strength": -0.5}}

Pass15 rank-one amplification50%: DEV35.52%->33.33%, gain-2.18pp, reported SE1.70pp; cutoffs3.87%->6.75%. Reject candidateD in both tested directions. No HOLDOUT inspected.

Repetition-specific MLP screen: within-problem associations controlling loglength,cut,correctness. L0 neuron1302 correlation0.334; split halves0.318/0.352. Nonrepeat means correct3.50/wrong3.43, repeat correct4.28/wrong4.51; prompt mean28.28 suggests possible useful syntax role. Observational only.

## Pass16 planned
CandidateE: restore all other weights; zero only L0 MLP down_proj column1302. DEV63,both,n16,max_tokens8192. Test accuracy and repetition, not brevity alone.
{"model.layers.0.mlp.down_proj.weight": {"columns": [1302], "removed_fraction": 1.0}}

Pass16 L0 neuron1302 removed: DEV36.01%->33.83%, gain-2.18pp, reported SE1.84pp; cutoffs4.46%->14.58%, missing finished answers1->24. Ablation contradicts harmful-neuron hypothesis; suggests stabilizing role.
[{"pass": 16, "n": 1008, "gain": -0.021825396825396824, "sample_se": 0.017292131992291862, "original": {"accuracy": 0.3601190476190476, "repeated": 0.1498015873015873, "many_finals": 0.10813492063492064, "cut": 0.044642857142857144}, "edited": {"accuracy": 0.3382936507936508, "repeated": 0.34523809523809523, "many_finals": 0.23115079365079366, "cut": 0.14583333333333334}}, [["math_train_number_theory_417", -0.0625], ["math_train_prealgebra_867", 0.0625], ["math_train_algebra_477", 0.0625], ["math_train_intermediate_algebra_358", 0.0625], ["math_train_precalculus_76", -0.0625], ["math_train_algebra_1491", 0.125], ["math_train_algebra_36", 0.1875], ["math_train_algebra_817", 0.0], ["math_train_algebra_1425", -0.125], ["math_train_precalculus_344", 0.125], ["math_train_prealgebra_161", 0.0], ["math_train_algebra_920", 0.0625], ["math_train_intermediate_algebra_1289", -0.125], ["math_train_prealgebra_783", -0.125], ["math_train_counting_and_probability_111", 0.0], ["math_train_number_theory_95", -0.25], ["math_train_algebra_29", -0.0625], ["math_train_algebra_580", -0.1875], ["math_train_intermediate_algebra_955", -0.125], ["math_train_intermediate_algebra_802", -0.0625], ["math_train_precalculus_125", -0.0625], ["math_train_number_theory_387", 0.1875], ["math_train_intermediate_algebra_901", 0.0], ["math_train_prealgebra_672", -0.3125], ["math_train_algebra_595", 0.0], ["math_train_algebra_486", -0.0625], ["math_train_prealgebra_668", -0.125], ["math_train_prealgebra_394", -0.125], ["math_train_number_theory_505", -0.0625], ["math_train_algebra_905", -0.125], ["math_train_geometry_310", 0.3125], ["math_train_counting_and_probability_586", 0.0625], ["math_train_intermediate_algebra_788", -0.0625], ["math_train_number_theory_745", 0.3125], ["math_train_algebra_697", -0.125], ["math_train_precalculus_404", 0.125], ["math_train_prealgebra_697", -0.0625], ["math_train_algebra_502", 0.0], ["math_train_prealgebra_937", 0.1875], ["math_train_intermediate_algebra_625", -0.25], ["math_train_intermediate_algebra_548", 0.0], ["math_train_algebra_1585", 0.0625], ["math_train_prealgebra_443", 0.1875], ["math_train_precalculus_233", -0.25], ["math_train_algebra_775", 0.0], ["math_train_prealgebra_34", -0.0625], ["math_train_intermediate_algebra_547", -0.0625], ["math_train_prealgebra_1022", -0.0625], ["math_train_number_theory_528", -0.125], ["math_train_geometry_161", -0.125], ["math_train_prealgebra_843", -0.1875], ["math_train_algebra_577", 0.0], ["math_train_prealgebra_408", 0.125], ["math_train_algebra_380", -0.3125], ["math_train_counting_and_probability_61", 0.0], ["math_train_precalculus_139", 0.125], ["math_train_prealgebra_758", -0.375], ["math_train_prealgebra_511", 0.1875], ["math_train_algebra_1374", 0.0], ["math_train_number_theory_530", 0.0], ["math_train_number_theory_19", -0.125], ["math_train_prealgebra_1138", 0.25], ["math_train_algebra_863", 0.0625]]]

## Pass17 planned
Only L0 MLP down_proj column1302 multiplied1.5; all other weights original. DEV63,both,n16,max_tokens8192. Reverse causal prediction: better termination and accuracy, accuracy required.
{"model.layers.0.mlp.down_proj.weight": {"columns": [1302], "removed_fraction": -0.5, "column_multiplier": 1.5}}

Pass17 L0 neuron1302x1.5: DEV36.31%->34.52%, gain-1.79pp, reported SE1.66pp; cutoffs3.77%->3.08%. Better termination did not improve math. CandidateE not accepted.
{"pass": 17, "n": 1008, "gain": -0.017857142857142856, "sample_se": 0.015494004174356823, "original": {"accuracy": 0.3630952380952381, "repeated": 0.13988095238095238, "many_finals": 0.09821428571428571, "cut": 0.037698412698412696}, "edited": {"accuracy": 0.34523809523809523, "repeated": 0.14285714285714285, "many_finals": 0.09325396825396826, "cut": 0.030753968253968252}}

New numerical-output hypothesis: L17 MLP neuron920 has stable within-problem correct-solution association, numerical/prose activity ratio1.548, answer activity0.3375 vs formatting0.1677. Static post-FFN-scaled output direction aligns with correct-digit contrast on69.84% of63 DEV problems. Neuron1062 rejected: alignment negative despite numerical selectivity. No network simulation or training.
{"layer": 17, "neuron": 920, "corr": -0.08687307641890037, "raw": -0.09243601601965246, "repeat": 0.04080985307178793, "half1": -0.06519743108745872, "half2": -0.10618089599962006, "activity": 0.3642396322923026, "output_norm": 0.30014362931251526, "correct_stable": 0.06519743108745872}

## Pass18 planned
CandidateF: restore all other tensors; double only L17 MLP down_proj column920. DEV63,both,n16,max_tokens8192. Test mathematical accuracy; no unconditional output bias or embedding/norm edits.
{"model.layers.17.mlp.down_proj.weight": {"columns": [920], "column_multiplier": 2.0, "removed_fraction": -1.0}}

Pass18 L17 neuron920x2: DEV36.01%->34.72%, gain-1.29pp, reported SE1.46pp; cutoffs4.27%->5.16%. No demonstrated benefit.
{"pass": 18, "n": 1008, "gain": -0.012896825396825396, "sample_se": 0.014749806981744693, "original": {"accuracy": 0.3601190476190476, "repeated": 0.14484126984126985, "many_finals": 0.10912698412698413, "cut": 0.04265873015873016}, "edited": {"accuracy": 0.3472222222222222, "repeated": 0.1636904761904762, "many_finals": 0.12797619047619047, "cut": 0.051587301587301584}}

## Pass19 planned
Only L17 MLP down_proj column920 zeroed. DEV63,both,n16,max_tokens8192. Opposite-strength causal test of the same numerically selective component.
{"model.layers.17.mlp.down_proj.weight": {"columns": [920], "column_multiplier": 0.0, "removed_fraction": 1.0}}

Pass19 L17 neuron920 removed: DEV36.01%->33.73%, gain-2.28pp, reported SE1.63pp; cutoffs unchanged4.37%. Neither tested extreme supports candidateF.
{"pass": 19, "n": 1008, "gain": -0.022817460317460316, "sample_se": 0.014262279552566452, "original": {"accuracy": 0.3601190476190476, "repeated": 0.14781746031746032, "many_finals": 0.10813492063492064, "cut": 0.04365079365079365}, "edited": {"accuracy": 0.3373015873015873, "repeated": 0.15575396825396826, "many_finals": 0.11011904761904762, "cut": 0.04365079365079365}}

Final DEV tuning plan: test protective L11H0 multipliers1.10 (pass20) and1.25 (pass21), then replicate DEV-selected best candidate (pass22). Freeze before HOLDOUT; passes23-25 are confirmation only, with unchanged weights.

## Pass20 planned
Only L11 self_attn.o_proj columns0:256 multiplied1.10. All other weights original. DEV63,both,n16,max_tokens8192. Earlier head removal increased loops and hurt accuracy;1.5x suppressed cutoffs but hurt accuracy. Test smaller protective-head reinforcement.
{"model.layers.11.self_attn.o_proj.weight": {"head": 0, "columns": [0, 256], "column_multiplier": 1.1, "removed_fraction": -0.1}}

Pass20 L11H0x1.10: DEV35.81%->32.74%, gain-3.08pp, reported SE1.33pp; cutoffs3.97%->4.07%. Small reinforcement also failed.
{"pass": 20, "n": 1008, "gain": -0.030753968253968252, "sample_se": 0.014178154634078961, "original": {"accuracy": 0.35813492063492064, "repeated": 0.14583333333333334, "many_finals": 0.10515873015873016, "cut": 0.03968253968253968}, "edited": {"accuracy": 0.3273809523809524, "repeated": 0.1488095238095238, "many_finals": 0.11904761904761904, "cut": 0.040674603174603176}}

## Pass21 planned
Only L11 self_attn.o_proj columns0:256 multiplied1.25. DEV63,both,n16,max_tokens8192. Complete predeclared intermediate-strength test; accuracy primary.
{"model.layers.11.self_attn.o_proj.weight": {"head": 0, "columns": [0, 256], "column_multiplier": 1.25, "removed_fraction": -0.25}}

Pass21 L11H0x1.25: DEV35.62%->33.33%, gain-2.28pp, reported SE1.70pp; cutoffs3.97%->3.67%. No head-reinforcement strength improved accuracy.
{"pass": 21, "n": 1008, "gain": -0.022817460317460316, "sample_se": 0.016141633208662177, "original": {"accuracy": 0.35615079365079366, "repeated": 0.14285714285714285, "many_finals": 0.10615079365079365, "cut": 0.03968253968253968}, "edited": {"accuracy": 0.3333333333333333, "repeated": 0.12996031746031747, "many_finals": 0.0873015873015873, "cut": 0.03670634920634921}}

## Pass22 planned
DEV-selected candidate: L17 MLP down_proj column920x2, all other weights original. Same63 DEV IDs in reverse order (/work/DEV_RETEST.json), both,n16,max_tokens8192. Independent batching-sensitive replication; pool passes18+22 for this candidate. HOLDOUT still unused. Candidate copied /work/best_1.
{"model.layers.17.mlp.down_proj.weight": {"columns": [920], "column_multiplier": 2.0, "removed_fraction": -1.0}}

Pass22 L17n920x2 replication: DEV35.52%->35.62%, gain+0.10pp, reported SE1.41pp.
Pooled DEV18+22: {"passes": [18, 22], "problems": 63, "answers_per_model": 2016, "original_accuracy": 0.3576388888888889, "edited_accuracy": 0.35168650793650796, "gain": -0.005952380952380952, "problem_cluster_se": 0.012823380815162802, "sample_paired_se": 0.010430214767814907, "samples_wrong_to_right": 215, "samples_right_to_wrong": 227}

CANDIDATE FROZEN BEFORE HOLDOUT. Exact edit L17 MLP down_proj column920x2; all other weights original. No proven DEV gain; best tested non-original candidate. SHA256 d38d76fbe02526e623c7b27655459f0cff667e447141a6add88374c831356649. Copy /work/frozen_submission. No further weight changes or HOLDOUT tuning.
Confirmation schedule fixed now: pass23 first128 HOLDOUT IDs n8; pass24 remaining128 n8; pass25 all256 n4. Both models,max_tokens8192 each; total3072 answers/model,12/problem. Pool all confirmation results with problem-cluster standard error.

## Pass23 planned
Frozen edit {"model.layers.17.mlp.down_proj.weight": {"columns": [920], "column_multiplier": 2.0, "removed_fraction": -1.0}}; /work/HOLDOUT_A.json (first128 original HOLDOUT IDs),both,n8,max_tokens8192.

Pass23 HOLDOUT_A: original8.30%,edited7.23%, gain-1.07pp, reported SE0.73pp; cutoffs8.01%->10.45%. Frozen weights unchanged. No tuning.
{"passes": [23], "problems": 128, "answers_per_model": 1024, "original_accuracy": 0.0830078125, "edited_accuracy": 0.072265625, "gain": -0.0107421875, "problem_cluster_se": 0.007340220994878845, "sample_paired_se": 0.006831026330713713, "samples_wrong_to_right": 19, "samples_right_to_wrong": 30}

## Pass24 planned
Frozen edit {"model.layers.17.mlp.down_proj.weight": {"columns": [920], "column_multiplier": 2.0, "removed_fraction": -1.0}}; /work/HOLDOUT_B.json (remaining128 original HOLDOUT IDs),both,n8,max_tokens8192. Same tensor SHA256 d38d76fbe02526e623c7b27655459f0cff667e447141a6add88374c831356649. Confirmation only.

Pass24 HOLDOUT_B: original8.59%,edited9.47%, gain+0.88pp, reported SE0.75pp; cutoffs6.35%->9.96%. Frozen weights unchanged. No tuning.
{"passes": [24], "problems": 128, "answers_per_model": 1024, "original_accuracy": 0.0859375, "edited_accuracy": 0.0947265625, "gain": 0.0087890625, "problem_cluster_se": 0.007490098880727775, "sample_paired_se": 0.007499751698748749, "samples_wrong_to_right": 34, "samples_right_to_wrong": 25}
Pooled HOLDOUT23+24: {"passes": [23, 24], "problems": 256, "answers_per_model": 2048, "original_accuracy": 0.08447265625, "edited_accuracy": 0.08349609375, "gain": -0.0009765625, "problem_cluster_se": 0.0052688966924415905, "sample_paired_se": 0.005075561018354749, "samples_wrong_to_right": 53, "samples_right_to_wrong": 55}

## Pass25 planned
Frozen edit {"model.layers.17.mlp.down_proj.weight": {"columns": [920], "column_multiplier": 2.0, "removed_fraction": -1.0}}; all256 IDs in original /work/HOLDOUT.json order (HOLDOUT_A then HOLDOUT_B),both,n4,max_tokens8192. Same SHA256 d38d76fbe02526e623c7b27655459f0cff667e447141a6add88374c831356649. Completes fixed3072 answers/model,12/problem. No weight changes allowed after observing HOLDOUT.

Pass25 HOLDOUT all256,n4: original7.42%,edited7.32%, gain-0.10pp, reported SE0.86pp. Frozen weights unchanged.

## FINAL SUBMISSION
Only model.layers.17.mlp.down_proj.weight[:,920] multiplied2. All other values original; embedding/output and final norm untouched. Selected on DEV, no HOLDOUT tuning. All25 forward passes used.
{"submitted_edit": {"model.layers.17.mlp.down_proj.weight": {"columns": [920], "column_multiplier": 2.0, "removed_fraction": -1.0}}, "tensor_sha256": "d38d76fbe02526e623c7b27655459f0cff667e447141a6add88374c831356649", "changed_entries": 640, "dev": {"passes": [18, 22], "problems": 63, "answers_per_model": 2016, "original_accuracy": 0.3576388888888889, "edited_accuracy": 0.35168650793650796, "gain": -0.005952380952380952, "problem_cluster_se": 0.012823380815162802, "sample_paired_se": 0.010430214767814907, "samples_wrong_to_right": 215, "samples_right_to_wrong": 227}, "holdout": {"passes": [23, 24, 25], "problems": 256, "answers_per_model": 3072, "original_accuracy": 0.0810546875, "edited_accuracy": 0.080078125, "gain": -0.0009765625, "problem_cluster_se": 0.005289048219691222, "sample_paired_se": 0.004232423080406631, "samples_wrong_to_right": 83, "samples_right_to_wrong": 86}, "holdout_95pct_ci": [-0.011343097010594796, 0.009389972010594796], "secondary_holdout": {"original": {"cut_share": 0.07096354166666667, "mean_tokens": 1581.9137369791667}, "edited": {"cut_share": 0.09765625, "mean_tokens": 1779.919921875}}, "conclusion": "No demonstrated accuracy improvement. Best tested non-original DEV candidate retained unchanged through HOLDOUT.", "selection_path": "Rejected broad failure-neuron, repetition-direction, and attention-head edits; selected numerically selective L17 MLP neuron920 using DEV associations and static digit alignment, tested removal and doubling, and replicated doubling before freezing."}
