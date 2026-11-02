# Math weight editing
Pass 1: original on 32 GSM8K + 64 MATH diagnostic problems. Accuracy 11/96 (11.46%); 85/96 cut at 32768; no finished wrong answers. Dominant target: pathological long/repetitive reasoning. No edits yet.
Pass 2 TF paired productive vs repetition examples showed a common repetition direction (cosine ~0.7 with generation contrast), but chat template adds an empty think block. Corrected in pass 3 (files fwd_4) using raw template. Testing layer19 MLP neuron11 output ablation; strongest matched-context repetition writer. Attention-return request failed and did not count.
Pass4 gen_5 paired32: ablate L19 neuron11, +6.25pp +/-4.35pp SE (3->5 correct), cutoff90.6->87.5%; weak positive, not confirmed. Trying rank-one removal of the matched repetition direction from L19 MLP only. Attention-output tool fails even for one raw text/layer.
Pass5 gen_7 paired32: L19 MLP repetition-direction removal alpha1, +6.25pp +/-4.35pp SE (4->6 correct), cut29->26, mean30123->27698. Directional and neuron probes modest. Next two-site L15+L19 alpha1.

Recovered passes 6-9: gen8/9 gave 7/32 vs 4/32; gen10/11 gave 9/32 vs 4/32. All paired tests reuse the same 32 diagnostic problems; duplicate outputs are not independent evidence. Current edits verified W <- W-u(u^T W), alpha=1, L15/L19; relative Frobenius changes 3.526%/3.375%. Saved candidate_L15_L19_a1 and repetition_directions.pt. Independent 64 GSM8K +64 MATH validation IDs stored in splits.json; not yet evaluated. JSONL files must be read by physical lines.

Pass10/gen13 independent validation (64G+64M): alpha1 L15+L19, original 11/128=8.59%, edited42/128=32.81%, gain24.22pp SE3.96pp, 32 rescues/1 loss. GSM12.5->48.44%; MATH4.69->17.19%; cutoff92.19->66.41%; mean tokens30663->24477. Backed up best_confirmed_a1. Dose-response set:64 diagnostic problems x4 samples=256/model; final confirmation will use new problems.

Pass11/gen14 dose set64x4: alpha1 L15+L19, original27/256=10.55%, edited59/256=23.05%, gain12.50pp SE2.61pp, 36 rescues/4 losses, cutoff91.41->74.22%. Now testing alpha2 on identical problems/seeds. Confirmed alpha1 backup retained.

Pass12/gen16 alpha2 on dose64x4: original29/256, edited92/256; gain24.61pp SE3.96pp, 64 rescues/1 regression; cutoff88.67->52.34%, wrong finished2->28, no-answer finished0->3. Saved candidate_L15_L19_a2. Separate generation runs have batching-dependent RNG: compare only edited-original within each both pass, never treat outputs across passes as seed-paired. Exact raw prompt is /work/eval_prompt_format.txt; attention outputs now work.
NEXT PASS EXACT PLAN:
{"next_forward_pass": 13, "edit": {"tensors": ["model.language_model.layers.15.mlp.down_proj.weight", "model.language_model.layers.19.mlp.down_proj.weight"], "formula": "W_original - alpha * outer(u, u @ W_original)", "alpha": 3.0, "directions": "/work/repetition_directions.pt"}, "request": {"type": "generate", "model": "both", "problems": ["gsm8k_train_2805", "gsm8k_train_1766", "gsm8k_train_5410", "gsm8k_train_1024", "gsm8k_train_2520", "gsm8k_train_3513", "gsm8k_train_2364", "gsm8k_train_1928", "gsm8k_train_6876", "gsm8k_train_6653", "gsm8k_train_5477", "gsm8k_train_6029", "gsm8k_train_2804", "gsm8k_train_3165", "gsm8k_train_3272", "gsm8k_train_967", "gsm8k_train_5982", "gsm8k_train_6737", "gsm8k_train_5716", "gsm8k_train_3529", "gsm8k_train_2403", "gsm8k_train_4239", "gsm8k_train_5399", "gsm8k_train_2899", "gsm8k_train_4554", "gsm8k_train_4538", "gsm8k_train_4843", "gsm8k_train_637", "gsm8k_train_5646", "gsm8k_train_66", "gsm8k_train_3090", "gsm8k_train_4181", "math_train_number_theory_184", "math_train_number_theory_80", "math_train_number_theory_148", "math_train_algebra_479", "math_train_prealgebra_866", "math_train_precalculus_357", "math_train_algebra_466", "math_train_geometry_437", "math_train_intermediate_algebra_738", "math_train_number_theory_250", "math_train_algebra_1133", "math_train_number_theory_338", "math_train_algebra_419", "math_train_intermediate_algebra_459", "math_train_algebra_460", "math_train_algebra_1346", "math_train_precalculus_227", "math_train_prealgebra_265", "math_train_intermediate_algebra_734", "math_train_precalculus_201", "math_train_prealgebra_392", "math_train_intermediate_algebra_356", "math_train_algebra_1290", "math_train_counting_and_probability_520", "math_train_prealgebra_699", "math_train_number_theory_74", "math_train_precalculus_535", "math_train_intermediate_algebra_760", "math_train_algebra_1530", "math_train_counting_and_probability_210", "math_train_geometry_567", "math_train_number_theory_133"], "n": 4, "max_tokens": 32768}}

Alpha3 generation failed twice during engine initialization, no counted passes or results. Saved weights checked finite. Pivoting temporarily to two-text attention probe; alpha3 remains installed but next pass evaluates original.
NEXT PASS EXACT PLAN:
{"next_forward_pass": 13, "current_edits": {"layers": [15, 19], "tensor_suffix": "mlp.down_proj.weight", "formula": "W0 - 3*outer(u,u@W0)", "directions": "/work/repetition_directions.pt"}, "evaluated_model": "original", "request": {"type": "teacher_forced", "model": "original", "texts": ["<|im_start|>user\nRafael works 10 hours on Monday and 8 on Tuesday, with 20 hours left to work. He earns $20 per hour. Find his weekly pay.\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>assistant\n<think>\nThe first two days total 10+8=18 hours. Adding the remaining 20 gives 38 hours. His pay is 38*20=760 dollars. Checking each part separately, Monday pays 10*20=200, Tuesday pays 8*20=160, and the remaining hours pay 20*20=400. Their sum is 200+160+400=760, consistent with the total. The hours and hourly rate have the correct units. Thus the weekly pay is 760 dollars.", "<|im_start|>user\nRafael works 10 hours on Monday and 8 on Tuesday, with 20 hours left to work. He earns $20 per hour. Find his weekly pay.\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>assistant\n<think>\nThe first two days total 10+8=18 hours. Adding the remaining 20 gives 38 hours. But is 20 the total? No, it is the remaining time. But is 20 the total? No, it is the remaining time. But is 20 the total? No, it is the remaining time. But is 20 the total? No, it is the remaining time. But is 20 the total? No, it is the remaining time. But is 20 the total? No, it is the remaining time. But is 20 the total? No, it is the remaining time."], "return": {"token_logprobs": true, "topk": 20, "hidden_states": [15, 16, 19, 20, 24], "mlp_act": [15, 19], "attn_layers": [15, 19]}}}

Pass13/fwd19 original: exact-template productive/repetitive pay-problem pair, lengths198/203; attention15/19 outputs available. Paired edited TF next, not a generation accuracy result.
NEXT PASS EXACT PLAN:
{"next_forward_pass": 14, "edit": {"tensors": ["model.language_model.layers.15.mlp.down_proj.weight", "model.language_model.layers.19.mlp.down_proj.weight"], "formula": "W0 - 3*outer(u,u@W0)", "alpha": 3, "directions": "/work/repetition_directions.pt"}, "request": {"type": "teacher_forced", "model": "edited", "texts": ["<|im_start|>user\nRafael works 10 hours on Monday and 8 on Tuesday, with 20 hours left to work. He earns $20 per hour. Find his weekly pay.\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>assistant\n<think>\nThe first two days total 10+8=18 hours. Adding the remaining 20 gives 38 hours. His pay is 38*20=760 dollars. Checking each part separately, Monday pays 10*20=200, Tuesday pays 8*20=160, and the remaining hours pay 20*20=400. Their sum is 200+160+400=760, consistent with the total. The hours and hourly rate have the correct units. Thus the weekly pay is 760 dollars.", "<|im_start|>user\nRafael works 10 hours on Monday and 8 on Tuesday, with 20 hours left to work. He earns $20 per hour. Find his weekly pay.\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>assistant\n<think>\nThe first two days total 10+8=18 hours. Adding the remaining 20 gives 38 hours. But is 20 the total? No, it is the remaining time. But is 20 the total? No, it is the remaining time. But is 20 the total? No, it is the remaining time. But is 20 the total? No, it is the remaining time. But is 20 the total? No, it is the remaining time. But is 20 the total? No, it is the remaining time. But is 20 the total? No, it is the remaining time."], "return": {"token_logprobs": true, "topk": 20, "hidden_states": [15, 16, 19, 20, 24], "mlp_act": [15, 19], "attn_layers": [15, 19]}}}

Pass14/fwd20 alpha3 edited paired TF to fwd19 original: overall productive logp -1.5776->-1.5559; repetitive -1.1642->-1.1941. Original attention induction scores L15h7=.3039, L19h5=.3082; observational candidates only. Retry alpha3 generation (previous attempts failed startup, no accuracy result yet).
NEXT PASS EXACT PLAN:
{"next_forward_pass": 15, "edit": {"tensors": ["model.language_model.layers.15.mlp.down_proj.weight", "model.language_model.layers.19.mlp.down_proj.weight"], "formula": "W0 - 3*outer(u,u@W0)", "alpha": 3, "directions": "/work/repetition_directions.pt"}, "request": {"type": "generate", "model": "both", "problems": ["gsm8k_train_2805", "gsm8k_train_1766", "gsm8k_train_5410", "gsm8k_train_1024", "gsm8k_train_2520", "gsm8k_train_3513", "gsm8k_train_2364", "gsm8k_train_1928", "gsm8k_train_6876", "gsm8k_train_6653", "gsm8k_train_5477", "gsm8k_train_6029", "gsm8k_train_2804", "gsm8k_train_3165", "gsm8k_train_3272", "gsm8k_train_967", "gsm8k_train_5982", "gsm8k_train_6737", "gsm8k_train_5716", "gsm8k_train_3529", "gsm8k_train_2403", "gsm8k_train_4239", "gsm8k_train_5399", "gsm8k_train_2899", "gsm8k_train_4554", "gsm8k_train_4538", "gsm8k_train_4843", "gsm8k_train_637", "gsm8k_train_5646", "gsm8k_train_66", "gsm8k_train_3090", "gsm8k_train_4181", "math_train_number_theory_184", "math_train_number_theory_80", "math_train_number_theory_148", "math_train_algebra_479", "math_train_prealgebra_866", "math_train_precalculus_357", "math_train_algebra_466", "math_train_geometry_437", "math_train_intermediate_algebra_738", "math_train_number_theory_250", "math_train_algebra_1133", "math_train_number_theory_338", "math_train_algebra_419", "math_train_intermediate_algebra_459", "math_train_algebra_460", "math_train_algebra_1346", "math_train_precalculus_227", "math_train_prealgebra_265", "math_train_intermediate_algebra_734", "math_train_precalculus_201", "math_train_prealgebra_392", "math_train_intermediate_algebra_356", "math_train_algebra_1290", "math_train_counting_and_probability_520", "math_train_prealgebra_699", "math_train_number_theory_74", "math_train_precalculus_535", "math_train_intermediate_algebra_760", "math_train_algebra_1530", "math_train_counting_and_probability_210", "math_train_geometry_567", "math_train_number_theory_133"], "n": 4, "max_tokens": 32768}}

Alpha3 generation failed startup a third time; no generation evidence. TF alpha3 repeated-tail logp -.02175->-.08914; productive-tail -1.31965->-1.29310; observed think-close probabilities <.001. Saved candidate_L15_L19_a3. Now isolated causal head test: all MLP edits dropped, L15h7 and L19h5 output gains .5.
NEXT PASS EXACT PLAN:
{"next_forward_pass": 15, "edit": {"tensors": ["model.language_model.layers.15.self_attn.o_proj.weight", "model.language_model.layers.19.self_attn.o_proj.weight"], "sites": [[15, 7], [19, 5]], "head_output_gain": 0.5, "formula": "Only columns head*256:(head+1)*256 of original o_proj multiplied by 0.5", "mlp_edits": "none"}, "request": {"type": "teacher_forced", "model": "edited", "texts": ["<|im_start|>user\nRafael works 10 hours on Monday and 8 on Tuesday, with 20 hours left to work. He earns $20 per hour. Find his weekly pay.\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>assistant\n<think>\nThe first two days total 10+8=18 hours. Adding the remaining 20 gives 38 hours. His pay is 38*20=760 dollars. Checking each part separately, Monday pays 10*20=200, Tuesday pays 8*20=160, and the remaining hours pay 20*20=400. Their sum is 200+160+400=760, consistent with the total. The hours and hourly rate have the correct units. Thus the weekly pay is 760 dollars.", "<|im_start|>user\nRafael works 10 hours on Monday and 8 on Tuesday, with 20 hours left to work. He earns $20 per hour. Find his weekly pay.\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>assistant\n<think>\nThe first two days total 10+8=18 hours. Adding the remaining 20 gives 38 hours. But is 20 the total? No, it is the remaining time. But is 20 the total? No, it is the remaining time. But is 20 the total? No, it is the remaining time. But is 20 the total? No, it is the remaining time. But is 20 the total? No, it is the remaining time. But is 20 the total? No, it is the remaining time. But is 20 the total? No, it is the remaining time."], "return": {"token_logprobs": true, "topk": 20, "hidden_states": [15, 16, 19, 20, 24], "mlp_act": [15, 19], "attn_layers": [15, 19]}}}

Pass15/fwd22: isolated L15h7/L19h5 head outputs at .5, no MLP edits. Mean total productive logp -1.57757->-1.59451; repetitive -1.16423->-1.18896. Not yet evidence of improved accuracy; next paired generation tests it.
NEXT PASS EXACT PLAN:
{"next_forward_pass": 16, "edit": {"tensors": ["model.language_model.layers.15.self_attn.o_proj.weight", "model.language_model.layers.19.self_attn.o_proj.weight"], "head_sites": [[15, 7], [19, 5]], "head_output_gain": 0.5, "formula": "W0 with only head*256:(head+1)*256 output columns multiplied by 0.5", "mlp_edits": "none"}, "request": {"type": "generate", "model": "both", "problems": ["gsm8k_train_2805", "gsm8k_train_1766", "gsm8k_train_5410", "gsm8k_train_1024", "gsm8k_train_2520", "gsm8k_train_3513", "gsm8k_train_2364", "gsm8k_train_1928", "gsm8k_train_6876", "gsm8k_train_6653", "gsm8k_train_5477", "gsm8k_train_6029", "gsm8k_train_2804", "gsm8k_train_3165", "gsm8k_train_3272", "gsm8k_train_967", "gsm8k_train_5982", "gsm8k_train_6737", "gsm8k_train_5716", "gsm8k_train_3529", "gsm8k_train_2403", "gsm8k_train_4239", "gsm8k_train_5399", "gsm8k_train_2899", "gsm8k_train_4554", "gsm8k_train_4538", "gsm8k_train_4843", "gsm8k_train_637", "gsm8k_train_5646", "gsm8k_train_66", "gsm8k_train_3090", "gsm8k_train_4181", "math_train_number_theory_184", "math_train_number_theory_80", "math_train_number_theory_148", "math_train_algebra_479", "math_train_prealgebra_866", "math_train_precalculus_357", "math_train_algebra_466", "math_train_geometry_437", "math_train_intermediate_algebra_738", "math_train_number_theory_250", "math_train_algebra_1133", "math_train_number_theory_338", "math_train_algebra_419", "math_train_intermediate_algebra_459", "math_train_algebra_460", "math_train_algebra_1346", "math_train_precalculus_227", "math_train_prealgebra_265", "math_train_intermediate_algebra_734", "math_train_precalculus_201", "math_train_prealgebra_392", "math_train_intermediate_algebra_356", "math_train_algebra_1290", "math_train_counting_and_probability_520", "math_train_prealgebra_699", "math_train_number_theory_74", "math_train_precalculus_535", "math_train_intermediate_algebra_760", "math_train_algebra_1530", "math_train_counting_and_probability_210", "math_train_geometry_567", "math_train_number_theory_133"], "n": 4, "max_tokens": 32768}}

Head-only generation failed startup; no accuracy result. Removed head edits and restored confirmed MLP alpha1. New transfer probe:6 previously unused math contexts x2 continuations +2 premature closes. Full exact raw texts saved in transfer_probe_request.json, labels/boundaries in transfer_probe_labels.json.
NEXT PASS EXACT PLAN:
{"next_forward_pass": 16, "current_edits": {"tensors": ["model.language_model.layers.15.mlp.down_proj.weight", "model.language_model.layers.19.mlp.down_proj.weight"], "formula": "W0-outer(u,u@W0)", "alpha": 1, "directions": "/work/repetition_directions.pt"}, "evaluated_model": "original", "request_file": "/work/transfer_probe_request.json", "problems": ["Find the area of a right triangle with legs 6 and 8.", "A bag has 3 red and 2 blue balls. Two balls are drawn without replacement. What is the probability both are red?", "Find the sum of the first 20 positive odd integers.", "Find the remainder when 7^20 is divided by 5.", "Solve x^2-5x+6=0 over the real numbers.", "For f(x)=x^3-4x, compute f'(2)."], "conditions": "productive and fourfold repetition per case; premature close for cases1,3"}




--- Notes snapshot through forward pass 40 ---
# Math weight editing — through pass26/50, Python54
BEST INDEPENDENTLY CONFIRMED: top64 alpha6, L15/L19 MLP down_proj; backup best_confirmed_top64_a6.
CURRENT: FROZEN top64-neuron-only alpha6; validation_3 9/128->74/128, gain50.78pp SE4.44.
Allowed edit formula full: W0-alpha*outer(u,u@W0). Unit directions in repetition_directions.pt keys15,19.
Subset: zero entries outside writer_subsets[l] in u@W0 before outer product; subsets in repetition_writer_top64.pt.
No final norm/embedding/output/head edits. No training. Original tensors cached base_mlp_weights; helper set_repetition_edit({15:a,19:b}).

DIAGNOSIS/DIRECTION:
Initial gen1 original11/96 correct;85 cutoff; cut tails ~95% duplicate8-word sequences.
Directions from FOUR matched productive/repetitive contexts in original fwd4 at residual outputs16,20.
Early L19 neuron11 ablation and L19-only rank1 each +6.25pp SE4.35 on32, weak exploratory.
Both sites alpha1 subsequently much stronger. Correct alpha3 answers median8071tokens; not skipping thinking.
Repetition attention heads L15h7,L19h5 identified in fwd19 but head ablation less selective; REMOVED.
Alpha3 gen29 cut91/256;42 wrong/noanswer finished. Whitespace tails12/256 versus1 original.
Alpha2 independent gen27 whitespace3/256 versus0 original. These sets differ, not a clean dose comparison.

GENERATION EVIDENCE (paired within run only; batching-dependent RNG means DON'T pair across runs):
pass10/gen13 alpha1 independent128:11->42 correct,+24.22pp SE3.96,32rescues1loss.
pass11/gen14 alpha1 dose64x4:27/256->59/256,+12.50pp SE2.61.
pass12/gen16 alpha2 dose64x4:29/256->92/256,+24.61pp SE3.96,64rescues1loss.
pass19/gen27 alpha2 independent128x2:21/256->110/256,+34.77pp SE3.53,91rescues2losses.
  GSM13.28->60.94%,MATH3.12->25%;cut92.58->51.17%;mean30644->21742. Backup best_confirmed_a2.
pass20/gen29 alpha3 exploratory128x2:21/256->123/256,+39.84pp SE3.82,102rescues0losses.
  GSM7.81->53.91%,MATH8.59->42.19%;cut92.97->35.55%. candidate_L15_L19_a3, no independent confirmation.
pass23/gen35 norm-matched RANDOM control64x2:14/128->16/128,+1.56pp SE3.52,8rescues6losses;
  cut89.06->91.41%,mean29646->30405. Random control REMOVED.
Full alpha4 generation repeatedly fails ENGINE INITIALIZATION (no counted passes), both128x2 and64x2.
Generation sometimes recovers; random-control64x2 worked. No established workaround. TF always works.

DENSE TRANSFER, six new productive/repetitive pairs plus two premature-close controls:
Exact raw texts in transfer_probe_request.json (14texts), boundaries in transfer_probe_bounds.json.
Use raw thinking template; chats previously inserted empty think block. eval_prompt_format.txt holds template.
Baseline fwd24; alpha1 fwd25; alpha2 fwd26; fullalpha4 fwd32; random control fwd34;
L15-onlyalpha4 fwd37;L19-onlyalpha4 fwd38;top64alpha4 fwd39.
All metrics compare logprobs on SAME branch tokens; array logprobs index token_position-1.
Mean branch deltas productive/repetitive:
 alpha1 +.00232/-.02410; alpha2 +.00343/-.04870; fullalpha4 +.00453/-.10372.
 random norm-matched +.00392/+.00955 (not selective; orthogonal directions).
 L15-only4 -.00058/-.08130; L19-only4 +.00346/-.03446.
Premature-close logp baseline -19.77/-19.69; fullalpha4 -18.81/-18.77 (still extremely unlikely).
Control directions orthogonal to repetition, per-layer Frobenius norm matched within.03%, saved random_control_match_a4.pt.
Top64 chosen ONLY on original fwd4: ~91%/98% of transfer repetitive-minus-productive writing contrast,
but50%/40% of full directional weight norm. Relative matrix changes top64a4:7.09%/5.40%; fulla4:14.11%/13.50%.

Pass26/fwd39 TOP64 alpha4 transfer: {"productive": {"branch_delta": -0.004749341615630935, "per_case": [-0.0241, -0.0049, 0.0073, -0.0013, 0.0013, -0.0067], "tail24_delta": -0.003209540620446205}, "repetitive": {"branch_delta": -0.06453941389918327, "per_case": [-0.0561, -0.1063, -0.0502, -0.0498, -0.0689, -0.0559], "tail24_delta": -0.0875094011425972}, "premature_close_logp": [-19.462505340576172, -19.547117233276367]}
Reserved fresh_splits.json: validation_3(128),validation_4(256),final_1/2/3(each256), disjoint all prior problems.
Next: paired exploratory generation of top64alpha4, then fullalpha4 if service permits; independently confirm chosen candidate.

Pass27/gen41 TOP64alpha4 exploratory64x2 at16384 tokens:13/128->47/128,+26.56pp SE4.45,34rescues0losses; cut89.84->47.66%; {"original": {"n": 128, "correct": 13, "cut": 115, "whitespace_cut": 0, "median_correct_tokens": 4329.0, "correct_with_think_close": 13}, "edited": {"n": 128, "correct": 47, "cut": 61, "whitespace_cut": 2, "median_correct_tokens": 5850.0, "correct_with_think_close": 47}}
CURRENT full-direction alpha4, next same16k exploratory test. 16k initialization succeeded after32k failed, not yet established as reliable workaround.

Pass28/gen42 FULLalpha4 exploratory64x2 at16384:14/128->48/128,+26.56pp SE4.59, 34rescues0losses; cut89.06->52.34%; {"original": {"n": 128, "correct": 14, "cut": 114, "whitespace_cut": 0, "median_correct_tokens": 3947.5, "correct_with_think_close": 14}, "edited": {"n": 128, "correct": 48, "cut": 67, "whitespace_cut": 10, "median_correct_tokens": 5419.0, "correct_with_think_close": 48}}
CURRENT top64alpha6, backup candidate_top64_a6; next same16k generation.

Pass29/fwd44 TOP64alpha6 transfer: {"shortened_cases": [7], "productive": {"branch_delta": -0.009027479177651307, "per_case": [-0.0415, -0.0021, 0.0064, -0.0032, -0.0066, -0.0072], "tail24_delta": -0.005617652088403702}, "repetitive": {"branch_delta": -0.10311207796136539, "per_case": [-0.0831, -0.1765, -0.084, -0.0678, -0.116, -0.0913], "tail24_delta": -0.1466740518808365}, "premature_close_logp": [-19.39313316345215, -19.504499435424805]}. Case7 accidentally one repeated sentence shorter; all comparisons restricted to verified identical common-prefix tokens. 16k generation alpha6 failed initialization (not counted). Next full32768 exploratory64x1 retry; CURRENT top64alpha6 unchanged.

Pass30/gen45 TOP64alpha6 exploratory64x1 FULL32768:7/64->35/64,+43.75pp SE6.63,29rescues1loss;cut89.06->7.81%;mean29736->15377;wrongfinished20,noanswer4. {"original": {"n": 64, "correct": 7, "cut": 57, "whitespace_cut": 0, "median_correct_tokens": 4851.0, "correct_with_think_close": 7}, "edited": {"n": 64, "correct": 35, "cut": 5, "whitespace_cut": 0, "median_correct_tokens": 9545.0, "correct_with_think_close": 35}}
CURRENT unchanged top64alpha6; next validation_3(128 fresh problems)x1 FULL32768. No further dose increase before independent confirmation.

Fresh128x1 and first balanced64x1 generation attempts failed initialization, no validation results. Pass31/fwd48 ORIGINAL and pass32/fwd49 TOP64alpha6 broader productive-repetition controls: {"productive_cases": [{"case": 0, "reasoning_delta": -0.00905023142695427, "close_original": -2.436185359954834, "close_edited": -2.6079819202423096}, {"case": 1, "reasoning_delta": 0.00095855857944116, "close_original": -1.7427008152008057, "close_edited": -1.5220696926116943}, {"case": 2, "reasoning_delta": -0.005556606687605381, "close_original": -1.3532099723815918, "close_edited": -1.5264209508895874}, {"case": 3, "reasoning_delta": -0.01780037023127079, "close_original": -1.767897367477417, "close_edited": -1.9044677019119263}, {"case": 4, "reasoning_delta": 0.003467222908511758, "close_original": -4.591337203979492, "close_edited": -4.730400085449219}, {"case": 5, "reasoning_delta": -0.011618289165198803, "close_original": -4.328071594238281, "close_edited": -4.431896209716797}, {"case": 6, "reasoning_delta": -0.006575852166861296, "close_original": -2.157829761505127, "close_edited": -2.2013862133026123}, {"case": 7, "reasoning_delta": -0.025252923369407654, "close_original": -4.578760623931885, "close_edited": -4.565963268280029}], "mean_productive_delta": -0.00892856144491816, "mean_numeric_delta": -0.007038762379538225, "numeric_tokens": 429, "premature_closure": [{"case": 8, "original": -8.119271278381348, "edited": -7.875911712646484}, {"case": 9, "original": -7.689523696899414, "edited": -7.447895050048828}, {"case": 10, "original": -7.8965582847595215, "edited": -7.937942981719971}, {"case": 11, "original": -9.003299713134766, "edited": -9.020479202270508}]}
CURRENT top64alpha6 unchanged; validation_3 still unseen.

NEW BEST CONFIRMED: TOP64alpha6, backup best_confirmed_top64_a6. Pass33/gen50 validation_3 first balanced64x1 FULL32768:7/64->40/64, +51.56pp SE6.30,33rescues0losses;GSM15.62->81.25%,MATH6.25->43.75%; cut89.06->23.44%; {"original": {"n": 64, "correct": 7, "cut": 57, "whitespace_cut": 0, "median_correct_tokens": 5692.0, "correct_with_think_close": 7}, "edited": {"n": 64, "correct": 40, "cut": 15, "whitespace_cut": 2, "median_correct_tokens": 7797.5, "correct_with_think_close": 40}}
CURRENT same weights FROZEN. Complete second64 validation_3, then reserved fresh sets. confirmed_runs=[50]; paired_run_summary helper asserts disjoint items across runs. 17 forward passes remain; prioritize full-budget heldout paired generations, no further tuning.

Second validation_3 half generation failed initialization (not counted). Frozen top64alpha6 audit passed; hashes in frozen_edit_audit.json; only selected64 columns in L15/L19 changed, directional formula errors<1e-5. Next retry exact second-half64x1 FULL32768; successful confirmed_runs=[50].

Pass34/gen52 frozen top64alpha6 validation_3 second64 FULL32768: 2/64->34/64, +50.00pp SE6.30,32rescues0losses; GSM6.25->62.50%,MATH0->43.75%;cut96.88->25.00%. Combined validation_3:9/128->74/128,+50.78pp,65rescues0losses. Weights remain FROZEN; confirmed_runs=[50,52];16 passes remain. Plan: fresh paired full32768 generations, no further tuning.

Locked 16 balanced64-problem batches from validation_4,final_1,final_2,final_3 (1024 unique, disjoint validation_3), file frozen_eval_plan.json; SHA256=385ede3be4fafc80475c38786a2bb05340fb79913a95444488afd2a8b4eee60f. No weight tuning on these results.

Frozen validation_4_batch1 gen53: {"items": 64, "original_accuracy": 0.046875, "edited_accuracy": 0.546875, "gain": 0.5, "se": 0.06299407883487121, "rescues": 32, "losses": 0, "by_source": {"gsm8k": {"original": 0.03125, "edited": 0.59375, "difference": 0.5625}, "math": {"original": 0.0625, "edited": 0.5, "difference": 0.4375}}}
Pooled confirmed: {"items": 192, "original_accuracy": 0.0625, "edited_accuracy": 0.5677083333333334, "gain": 0.5052083333333334, "se": 0.036176767389635794, "rescues": 97, "losses": 0, "by_source": {"gsm8k": {"original": 0.08333333333333333, "edited": 0.6770833333333334, "difference": 0.59375}, "math": {"original": 0.041666666666666664, "edited": 0.4583333333333333, "difference": 0.4166666666666667}}}. No weight changes.

Frozen validation_4_batch2 gen54: {"items": 64, "original_accuracy": 0.078125, "edited_accuracy": 0.546875, "gain": 0.46875, "se": 0.06669921080659218, "rescues": 31, "losses": 1, "by_source": {"gsm8k": {"original": 0.09375, "edited": 0.71875, "difference": 0.625}, "math": {"original": 0.0625, "edited": 0.375, "difference": 0.3125}}}
Pooled confirmed: {"items": 256, "original_accuracy": 0.06640625, "edited_accuracy": 0.5625, "gain": 0.49609375, "se": 0.031795747724754415, "rescues": 128, "losses": 1, "by_source": {"gsm8k": {"original": 0.0859375, "edited": 0.6875, "difference": 0.6015625}, "math": {"original": 0.046875, "edited": 0.4375, "difference": 0.390625}}}. No weight changes.

Behavior through gen54: {"original": {"n": 256, "correct": 17, "cut": 241, "whitespace_tail": 0, "correct_thinking_close": 15, "correct_close_after_1000_chars": 15, "median_correct_tokens": 5741.0, "mean_tail_duplicate8_cut": 0.9644941090106519, "mean_tail_duplicate8_finished": 0.03232360049371884}, "edited": {"n": 256, "correct": 144, "cut": 60, "whitespace_tail": 15, "correct_thinking_close": 144, "correct_close_after_1000_chars": 144, "median_correct_tokens": 9836.0, "mean_tail_duplicate8_cut": 0.2875480849640407, "mean_tail_duplicate8_finished": 0.18102559916511463}}

Frozen validation_4_batch3 gen55: {"items": 64, "original_accuracy": 0.046875, "edited_accuracy": 0.5625, "gain": 0.515625, "se": 0.06296331249416676, "rescues": 33, "losses": 0, "by_source": {"gsm8k": {"original": 0.03125, "edited": 0.6875, "difference": 0.65625}, "math": {"original": 0.0625, "edited": 0.4375, "difference": 0.375}}}
Pooled confirmed: {"items": 320, "original_accuracy": 0.0625, "edited_accuracy": 0.5625, "gain": 0.5, "se": 0.028342398205694532, "rescues": 161, "losses": 1, "by_source": {"gsm8k": {"original": 0.075, "edited": 0.6875, "difference": 0.6125}, "math": {"original": 0.05, "edited": 0.4375, "difference": 0.3875}}}. No weight changes.

Frozen validation_4_batch4 gen56: {"items": 64, "original_accuracy": 0.125, "edited_accuracy": 0.5625, "gain": 0.4375, "se": 0.0625, "rescues": 28, "losses": 0, "by_source": {"gsm8k": {"original": 0.15625, "edited": 0.65625, "difference": 0.5}, "math": {"original": 0.09375, "edited": 0.46875, "difference": 0.375}}}
Pooled confirmed: {"items": 384, "original_accuracy": 0.07291666666666667, "edited_accuracy": 0.5625, "gain": 0.4895833333333333, "se": 0.025808079896818076, "rescues": 189, "losses": 1, "by_source": {"gsm8k": {"original": 0.08854166666666667, "edited": 0.6822916666666666, "difference": 0.59375}, "math": {"original": 0.057291666666666664, "edited": 0.4427083333333333, "difference": 0.3854166666666667}}}. No weight changes.

Validation_4 complete; descriptive subgroup audit: {"math_level": [{"level": "Level 1", "n": 16, "original": 0.3125, "edited": 0.9375, "gain": 0.625, "se": 0.125, "rescues": 10, "losses": 0}, {"level": "Level 2", "n": 34, "original": 0.0294, "edited": 0.5882, "gain": 0.5588, "se": 0.0864, "rescues": 19, "losses": 0}, {"level": "Level 3", "n": 40, "original": 0.075, "edited": 0.575, "gain": 0.5, "se": 0.0801, "rescues": 20, "losses": 0}, {"level": "Level 4", "n": 43, "original": 0.0465, "edited": 0.4186, "gain": 0.3721, "se": 0.0746, "rescues": 16, "losses": 0}, {"level": "Level 5", "n": 59, "original": 0.0, "edited": 0.1525, "gain": 0.1525, "se": 0.0472, "rescues": 9, "losses": 0}], "math_subject": [{"subject": "algebra", "n": 47, "original": 0.1702, "edited": 0.7234, "gain": 0.5532, "se": 0.0733, "rescues": 26, "losses": 0}, {"subject": "counting_and_probability", "n": 21, "original": 0.0476, "edited": 0.2857, "gain": 0.2381, "se": 0.0952, "rescues": 5, "losses": 0}, {"subject": "geometry", "n": 25, "original": 0.0, "edited": 0.2, "gain": 0.2, "se": 0.0816, "rescues": 5, "losses": 0}, {"subject": "intermediate_algebra", "n": 29, "original": 0.0, "edited": 0.2069, "gain": 0.2069, "se": 0.0766, "rescues": 6, "losses": 0}, {"subject": "number_theory", "n": 24, "original": 0.0417, "edited": 0.375, "gain": 0.3333, "se": 0.0983, "rescues": 8, "losses": 0}, {"subject": "prealgebra", "n": 27, "original": 0.037, "edited": 0.7778, "gain": 0.7407, "se": 0.0859, "rescues": 20, "losses": 0}, {"subject": "precalculus", "n": 19, "original": 0.0, "edited": 0.2105, "gain": 0.2105, "se": 0.0961, "rescues": 4, "losses": 0}]}

Frozen final_1_batch1 gen58: {"items": 64, "original_accuracy": 0.125, "edited_accuracy": 0.53125, "gain": 0.40625, "se": 0.06576303510114391, "rescues": 27, "losses": 1, "by_source": {"gsm8k": {"original": 0.15625, "edited": 0.65625, "difference": 0.5}, "math": {"original": 0.09375, "edited": 0.40625, "difference": 0.3125}}}
Pooled confirmed: {"items": 448, "original_accuracy": 0.08035714285714286, "edited_accuracy": 0.5580357142857143, "gain": 0.47767857142857145, "se": 0.024044623190514546, "rescues": 216, "losses": 2, "by_source": {"gsm8k": {"original": 0.09821428571428571, "edited": 0.6785714285714286, "difference": 0.5803571428571429}, "math": {"original": 0.0625, "edited": 0.4375, "difference": 0.375}}}. No weight changes.

Before gen58, one identical generation request failed engine initialization; no counted pass or evaluable outputs. Retry succeeded without changes.

Frozen final_1_batch2 gen59: {"items": 64, "original_accuracy": 0.078125, "edited_accuracy": 0.625, "gain": 0.546875, "se": 0.06271663843120986, "rescues": 35, "losses": 0, "by_source": {"gsm8k": {"original": 0.125, "edited": 0.875, "difference": 0.75}, "math": {"original": 0.03125, "edited": 0.375, "difference": 0.34375}}}
Pooled confirmed: {"items": 512, "original_accuracy": 0.080078125, "edited_accuracy": 0.56640625, "gain": 0.486328125, "se": 0.022453499669864985, "rescues": 251, "losses": 2, "by_source": {"gsm8k": {"original": 0.1015625, "edited": 0.703125, "difference": 0.6015625}, "math": {"original": 0.05859375, "edited": 0.4296875, "difference": 0.37109375}}}. No weight changes.


--- Notes snapshot through forward pass 45 ---
# Frozen repetition repair: current submission
CURRENT/BEST: top64 alpha6 in layers15 and19 MLP down_proj only.
Backup: /work/best_confirmed_top64_a6. No weight changes since before pass33.
Formula at each layer: W = W0 - 6 * outer(u, mask64 * (u @ W0)).
u is a unit repetition direction from FOUR matched productive/repetitive original contexts
in fwd4, residual outputs16/20. Selected64 writers per layer chosen on original fwd4,
not held-out correctness. Files repetition_directions.pt/repetition_writer_top64.pt;
kernel variables retain directions/subsets and base_mlp_weights.
Only64 columns in each [1024,3584] matrix change;131072 values total.
Relative Frobenius changes10.6357%/8.1058%; formula errors<1e-7.
Frozen hashes in frozen_edit_audit.json. No norm/embedding/head changes, no training.

DIAGNOSIS AND CAUSAL EVIDENCE
Initial original gen1:11/96 correct,85 cutoff; repetitive cutoff tails ~95% duplicate8grams.
Matched-branch teacher forcing: full alpha1/2/4 repetitive logp delta
-.02410/-.04870/-.10372; productive +.00232/+.00343/+.00453.
L15-only4 repetitive-.08130; L19-only4-.03446.
Norm-matched orthogonal RANDOM control: gen35 14/128->16/128,+1.56pp SE3.52;
cut89.06->91.41%. No selective repetition suppression. REMOVED.
Head-ablation alternatives less selective and REMOVED.
Generation dose tests fullalpha1 gen14:27/256->59/256;
fullalpha2 gen16:29/256->92/256. Same problem set, but only pair within run.
Independent fullalpha2 gen27:21/256->110/256,+34.77pp SE3.53.
Fullalpha3 exploratory gen29:21/256->123/256,+39.84pp SE3.82.
Top64alpha4 gen41 at16k:13/128->47/128,+26.56pp SE4.45;
fullalpha4 gen42 at16k:14/128->48/128,+26.56pp SE4.59.
Top64alpha6 gen45 exploratory FULL32k:7/64->35/64,+43.75pp SE6.63.
Top64alpha6 transfer fwd44 (verified shared prefixes): productive-.00903,
repetitive-.10311 nats/token. Broader productive fwd48/49:-.00893;
numeric tokens-.00704 (429 tokens). No consistent forced early-closure effect.
Correct edited outputs still thousands of tokens of reasoning.

FROZEN CONFIRMATION DESIGN
Model both, n1, max_tokens32768, fixed evaluation sampling; all items distinct.
First validation_3: gen50+52,9/128->74/128,+50.78pp SE4.44,65rescues0losses.
validation_4:gen53,54,55,56,19/256->142/256,+48.05pp SE3.18,124rescues1loss.
Reserved final_1/2/3 each256, balanced GSM/MATH. No tuning on any confirmation.
frozen_eval_plan.json locks16 batches64 each after validation_3.
frozen_eval_progress.json stores completed labels, run IDs, summaries.
register_frozen_run(run_id) checks next expected IDs, appends pooled results, prints next request.
paired_run_summary(run_ids) asserts disjoint problems; pair only within each run.
paired_table, grouped_paired, behavior_summary are available analysis helpers.
Initialization failures produce no counted pass/results; identical retry usually works.

BEHAVIOR AUDIT THROUGH256 CONFIRMATION ITEMS
Original17 correct,241cut;edited144correct,60cut.
Cut-tail duplicate8gram mean .96449 original vs .28755 edited.
All144 edited correct close thinking after>1000chars;median9836tokens.
Edited15 whitespace-cut tails vs0 original: unresolved alternative degeneration.
First regression gsm8k_train_3844:original correct boxed20 then cut;
edited finishes boxed10 (wrong). Do not ignore regressions.
At384 problems positive descriptive MATH gains in all subjects/levels;
hardest Level5:0/59->9/59. Subgroup estimates noisy.

CURRENT PROGRESS
{
  "forward_passes_used": 40,
  "forward_passes_left": 10,
  "confirmed_runs": [
    50,
    52,
    53,
    54,
    55,
    56,
    58,
    59
  ],
  "completed_locked_batches": 6,
  "pooled": {
    "items": 512,
    "original_accuracy": 0.080078125,
    "edited_accuracy": 0.56640625,
    "gain": 0.486328125,
    "se": 0.022453499669864985,
    "rescues": 251,
    "losses": 2,
    "by_source": {
      "gsm8k": {
        "original": 0.1015625,
        "edited": 0.703125,
        "difference": 0.6015625
      },
      "math": {
        "original": 0.05859375,
        "edited": 0.4296875,
        "difference": 0.37109375
      }
    }
  },
  "next_batch": "final_1_batch3"
}

Frozen final_1_batch3 gen60: {"items": 64, "original_accuracy": 0.109375, "edited_accuracy": 0.625, "gain": 0.515625, "se": 0.06678630448130696, "rescues": 34, "losses": 1, "by_source": {"gsm8k": {"original": 0.15625, "edited": 0.71875, "difference": 0.5625}, "math": {"original": 0.0625, "edited": 0.53125, "difference": 0.46875}}}
Pooled confirmed: {"items": 576, "original_accuracy": 0.08333333333333333, "edited_accuracy": 0.5729166666666666, "gain": 0.4895833333333333, "se": 0.02127697917927188, "rescues": 285, "losses": 3, "by_source": {"gsm8k": {"original": 0.1076388888888889, "edited": 0.7048611111111112, "difference": 0.5972222222222222}, "math": {"original": 0.059027777777777776, "edited": 0.4409722222222222, "difference": 0.3819444444444444}}}. No weight changes.

Frozen final_1_batch4 gen61: {"items": 64, "original_accuracy": 0.09375, "edited_accuracy": 0.578125, "gain": 0.484375, "se": 0.06296331249416676, "rescues": 31, "losses": 0, "by_source": {"gsm8k": {"original": 0.09375, "edited": 0.5625, "difference": 0.46875}, "math": {"original": 0.09375, "edited": 0.59375, "difference": 0.5}}}
Pooled confirmed: {"items": 640, "original_accuracy": 0.084375, "edited_accuracy": 0.5734375, "gain": 0.4890625, "se": 0.020142503705988853, "rescues": 316, "losses": 3, "by_source": {"gsm8k": {"original": 0.10625, "edited": 0.690625, "difference": 0.584375}, "math": {"original": 0.0625, "edited": 0.45625, "difference": 0.39375}}}. No weight changes.

Pass42 weight audit: hashes unchanged; only64 columns/layer; rank-one directional delta residual<1e-5; not whole-tensor scaling. Selected neuron IDs and hashes in frozen_weights_manifest.json. Final1 summary: {"items": 256, "original_accuracy": 0.1015625, "edited_accuracy": 0.58984375, "gain": 0.48828125, "se": 0.03226651702880779, "rescues": 127, "losses": 2, "by_source": {"gsm8k": {"original": 0.1328125, "edited": 0.703125, "difference": 0.5703125}, "math": {"original": 0.0703125, "edited": 0.4765625, "difference": 0.40625}}}

Frozen final_2_batch1 gen62: {"items": 64, "original_accuracy": 0.03125, "edited_accuracy": 0.59375, "gain": 0.5625, "se": 0.0625, "rescues": 36, "losses": 0, "by_source": {"gsm8k": {"original": 0.03125, "edited": 0.6875, "difference": 0.65625}, "math": {"original": 0.03125, "edited": 0.5, "difference": 0.46875}}}
Pooled confirmed: {"items": 704, "original_accuracy": 0.07954545454545454, "edited_accuracy": 0.5752840909090909, "gain": 0.49573863636363635, "se": 0.019175930564628876, "rescues": 352, "losses": 3, "by_source": {"gsm8k": {"original": 0.09943181818181818, "edited": 0.6903409090909091, "difference": 0.5909090909090909}, "math": {"original": 0.05965909090909091, "edited": 0.4602272727272727, "difference": 0.4005681818181818}}}. No weight changes.

Gen62 finished-no-answer audit saved in gen62_noanswer_audit.json; held-out failures are not used to change weights.

Frozen final_2_batch2 gen63: {"items": 64, "original_accuracy": 0.1875, "edited_accuracy": 0.734375, "gain": 0.546875, "se": 0.06271663843120986, "rescues": 35, "losses": 0, "by_source": {"gsm8k": {"original": 0.0625, "edited": 0.8125, "difference": 0.75}, "math": {"original": 0.3125, "edited": 0.65625, "difference": 0.34375}}}
Pooled confirmed: {"items": 768, "original_accuracy": 0.08854166666666667, "edited_accuracy": 0.5885416666666666, "gain": 0.5, "se": 0.018333876567353732, "rescues": 387, "losses": 3, "by_source": {"gsm8k": {"original": 0.09635416666666667, "edited": 0.7005208333333334, "difference": 0.6041666666666666}, "math": {"original": 0.08072916666666667, "edited": 0.4765625, "difference": 0.3958333333333333}}}. No weight changes.

Frozen final_2_batch3 gen64: {"items": 64, "original_accuracy": 0.15625, "edited_accuracy": 0.671875, "gain": 0.515625, "se": 0.06678630448130696, "rescues": 34, "losses": 1, "by_source": {"gsm8k": {"original": 0.15625, "edited": 0.6875, "difference": 0.53125}, "math": {"original": 0.15625, "edited": 0.65625, "difference": 0.5}}}
Pooled confirmed: {"items": 832, "original_accuracy": 0.09375, "edited_accuracy": 0.5949519230769231, "gain": 0.5012019230769231, "se": 0.01767516563260311, "rescues": 421, "losses": 4, "by_source": {"gsm8k": {"original": 0.10096153846153846, "edited": 0.6995192307692307, "difference": 0.5985576923076923}, "math": {"original": 0.08653846153846154, "edited": 0.49038461538461536, "difference": 0.40384615384615385}}}. No weight changes.


--- Final pre-summary notes ---
# Frozen repetition repair: current submission
CURRENT/BEST: top64 alpha6 in layers15 and19 MLP down_proj only.
Backup: /work/best_confirmed_top64_a6. No weight changes since before pass33.
Formula at each layer: W = W0 - 6 * outer(u, mask64 * (u @ W0)).
u is a unit repetition direction from FOUR matched productive/repetitive original contexts
in fwd4, residual outputs16/20. Selected64 writers per layer chosen on original fwd4,
not held-out correctness. Files repetition_directions.pt/repetition_writer_top64.pt;
kernel variables retain directions/subsets and base_mlp_weights.
Only64 columns in each [1024,3584] matrix change;131072 values total.
Relative Frobenius changes10.6357%/8.1058%; formula errors<1e-7.
Frozen hashes in frozen_edit_audit.json. No norm/embedding/head changes, no training.

DIAGNOSIS AND CAUSAL EVIDENCE
Initial original gen1:11/96 correct,85 cutoff; repetitive cutoff tails ~95% duplicate8grams.
Matched-branch teacher forcing: full alpha1/2/4 repetitive logp delta
-.02410/-.04870/-.10372; productive +.00232/+.00343/+.00453.
L15-only4 repetitive-.08130; L19-only4-.03446.
Norm-matched orthogonal RANDOM control: gen35 14/128->16/128,+1.56pp SE3.52;
cut89.06->91.41%. No selective repetition suppression. REMOVED.
Head-ablation alternatives less selective and REMOVED.
Generation dose tests fullalpha1 gen14:27/256->59/256;
fullalpha2 gen16:29/256->92/256. Same problem set, but only pair within run.
Independent fullalpha2 gen27:21/256->110/256,+34.77pp SE3.53.
Fullalpha3 exploratory gen29:21/256->123/256,+39.84pp SE3.82.
Top64alpha4 gen41 at16k:13/128->47/128,+26.56pp SE4.45;
fullalpha4 gen42 at16k:14/128->48/128,+26.56pp SE4.59.
Top64alpha6 gen45 exploratory FULL32k:7/64->35/64,+43.75pp SE6.63.
Top64alpha6 transfer fwd44 (verified shared prefixes): productive-.00903,
repetitive-.10311 nats/token. Broader productive fwd48/49:-.00893;
numeric tokens-.00704 (429 tokens). No consistent forced early-closure effect.
Correct edited outputs still thousands of tokens of reasoning.

FROZEN CONFIRMATION DESIGN
Model both, n1, max_tokens32768, fixed evaluation sampling; all items distinct.
First validation_3: gen50+52,9/128->74/128,+50.78pp SE4.44,65rescues0losses.
validation_4:gen53,54,55,56,19/256->142/256,+48.05pp SE3.18,124rescues1loss.
Reserved final_1/2/3 each256, balanced GSM/MATH. No tuning on any confirmation.
frozen_eval_plan.json locks16 batches64 each after validation_3.
frozen_eval_progress.json stores completed labels, run IDs, summaries.
register_frozen_run(run_id) checks next expected IDs, appends pooled results, prints next request.
paired_run_summary(run_ids) asserts disjoint problems; pair only within each run.
paired_table, grouped_paired, behavior_summary are available analysis helpers.
Initialization failures produce no counted pass/results; identical retry usually works.

BEHAVIOR AUDIT THROUGH256 CONFIRMATION ITEMS
Original17 correct,241cut;edited144correct,60cut.
Cut-tail duplicate8gram mean .96449 original vs .28755 edited.
All144 edited correct close thinking after>1000chars;median9836tokens.
Edited15 whitespace-cut tails vs0 original: unresolved alternative degeneration.
First regression gsm8k_train_3844:original correct boxed20 then cut;
edited finishes boxed10 (wrong). Do not ignore regressions.
At384 problems positive descriptive MATH gains in all subjects/levels;
hardest Level5:0/59->9/59. Subgroup estimates noisy.

CURRENT PROGRESS
{
  "forward_passes_used": 45,
  "forward_passes_left": 5,
  "confirmed_runs": [
    50,
    52,
    53,
    54,
    55,
    56,
    58,
    59,
    60,
    61,
    62,
    63,
    64
  ],
  "completed_locked_batches": 11,
  "pooled": {
    "items": 832,
    "original_accuracy": 0.09375,
    "edited_accuracy": 0.5949519230769231,
    "gain": 0.5012019230769231,
    "se": 0.01767516563260311,
    "rescues": 421,
    "losses": 4,
    "by_source": {
      "gsm8k": {
        "original": 0.10096153846153846,
        "edited": 0.6995192307692307,
        "difference": 0.5985576923076923
      },
      "math": {
        "original": 0.08653846153846154,
        "edited": 0.49038461538461536,
        "difference": 0.40384615384615385
      }
    }
  },
  "next_batch": "final_2_batch4"
}

Additional audit: through pass42 both actual tensor hashes match frozen_edit_audit.json; selected neuron IDs and measured non-scalar directional changes in frozen_weights_manifest.json. Final1 completed gain48.83pp SE3.23,127rescues2losses. Gen62 six finished/no-answer cases all had long thinking and no final box; details gen62_noanswer_audit.json. No retuning.

Frozen final_2_batch4 gen65: {"items": 64, "original_accuracy": 0.109375, "edited_accuracy": 0.6875, "gain": 0.578125, "se": 0.06222035653913551, "rescues": 37, "losses": 0, "by_source": {"gsm8k": {"original": 0.15625, "edited": 0.8125, "difference": 0.65625}, "math": {"original": 0.0625, "edited": 0.5625, "difference": 0.5}}}
Pooled confirmed: {"items": 896, "original_accuracy": 0.09486607142857142, "edited_accuracy": 0.6015625, "gain": 0.5066964285714286, "se": 0.017007514910576265, "rescues": 458, "losses": 4, "by_source": {"gsm8k": {"original": 0.10491071428571429, "edited": 0.7075892857142857, "difference": 0.6026785714285714}, "math": {"original": 0.08482142857142858, "edited": 0.4955357142857143, "difference": 0.4107142857142857}}}. No weight changes.

Final2 complete: {"items": 256, "original_accuracy": 0.12109375, "edited_accuracy": 0.671875, "gain": 0.55078125, "se": 0.03163726796219167, "rescues": 142, "losses": 1, "by_source": {"gsm8k": {"original": 0.1015625, "edited": 0.75, "difference": 0.6484375}, "math": {"original": 0.140625, "edited": 0.59375, "difference": 0.453125}}}. Four passes remain, all reserved for final3; weights unchanged.

Frozen final_3_batch1 gen66: {"items": 64, "original_accuracy": 0.140625, "edited_accuracy": 0.625, "gain": 0.484375, "se": 0.06296331249416676, "rescues": 31, "losses": 0, "by_source": {"gsm8k": {"original": 0.0625, "edited": 0.75, "difference": 0.6875}, "math": {"original": 0.21875, "edited": 0.5, "difference": 0.28125}}}
Pooled confirmed: {"items": 960, "original_accuracy": 0.09791666666666667, "edited_accuracy": 0.603125, "gain": 0.5052083333333334, "se": 0.016411871850598943, "rescues": 489, "losses": 4, "by_source": {"gsm8k": {"original": 0.10208333333333333, "edited": 0.7104166666666667, "difference": 0.6083333333333333}, "math": {"original": 0.09375, "edited": 0.49583333333333335, "difference": 0.40208333333333335}}}. No weight changes.

Frozen final_3_batch2 gen67: {"items": 64, "original_accuracy": 0.078125, "edited_accuracy": 0.609375, "gain": 0.53125, "se": 0.06669921080659218, "rescues": 35, "losses": 1, "by_source": {"gsm8k": {"original": 0.125, "edited": 0.625, "difference": 0.5}, "math": {"original": 0.03125, "edited": 0.59375, "difference": 0.5625}}}
Pooled confirmed: {"items": 1024, "original_accuracy": 0.0966796875, "edited_accuracy": 0.603515625, "gain": 0.5068359375, "se": 0.015933601686876304, "rescues": 524, "losses": 5, "by_source": {"gsm8k": {"original": 0.103515625, "edited": 0.705078125, "difference": 0.6015625}, "math": {"original": 0.08984375, "edited": 0.501953125, "difference": 0.412109375}}}. No weight changes.

Registered gen68 once; pooled: {"items": 1088, "original_accuracy": 0.09834558823529412, "edited_accuracy": 0.6029411764705882, "gain": 0.5045955882352942, "se": 0.01544107882887705, "rescues": 554, "losses": 5, "by_source": {"gsm8k": {"original": 0.10845588235294118, "edited": 0.7003676470588235, "difference": 0.5919117647058824}, "math": {"original": 0.08823529411764706, "edited": 0.5055147058823529, "difference": 0.4172794117647059}}}. Weights unchanged.

Kernel restarted after pass49; recovered state from persisted JSON files. Repeated gen68 summaries counted once. Plan and both frozen weight hashes verified. One pass remains: final_3_batch4. No changes to submitted weights.

Pre-final submission audit at pass49: exact frozen hashes verified again; final_choice.json records KEEP top64alpha6. submission_evidence.json and confirmation_through_pass49.csv contain 1088-problem evidence; final64 still unseen. Behavior: {"original": {"n": 1088, "correct": 107, "cut": 988, "whitespace_tail_cut": 0, "failure_cut": 975, "correct_thinking_closed": 94, "correct_close_after_1000_chars": 94, "failure_wrong_answer_finished": 6, "median_correct_tokens": 5536.0, "mean_cut_tail_duplicate8": 0.9569762572217041, "mean_nonwhite_cut_tail_duplicate8": 0.9569762572217041}, "edited": {"n": 1088, "correct": 656, "cut": 201, "whitespace_tail_cut": 50, "correct_thinking_closed": 656, "correct_close_after_1000_chars": 656, "failure_wrong_answer_finished": 179, "failure_cut": 200, "failure_no_answer_finished": 53, "median_correct_tokens": 9392.0, "mean_cut_tail_duplicate8": 0.38229341776054565, "mean_nonwhite_cut_tail_duplicate8": 0.4711324071335768}}
No further weight changes planned. Final request is frozen_eval_plan[-1]['request'].

Registered gen69 once; pooled: {"items": 1152, "original_accuracy": 0.10590277777777778, "edited_accuracy": 0.6041666666666666, "gain": 0.4982638888888889, "se": 0.014991383029525228, "rescues": 579, "losses": 5, "by_source": {"gsm8k": {"original": 0.11631944444444445, "edited": 0.7083333333333334, "difference": 0.5920138888888888}, "math": {"original": 0.0954861111111111, "edited": 0.5, "difference": 0.4045138888888889}}}. Weights unchanged.
