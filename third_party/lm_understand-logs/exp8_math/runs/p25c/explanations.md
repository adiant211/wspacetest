## Turn 15 (forward; passes used 2/25)

(no explanation)

## Turn 15 (forward; passes used 2/25)

(no explanation)

## Turn 16 (python; passes used 3/25)

Removing layer 23 neuron 22 did not improve accuracy: the paired change was −0.8 percentage points with a 3.6-point standard error. Its activation is a useful marker of repetition, but this test does not establish it as the main cause. I will undo that edit and examine whether several components jointly reinforce a shared repetition direction, using identical phrases at their first and later occurrences. A convincing direction should distinguish repeated phrases from productive reasoning across problems, rather than merely identify long answers.

## Turn 17 (python; passes used 3/25)

(no explanation)

## Turn 18 (python; passes used 3/25)

(no explanation)

## Turn 19 (python; passes used 3/25)

The single-neuron ablation did not help: −0.8 percentage points, with a 3.4-point standard error. I’ll reject it and test whether repetition is sustained by a distributed circuit rather than one neuron.

## Turn 20 (python; passes used 3/25)

(no explanation)

## Turn 21 (forward; passes used 3/25)

The baseline’s main failure is repetition: 229 of 256 answers hit the token limit, often after already finding the correct arithmetic result. Layer 23 MLP neuron 22 was more active in every one of the 13 matched failure-versus-success pairs, and its activation rose during the controlled repetition probe. This is still correlational, so I have zeroed only that neuron’s output column for a causal test. I will now compare both models on identical problems and seeds; increased accuracy alongside reduced repetitive cutoffs would support the hypothesis, whereas merely shorter or less accurate answers would not.

## Turn 22 (python; passes used 4/25)

(no explanation)

## Turn 23 (forward; passes used 4/25)

The baseline failed mainly by repetition: 229 of 256 answers hit the token limit, often after already finding the correct calculation. Layer 23 MLP neuron 22 was more active in the failing answer in all 13 same-problem success/failure pairs, and its activation increased during the controlled repetition probe. This identifies a candidate, but does not establish causality. I have temporarily zeroed only that neuron’s output column and will compare both models on the same 128 seeded generations. More correct answers together with fewer repetitive cutoffs would support the hypothesis; unchanged accuracy or lost correct answers would weaken it.

## Turn 24 (python; passes used 5/25)

(no explanation)

## Turn 25 (forward; passes used 5/25)

(no explanation)

## Turn 26 (forward; passes used 5/25)

The baseline fails mainly by repeating its reasoning until the token limit, often after already computing the correct answer. Layer 23 MLP neuron 22 is consistently more active in failing than successful answers to the same problems, and its activity rises during a controlled repetition probe. I have zeroed only that neuron’s output column; this is a causal test, not yet an accepted repair. The paired generation below will test whether this change reduces repetition **and improves accuracy** without disrupting successful reasoning.

## Turn 27 (python; passes used 6/25)

(no explanation)

## Turn 28 (forward; passes used 6/25)

(no explanation)

## Turn 29 (forward; passes used 6/25)

(no explanation)

## Turn 29 (forward; passes used 6/25)

(no explanation)

## Turn 29 (python; passes used 6/25)

(no explanation)

## Turn 30 (python; passes used 6/25)

(no explanation)

## Turn 31 (python; passes used 6/25)

The saved files contain more paired trials than the notes describe, but their grades are nested inside each problem’s samples. I will unpack those grades and check the activation-array shapes before interpreting any neuron indices. I will also inspect short repetition excerpts and MLP weight norms to nominate possible drivers, without treating correlation or unusual weights as causal evidence. For future passes, I will record the exact intervention and problem set beforehand and use only within-pass paired comparisons to estimate gains.

## Turn 32 (python; passes used 6/25)

The recovered trials do not establish an improvement: the only positive recorded difference was 2.34 percentage points with a 3.23-point standard error. The examples confirm several forms of repetition, including identical self-checks and irrelevant enumeration, so a single late-layer neuron may only reflect the problem rather than drive it. I will compare same-problem successes and failures with the controlled repetition probe, then examine full-attention heads as another possible source of copying feedback. A useful candidate should distinguish repetitive continuation from productive reasoning in both comparisons, not just have large weights.

## Turn 33 (python; passes used 6/25)

The old teacher-forced probe has a serious confound: one text contains an inserted `</think>` followed by another `<think>` before the reasoning. Its apparent neuron differences therefore cannot cleanly identify repetition, and I will not rely on that evidence. I will replace it with two raw texts sharing exactly the same prompt and initial calculation, then compare distinct mathematical checks with repeated checks. Attention concentrated on previous copies of the repeated passage would nominate a copying head for an intervention, while the matched prefix provides a control.

## Turn 34 (forward; passes used 6/25)

The replacement probe is recorded, uses no edits, and keeps both texts inside the same thinking context until their continuations diverge. I will now collect attention, residual, and MLP measurements for both texts. This is a localization step only: even a strong repetition-specific signal will need a targeted intervention before it can support an edit.

## Turn 35 (python; passes used 7/25)

The clean probe ran successfully, and the repeated continuation has much higher average token likelihood than the productive continuation. That alone is not proof of a defect, because exact repetition is intrinsically predictable. I will now measure which heads retrieve earlier matching passages and which MLP neurons distinguish repetition from valid checking, excluding the closing-thinking tokens. The strongest candidates should show repetition-specific behavior rather than merely attend locally or respond to all arithmetic.

## Turn 36 (python; passes used 7/25)

Layer 15, head 7 is the clearest copying candidate: it assigns 88.7% of attention to continuations of earlier matching passages during the loop, versus 8.2% during productive checks. Layer 11, head 1 and layer 23, head 5 show similar but weaker patterns. I will first remove only head 7’s contribution through its output-projection columns, leaving every other head unchanged. The prediction is a selective reduction in repetitive continuation—not a general push to end thinking—and I will test both token likelihoods and paired problem-solving accuracy.

## Turn 37 (forward; passes used 7/25)

The edit now zeros only the 256 output channels belonging to layer 15’s head 7; no other weights have changed. I will replay the exact clean probe so that any likelihood change is attributable to this intervention rather than a different prompt. Selectively lowering the repeated passage’s likelihood while preserving the shared calculation would support the copying-feedback hypothesis; comparable damage to valid reasoning would argue against keeping the ablation.

## Turn 38 (python; passes used 8/25)

Removing the head slightly lowered the repeated text’s average log probability while leaving the productive text’s average almost unchanged. This is in the predicted direction, but the effect is small and does not establish an accuracy benefit. I will inspect the reasoning-only regions and then run a paired generation comparison on the recorded development problems. The edit will be useful only if it converts failures into correct answers, not merely shorter outputs.

## Turn 39 (forward; passes used 8/25)

The reasoning-only comparison is selective: log probability increased by 0.025 nats per token for the shared calculation and 0.008 for productive checks, but decreased by 0.032 for repeated checks. The intervention also reduced the loop activation of layer 23’s neuron 22 from 3.05 to 1.49, consistent with that neuron being downstream of the copying mechanism. I will now test whether this causal change produces more correct answers on the same problems and seeds within a paired pass. A reduction in repetition without an accuracy gain would not justify retaining it.

## Turn 40 (python; passes used 8/25)

The generation worker failed during engine initialization, so this attempt produced no accuracy evidence and did not consume a pass. I will verify that the saved edit has the expected shape, finite values, and changes confined to the intended head. If those checks pass, I will retry the same paired experiment once; this is an infrastructure retry, not a repeat measurement of a completed trial.

## Turn 41 (forward; passes used 8/25)

The integrity checks passed: the tensor is finite, its shape is unchanged, and only columns 1792–2047 differ from the original. The edit therefore matches the intended single-head ablation. I will retry the recorded paired generation request and judge it by accuracy, with repetition and length serving only as explanatory measurements.

## Turn 42 (python; passes used 8/25)

The paired generation failed a second time at engine startup, so I still have no accuracy evidence for the head edit. I will stop retrying the identical request and check whether a local diagnostic log exposes the cause. Meanwhile, teacher-forced evaluation is working, so I will prepare a half-strength intervention to test dose-response on the same controlled texts. I will not accept any edit solely on those likelihood measurements if paired generation remains unavailable.

## Turn 43 (forward; passes used 8/25)

No accessible startup log was found, and the current edit now retains half of layer 15 head 7’s original output. I will measure it on the same two raw texts used for the original model and full ablation. An intermediate reduction in repeated-token likelihood, with little effect on productive calculations, would strengthen the causal interpretation; an inconsistent response would weaken it.

## Turn 44 (python; passes used 9/25)

The half-strength edit produces an intermediate likelihood change, between the original model and full ablation, which supports a dose-response rather than an arbitrary perturbation. This remains evidence about repetition, not mathematical accuracy. I will summarize the three conditions and try a smaller paired generation batch while retaining the full 32,768-token limit. If generation works, the result will screen the half-strength edit; any final claim will still require larger, untouched problem sets.

## Turn 45 (forward; passes used 9/25)

The measured response is monotonic: established-loop log probability changes from −0.013 at full strength to −0.016 at half strength and −0.030 after ablation. Productive-check likelihood is essentially preserved in both edited conditions. I will now test the half-strength edit against the original on 64 paired problems, with the usual thinking prompt and full token allowance. This smaller batch is a preliminary accuracy check, not enough by itself to establish a modest gain.

## Turn 46 (python; passes used 10/25)

Generation is working again. The half-strength edit improved development accuracy from 9/64 to 14/64, but the paired gain of 7.81 percentage points has a 6.79-point standard error, so it is not convincing yet. I will inspect whether the rescued answers still contain substantive reasoning and then test full ablation on the same development problems with two samples each. A stronger reduction in repetition accompanied by a clearer accuracy gain would support the mechanism; deterioration would favor backing off.

## Turn 47 (forward; passes used 10/25)

The half-strength outputs still contain substantial reasoning: correct answers had a median length of about 7,300 tokens, rather than abruptly skipping thinking. Their mean repeated-line fraction fell from 0.705 to 0.638, although some graded successes still looped after stating an answer. I will now test full ablation with 128 generations per model. Accuracy remains the deciding measure, and I will account for the two samples belonging to the same problem when estimating uncertainty.

## Turn 48 (python; passes used 10/25)

The larger generation request again failed at startup, while the earlier one-sample request completed. That pattern suggests a batch-size or multi-sample issue, although the error does not identify the cause. I will keep the full-ablation weights unchanged and reduce this development request to one sample per problem. If that works, I will build validation power from additional distinct problems rather than repeated samples.

## Turn 49 (forward; passes used 10/25)

The revised request is recorded and still uses full-length thinking and the fixed decoding protocol. Only the number of samples per problem has changed. I will compare the original and fully ablated models within this pass, without using the previous pass’s original-model accuracy as a substitute baseline.

## Turn 50 (python; passes used 11/25)

Full ablation improved development accuracy from 8/64 to 17/64: a paired gain of 14.06 points with a 5.84-point standard error. Together with the likelihood dose-response, that makes layer 15 head 7 a promising candidate, but it still needs untouched-problem validation. I will preserve this candidate and test layer 23 head 5 separately, since its attention also strongly favors repeated continuations. Testing it alone will reveal whether it reinforces the loop or merely attends to it while performing a useful corrective function.

## Turn 51 (forward; passes used 11/25)

The full layer-15 ablation reduced mean repeated-line fraction from 0.736 to 0.542, while correct answers still had a median length of about 9,500 tokens. That is consistent with repairing repetition inside thinking rather than forcing uniformly short answers. The layer-15 candidate is backed up, and the active edit now affects only layer 23 head 5. I will test whether this second head has the same selective causal effect before considering any combination.

## Turn 52 (python; passes used 12/25)

Ablating layer 23 head 5 lowered both texts’ overall likelihoods by similar amounts, so the initial summary does not show the selectivity seen for layer 15 head 7. I will check the reasoning-only regions before drawing that conclusion, and I will not add this head merely because its attention follows repeated passages. The next isolated candidate is layer 11 head 1, which showed strong retrieval of earlier matching continuations. This comparison should help separate useful copying heads from heads that specifically sustain the pathological loop.

## Turn 53 (forward; passes used 12/25)

The region analysis argues against layer 23 head 5: its ablation lowered shared-arithmetic likelihood by 0.049 nats per token but repeated-check likelihood by only 0.004. I have restored that head and will test layer 11 head 1 alone. For this candidate to advance, its effect should be substantially more specific to repetition than to the valid calculation.

## Turn 54 (python; passes used 13/25)

The layer-11 intervention also looks unhelpful in the aggregate: productive-text likelihood fell while repeated-text likelihood rose. I will quantify the reasoning regions, then restore the simpler layer-15-only candidate rather than combining heads with contrary evidence. That candidate is now fixed for an initial held-out test on 128 untouched problems, balanced between GSM8K and MATH. A clear paired accuracy gain here would be much stronger evidence than the development results used to select it.

## Turn 55 (forward; passes used 13/25)

The detailed probe confirms that layer 11 head 1 is not a useful repair: its ablation harms productive calculations without meaningfully suppressing repetition. Only layer 15 head 7 remains edited. I will now evaluate that frozen candidate on 128 new problems spanning both sources and all MATH difficulty levels. The comparison is paired within this pass, and these problems were not used to identify the head or choose its strength.

## Turn 56 (python; passes used 14/25)

The first held-out test improved accuracy from 16/128 to 28/128, a paired gain of 9.38 points with a 3.91-point standard error. Both GSM8K and MATH improved, which is encouraging, but the estimate is still fairly uncertain. I will leave the weights frozen and test the other 128 problems from the preselected held-out set. Pooling two disjoint paired tests will provide a more reliable estimate without selecting a new edit on the validation outcomes.

## Turn 57 (forward; passes used 14/25)

On the first held-out batch, repeated-line fraction fell from 0.687 to 0.592, and completed correct answers increased from 15 to 26. Correct edited answers were not uniformly shortened—their median length was about 11,700 tokens. I will now run the independent replication with the exact same weights. Consistent accuracy improvement, rather than just fewer cutoffs, is the criterion for retaining this repair.

## Turn 58 (python; passes used 15/25)

The second held-out batch was much weaker: 18/128 correct for the original versus 20/128 for the edit, a gain smaller than its standard error. That means the earlier improvement is not yet a robust general result, especially for MATH. I will compute the pooled estimate and test one stronger, predefined intervention on the same identified head: reversing its output at half magnitude. This tests whether actively opposing the copying signal helps more than removing it; any revised candidate will need fresh validation, not reuse these outcomes as confirmation.

## Turn 59 (forward; passes used 15/25)

Pooling the two held-out batches gives a 5.47-point gain with a 2.63-point standard error; the MATH-specific gain is only 2.34 points with a 3.76-point standard error. I therefore regard full ablation as provisional rather than a confirmed strong repair. The active experiment now reverses only the identified copying head at half magnitude. I will first check whether this stronger dose suppresses repeated checks without damaging the shared calculation or productive checks.

## Turn 60 (python; passes used 16/25)

The stronger dose has a much larger effect on the repeated text than on the productive text, which is promising but still only a controlled-probe result. I will inspect the reasoning-only changes and then test this dose on the development set, not on fresh validation problems yet. If it mainly reduces repetition while increasing correct answers, it can advance to a new held-out test. If it disrupts arithmetic or produces shorter but wrong answers, I will reject it.

## Turn 61 (forward; passes used 16/25)

The reasoning-only effect is selective: the reversed head improves shared-calculation likelihood by 0.048 nats per token and leaves productive checks essentially unchanged, while lowering repeated-check likelihood by 0.142. It also reduces the downstream loop-associated neuron 22’s activation from 3.05 to 0.22. I will now test whether that stronger causal effect translates into correct answers under generation. This remains a development trial; no claim of general improvement will rest on it alone.

## Turn 62 (python; passes used 17/25)

The stronger reversal improved development accuracy by 10.94 points, but its 7.08-point standard error is large, and it did not establish an advantage over the simpler ablation. Stronger suppression on one synthetic loop therefore does not justify a more aggressive final edit. I will return to full ablation, freeze that choice, and use the remaining untouched sets for higher-powered confirmation. I will also use separate diagnostic probes to check whether useful multi-step and repetitive mathematical procedures remain intact, without tuning the weights on those results.

## Turn 63 (forward; passes used 17/25)

The final candidate is now fixed at zeroing layer 15 head 7’s output contribution; the stronger reversal is archived but not selected. This next batch is entirely separate from the original diagnostics, development problems, and both earlier validation batches. I will use its paired accuracy difference as fresh confirmation and keep the weights unchanged regardless of individual examples. Further diagnostics will explain the effect, not serve as opportunities to tune against this set.

## Turn 64 (python; passes used 17/25)

The fresh generation request failed at engine startup and produced no results, so the held-out set remains unused. I will move the already planned diagnostic probes ahead of that retry rather than repeatedly launching the failed request. These probes contrast productive calculations with loops across arithmetic, algebra, and modular exponentiation, including a deliberately repetitive procedure that genuinely makes mathematical progress. All texts remain inside thinking, so the measurements will not be driven by closing the thinking section.

## Turn 65 (forward; passes used 17/25)

The broader probe is ready, including a useful repeated-multiplication sequence alongside a loop that repeats only its first step. This distinction matters because removing all repetition would also damage legitimate mathematical reasoning. I will first measure the original model, then replay the identical texts with the frozen edit. The prediction is a larger effect on stalled loops than on calculations that continue to make progress.

## Turn 66 (python; passes used 18/25)

The original-model measurements are available, including a modular-arithmetic example where productive repetition is already highly predictable. That is a useful control against mistaking predictability itself for a failure. I will record reasoning-only and numeric-token scores, then evaluate the identical texts with the frozen head ablation. Preserved numeric calculations together with selectively weakened stalled loops would support the proposed mechanism beyond the original speed problem.

## Turn 67 (forward; passes used 18/25)

The original model assigns high likelihood to correct numeric steps in all three productive examples, including the repeated modular procedure. The looping versions become especially predictable after their first two repetitions. I will now apply the frozen edit to exactly the same texts and compare those regions directly. This test can reveal collateral damage to useful copying even if the edit reduces generation cutoffs.

## Turn 68 (python; passes used 19/25)

The broader probe shows some likelihood reduction on productive reasoning as well as on loops, so the edit is not perfectly selective. I will quantify the numeric-token effects rather than assume that all of the loss is harmless wording variation. The final decision still depends on net mathematical accuracy, and the weights remain frozen. I will now retry the untouched paired generation batch after these diagnostic passes.

## Turn 69 (python; passes used 19/25)

The broader probes show that the edit reduces repeated-step likelihood more than productive-reasoning likelihood, but it is not perfectly selective. This supports the repetition hypothesis without establishing an accuracy gain. I will keep the candidate frozen and resume fresh paired validation. First I will verify the saved weights and recover the exact held-out problem IDs.

## Turn 70 (forward; passes used 19/25)

The weight check confirms that only layer 15, head 7’s output columns are zeroed. Earlier experiments found a dose-dependent reduction in repeated continuations, while other candidate heads did not show the same selective effect. I will now test the unchanged edit on 128 untouched problems; paired accuracy, not merely fewer cutoffs, will determine whether this supports keeping it.

## Turn 71 (python; passes used 19/25)

The generation request failed during engine initialization, so it provides no accuracy evidence and consumes no pass. I will preserve the untouched validation set and run the planned causal controls while generation is unavailable. The next control will ablate a different head in the same layer: if it suppresses loops equally, the claimed specificity of head 7 would weaken.

## Turn 72 (python; passes used 19/25)

The existing probes contain three productive-versus-repetitive pairs, including a productive calculation with repeated formatting. I will reuse those exact texts and choose a control head that shares head 7’s key/value group, has low repetition-retrieval attention, and has a similar output-weight norm. This tests whether the effect comes from the identified head rather than simply removing a comparable amount of attention output. The control is temporary and will not replace the frozen candidate.

## Turn 73 (forward; passes used 19/25)

Head 4 is the control: its repetition-retrieval attention is 16.1%, versus 88.7% for head 7, and its output-weight norm is comparable. Only head 4 is currently ablated; head 7 has been restored for this test. Much weaker or less selective loop suppression would support the localization, while a similar effect would argue against head-7 specificity.

## Turn 74 (python; passes used 20/25)

The control pass completed, but I accidentally changed one number in the productive modular-arithmetic text. That row is not an exact matched comparison and must be excluded; the other five texts can still provide valid controls. I will audit text identity, calculate those comparisons, and then temporarily amplify head 7 to test the opposite predicted effect. Increased repeated-continuation likelihood would strengthen the causal interpretation.

## Turn 75 (forward; passes used 20/25)

Ablating head 4 barely affected the three loops and reduced likelihood on both valid matched productive texts. Head 7’s ablation suppressed those loops substantially more while preserving the productive texts better, supporting component specificity. I will now test head 7 at 1.5 times its original strength; reinforcing loops would provide an opposite-direction causal check.

## Turn 76 (python; passes used 21/25)

The amplification pass completed, so I will compare its reasoning-token likelihoods with the original model before drawing conclusions. I will then restore the frozen head-7 ablation exactly and use all four remaining passes for fresh paired accuracy measurements. The causal probes explain a plausible mechanism, but the edit still needs independent evidence that more problems are answered correctly.

## Turn 77 (forward; passes used 21/25)

Amplifying head 7 increased repeated-continuation likelihood in all three probes, opposite to ablation; the increases were small but directionally consistent. Together with the same-layer control and earlier dose response, this supports a repetition-reinforcement role. The frozen ablation is restored, and I will retry the untouched paired accuracy test without changing its problems or decoding settings.

## Turn 78 (python; passes used 22/25)

The fresh test improved from 18/128 to 34/128 correct: a paired gain of 12.5 percentage points with a 3.84-point standard error. Most of the gain came from grade-school problems; competition-math accuracy was nearly unchanged. This is encouraging independent evidence, but I will finish the preselected validation sets without adjusting the edit.

## Turn 79 (forward; passes used 22/25)

The saved records confirm that paired correctness, token counts, and cutoff status are available for analysis. I will now run the complementary 128-problem holdout, again balanced between GSM8K and MATH practice problems. A positive pooled accuracy difference would support replication; fewer cutoffs alone would not.

## Turn 80 (python; passes used 23/25)

The second fresh batch showed no net accuracy gain: seven problems improved and seven regressed. That makes the evidence less uniform than the first batch suggested, especially for competition math. I will pool the paired results and continue with the two larger, preselected holdouts; the edit remains unchanged.

## Turn 81 (forward; passes used 23/25)

Across the first 256 fresh pairs, accuracy improved by 6.25 points with a 2.44-point standard error. The benefit is currently confined to GSM8K practice problems; the MATH difference is −0.78 points and inconclusive. I will now evaluate the unchanged edit on the next 256 untouched problems to sharpen both estimates.

## Turn 82 (python; passes used 24/25)

The larger holdout improved from 29/256 to 58/256 correct, a paired gain of 11.33 points with a 2.59-point standard error. Unlike the earlier batch, this one improved both grade-school and competition-math accuracy. I will run the final preselected holdout and base the decision on all 768 fresh pairs, including the batch with no gain.

## Turn 83 (forward; passes used 24/25)

The pooled fresh result is now +8.79 accuracy points with a 1.78-point standard error. The GSM8K benefit is clear; the smaller MATH benefit remains less certain. This final 256-problem comparison will complete the planned validation, with no further weight or strength changes.

## Turn 84 (forward; passes used 24/25)

The final request failed before generation began, so no results were observed and one pass remains. This is the same initialization failure that previously resolved on retry. I will repeat the identical request, preserving the held-out set, frozen weights, and full token budget.

## Turn 85 (python; passes used 25/25)

The final batch improved from 28/256 to 47/256 correct, adding another positive replication. Across all 768 fresh pairs, the frozen edit produced 64 additional correct answers. I will keep that edit, calculate the final uncertainty and source-specific results, and verify that no temporary control edits remain. I will also inspect both a rescued answer and a regression so the final report reflects the remaining failure modes.

## Turn 86 (final answer; passes used 25/25)

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

