# FINAL SUBMISSION — all 50 forward passes complete
KEEP the frozen top64 alpha6 edit. Only:
model.language_model.layers.15.mlp.down_proj.weight
model.language_model.layers.19.mlp.down_proj.weight

Formula: W = W0 - 6*outer(u, mask64*(u@W0)).
Unit repetition directions from four matched productive/repetitive original contexts
at residual outputs16/20;64 selected MLP writer neurons per layer, chosen on original fwd4.
131072 changed values total; same names/shapes. No norm/embedding/head edits or training.
Relative matrix changes10.6357%/8.1058%; hashes unchanged throughout confirmation.
Selected neuron IDs and hashes: frozen_weights_manifest.json.
Identical final backup: final_verified_edit/. Actual submission remains /work/edited.

PATH AND CAUSAL EVIDENCE
Initial11/96 correct,85 cutoff,~95% duplicate tail phrases. Targeted repetition, not math facts.
Full directional doses alpha1/2/4 selectively reduced repetitive branch logp
by .02410/.04870/.10372 nats/token; productive branches barely changed.
L15-only and L19-only interventions supported both sites; attention-head alternatives removed.
Norm-matched orthogonal random control gained1.56pp SE3.52, not the targeted intervention's gain.
Top64 alpha6 transfer: repetitive-.10311 versus productive-.00903 nats/token.
Broader productive/numeric controls changed little; no consistent premature closure.
Fullalpha1/2 generation dose tests showed increased rescue; sparse top64 retained concentrated
repetition writing while changing fewer weights. Frozen before any confirmation below.
Remaining errors include wrong math, missing boxes, whitespace degeneration and truncation.

FINAL CONFIRMATION
All1152 unique problems were independent of edit selection. Paired within run,
both models, n1, max32768, fixed thinking-enabled evaluation sampling.
Repeated gen68 summaries counted once; kernel restart did not change saved hashes.
All five independent confirmation splits and all individual batch results retained.
These are held-out PRACTICE results, not unseen benchmark scores.

POOLED
{
  "items": 1152,
  "original_accuracy": 0.10590277777777778,
  "edited_accuracy": 0.6041666666666666,
  "gain": 0.4982638888888889,
  "se": 0.014991383029525228,
  "rescues": 579,
  "losses": 5,
  "by_source": {
    "gsm8k": {
      "original": 0.11631944444444445,
      "edited": 0.7083333333333334,
      "difference": 0.5920138888888888
    },
    "math": {
      "original": 0.0954861111111111,
      "edited": 0.5,
      "difference": 0.4045138888888889
    }
  },
  "gain_95pct_normal_interval": [
    0.4688807781510195,
    0.5276469996267583
  ]
}
SOURCE COUNTS
{
  "gsm8k": {
    "n": 576,
    "original_correct": 67,
    "edited_correct": 408,
    "gain": 0.5920138888888888,
    "paired_se": 0.021219128177913046
  },
  "math": {
    "n": 576,
    "original_correct": 55,
    "edited_correct": 288,
    "gain": 0.4045138888888889,
    "paired_se": 0.020467679827936427
  }
}
BEHAVIOR
{
  "original": {
    "n": 1152,
    "correct": 122,
    "cut": 1038,
    "whitespace_tail_cut": 0,
    "failure_cut": 1023,
    "correct_thinking_closed": 107,
    "correct_close_after_1000_chars": 107,
    "failure_wrong_answer_finished": 6,
    "failure_no_answer_finished": 1,
    "median_correct_tokens": 5310.5,
    "mean_cut_tail_duplicate8": 0.9580040916208376,
    "mean_nonwhite_cut_tail_duplicate8": 0.9580040916208376
  },
  "edited": {
    "n": 1152,
    "correct": 696,
    "cut": 212,
    "whitespace_tail_cut": 51,
    "correct_thinking_closed": 696,
    "correct_close_after_1000_chars": 696,
    "failure_wrong_answer_finished": 187,
    "failure_cut": 211,
    "failure_no_answer_finished": 58,
    "median_correct_tokens": 9171.5,
    "mean_cut_tail_duplicate8": 0.4022594815366762,
    "mean_nonwhite_cut_tail_duplicate8": 0.49324499100890384
  }
}
Files: submission_evidence.json, final_confirmation.csv, final_choice.json, frozen_eval_progress.json, frozen_weights_manifest.json, experiment_history.md.
