## Observations:

All measurements below are **weight-only**, not forward passes. Head labels are hypotheses; attention distances cannot be verified without the missing embeddings and activations.

For compactness, define
\[
A=G W_q^\top W_kG/8,\qquad B=W_oW_vG,\qquad
r=G W_k^\top W_q\,\mathrm{ln1_b}/8,
\]
where \(G=\operatorname{diag}(\mathrm{ln1_w})\), using each head’s slices. Thus \(A\) is its bilinear score matrix, \(B\) its value/output product, and \(r\) a key-dependent affine coefficient.

- **L0H3 stands out as a directional positional head.** Its \(W_q,W_k\) norms are **14.141, 14.661**, substantially above the other L0 heads. Its unscaled QK product has complex eigenvalues including **\(0.724\pm0.691i\)** and **\(0.446\pm0.842i\)**. The leading 16-dimensional Q/K subspaces overlap strongly (**0.620**), whereas Q/V overlap is only **0.0177**. This supports position-controlled transport of different, likely character-related information.
- **Several L0 heads construct shared low-dimensional signals.** Coordinates **21,39,58,99,184** contain **94.0%, 94.2%, 87.8%, 97.0%** of \(B\)’s squared norm for H1,H2,H4,H5. These coordinates have no intrinsic semantic labels.
- **Later heads strongly read those shared signals.** For example, `ln1_b.1[39]=-1.5676`, `[58]=1.3651`, and `[99]=-1.3355`. All six L1 heads have positive \(r_{21}\), ranging from **2.008 to 3.627**. This is consistent with a shared positional/recency mechanism, although it does not establish its meaning conclusively.
- **L3H3 has a distinctive induction-compatible connection to L0H3.** Let \(T\) contain the leading 20 right-singular directions of L0H3’s scaled value matrix. For
  \[
  C=T^\top A_{3,3}B_{0,3}T,
  \]
  \(\operatorname{tr}(C)/\|C\|_F=\mathbf{2.860}\). Signed/permuted-coordinate controls had standard deviation **0.234** and maximum **0.682** across 1,000 trials. This is strong evidence of learned matching through the earlier transport circuit.
- **The corpus-specific evidence is weaker.** The leading 24 singular directions of `W_proj.3` contain **92.14%** of its squared norm; directions 25–28 add only **1.62%**. This is compatible with a comparatively restricted frequently used alphabet, but is **not** a recovered character count.

## Heads:

The “local” assignments below largely depend on the tentative recency interpretation of the shared channel.

L0H0: **local** — Its positional-subspace score matrix resembles a higher power of L0H3’s directional factor more than the first power; compatible with a longer short offset.

L0H1: **unclear** — Almost rank-two score/output structure; 94.0% of output-product energy enters the shared coordinates. Boundary selection and positional bookkeeping remain alternatives.

L0H2: **unclear** — One score singular mode carries 90.8% of energy; 94.2% of output energy enters the shared coordinates. Distance is unresolved.

L0H3: **previous** — Strongest predecessor candidate: high-rank directional QK structure, separated positional and value subspaces, and the downstream induction-compatible connection.

L0H4: **unclear** — Dominant rank-one score component and 87.8% shared-coordinate output concentration; insufficient evidence for a particular attention distance.

L0H5: **broad** — Very weak score product, with leading singular value **0.032**, but strong low-rank value transport (**16.282, 8.786**); consistent with context aggregation.

L1H0: **local** — Strong shared-channel preference: \(r_{21}=3.627\); key column 21 contains 30.1% of score-product energy.

L1H1: **local** — \(r_{21}=3.352\), with 32.5% key-column-21 energy; closely resembles the other L1 recency candidates.

L1H2: **local** — \(r_{21}=2.769\), with 36.4% key-column-21 energy.

L1H3: **local** — \(r_{21}=3.079\), with 36.4% key-column-21 energy.

L1H4: **local** — \(r_{21}=2.008\), with 34.8% key-column-21 energy; weaker shared preference than most L1 heads.

L1H5: **local** — \(r_{21}=3.563\), with 31.9% key-column-21 energy.

L2H0: **local** — Retains the shared mechanism strongly: \(r_{21}=1.815\), with 15.4% key-column-21 energy.

L2H1: **local** — \(r_{21}=1.682\); substantial shared-channel preference, but more distributed content-sensitive score structure.

L2H2: **local** — \(r_{21}=1.411\); compatible with content-conditioned local retrieval.

L2H3: **local** — Lower-confidence assignment: the shared preference remains, \(r_{21}=0.803\), but is weaker.

L2H4: **local** — Strongest L2 shared-channel preference, \(r_{21}=2.069\); supports recency-sensitive retrieval.

L2H5: **local** — \(r_{21}=1.859\), without a distinctive positive L0H3 matching composition.

L3H0: **local** — Residual shared-channel preference, \(r_{21}=0.408\); the tested predecessor-matching composition is negative, unlike L3H3.

L3H1: **unclear** — Weaker shared-channel preference, \(r_{21}=0.181\), and no strong tested induction signature; local versus broader retrieval remains unresolved.

L3H2: **local** — \(r_{21}=0.382\); retains the common positional mechanism without L3H3’s matching signature.

L3H3: **copy** — Strong positive matching through L0H3: projected composition trace/norm **2.860**, far beyond the coordinate-randomization controls.

L3H4: **broad** — Little shared-channel preference, \(r_{21}=0.028\), combined with large value/output singular values **21.271, 17.680**; favors an aggregation interpretation.

L3H5: **local** — \(r_{21}=0.416\), with no comparable positive predecessor-matching signature.

## Reasoning:

**Intact order is strongly favored.** A directional character-transport circuit followed by an induction-compatible matching circuit is useful for ordered sequences and repetition. Randomly shuffling each line largely removes those predictive relationships, although it preserves line-level composition and boundaries.

**Romaji is my best, less-confident corpus judgment.** The comparatively concentrated character-related geometry and substantial short-context machinery fit a restricted, regular romanized orthography. English remains a serious alternative; the spectra do not identify letters, vowels, words, or language directly. I find less positive evidence specifically favoring Python.

corpus: romaji  
order: as is  
probabilities: python=0.11 romaji=0.62 english=0.27; as is=0.99 shuffled=0.01