## Observations

All results below are **weight-only evidence, not measured attention patterns**. Exact attention distances remain uncertain without the embeddings.

Let \(A_h\) denote the head’s \(W_q^\top W_k/8\) product, incorporating `ln1_w` and mean-centering, and \(C_h\) its similarly adjusted \(W_oW_v\) product. Define the composition score
\[
R(t,s)=\frac{\operatorname{tr}(A_tC_s)}{\|A_t\|_F\|C_s\|_F}.
\]

Three findings stand out:

- **Offset-like early heads:** L0H1 and L0H5 have strongly overlapping Q/K subspaces but relatively separate value subspaces. Moreover, \(\cos(A_{01},A_{05}^2)=0.546\), versus \(\cos(A_{01},A_{05})=0.414\). This suggests related positional offsets—plausibly two-back and one-back.
- **A broad aggregation head:** L0H4 has \(\|A\|_F=0.0093\), compared with L0H5’s \(0.9704\). Its \(C\) product is large, \(\|C\|_F=17.158\), with **99.45%** of squared singular-value mass in four directions. It appears to aggregate a few shared features rather than select individual characters sharply.
- **Strong induction-compatible compositions:** \(R(\mathrm{L2H1},\mathrm{L0H5})=0.0776\), \(R(\mathrm{L3H1},\mathrm{L2H2})=0.1104\), and \(R(\mathrm{L3H5},\mathrm{L2H2})=0.1836\). Coordinate-permutation controls had standard deviations around \(0.0026\). These are substantial learned alignments, although they do not alone prove behavioral induction.

## Heads

Here, “shared bias alignment” means cosine similarity to the common L1 key-bias direction. Its interpretation as recency is tentative.

L0H0: **unclear** — Low-rank summary: four singular directions contain 97.83% of \(C\)’s energy; distance is unidentified.  
L0H1: **local** — Offset-like Q/K structure; stronger agreement with \(A_{05}^2\) suggests a somewhat longer offset than L0H5.  
L0H2: **unclear** — Low-rank QK structure, stable rank 1.75, but no decisive distance or induction signature.  
L0H3: **unclear** — QK stable rank 2.12; specialized routing is evident, but its positional range is not.  
L0H4: **broad** — Exceptionally small QK product, \(0.0093\), paired with strong, low-dimensional value aggregation.  
L0H5: **previous** — Best one-back candidate: offset-like positional structure and strong downstream induction-compatible compositions.

L1H0: **local** — Shared bias alignment 0.879; likely a weaker member of the position-biased L1 family.  
L1H1: **local** — Strong key-bias norm 3.150 and shared alignment 0.965; likely strongly recency-biased.  
L1H2: **local** — Key-bias norm 2.200, shared alignment 0.954; similar routing machinery.  
L1H3: **local** — Key-bias norm 2.719, shared alignment 0.963; also contributes to later matching paths.  
L1H4: **local** — Key-bias norm 2.518, shared alignment 0.961; closely resembles the other local candidates.  
L1H5: **local** — Key-bias norm 2.460, shared alignment 0.937; exact lag remains unresolved.

L2H0: **local** — Moderate shared bias alignment, 0.664, without a strong induction composition.  
L2H1: **copy** — Clear positive composition with proposed previous-character head L0H5: \(R=0.0776\).  
L2H2: **local** — Strong shared bias alignment, 0.904; especially important as a source for later copy-like matching.  
L2H3: **local** — Shared bias alignment 0.872; also supplies a positive matching path to L3H5.  
L2H4: **local** — Shared bias alignment 0.811 and substantial position-summary coupling.  
L2H5: **unclear** — Weaker positional alignment, 0.523; insufficient evidence to distinguish local from broad retrieval.

L3H0: **unclear** — Distributed QK structure, stable rank 26.26; neither distance nor induction is clearly established.  
L3H1: **copy** — Positive compositions through both L0H5, \(R=0.0610\), and L2H2, \(R=0.1104\).  
L3H2: **unclear** — Some shared positional bias remains, but no compelling copy signature or identifiable range.  
L3H3: **unclear** — Large value-writing product, \(\|C\|_F=37.938\), does not establish where it retrieves information.  
L3H4: **unclear** — Mixed routing evidence; neither broadness nor a particular local offset is securely supported.  
L3H5: **copy** — Strongest induction candidate: \(R=0.1836\) through L2H2, plus \(0.0713\) through L0H5.

## Reasoning

The offset-like early machinery and multiple strong matching compositions favor **preserved character order** over independently shuffled lines. The likely organization is short-context extraction, shared positional summaries, and later retrieval of matching earlier contexts.

**Python is my slight corpus favorite:** substantial multi-stage copying fits repeated identifiers and recurring code fragments. English remains a close alternative. Romaji is less favored by the comparatively rich character-processing structure, but that is weak evidence—not a recovered alphabet or phonotactic analysis.

The corpus judgment is substantially less secure than the order judgment; no specific letters, keywords, or indentation rules were identified.

corpus: python  
order: as is  
probabilities: python=0.52 romaji=0.08 english=0.40; as is=0.99 shuffled=0.01