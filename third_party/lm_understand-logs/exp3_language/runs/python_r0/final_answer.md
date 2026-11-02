## Observations

These are **weight-based hypotheses, not measured attention patterns**. Missing embeddings make corpus identification and exact attention distances uncertain.

Define \(M=PDW_q^\top W_kDP/8\) and \(B=PW_oW_vDP\) for each head, where \(D=\operatorname{diag}(\texttt{ln1_w})\) and \(P\) subtracts the coordinate mean. These are linear weight products, not layer evaluations.

- **A broad aggregation head:** L0H4 has \(\|M\|_F=0.0093\), versus **0.8775** for L0H1 and **0.9704** for L0H5. Its \(B\) has dominant singular values **13.877, 9.644**, suggesting weakly selective aggregation into a small feature space.
- **Two likely positional character-routing heads:** L0H1 and L0H5 read largely outside their shared value-input subspace. Their generalized-eigenvalue phases support an approximately **2:1 shift relationship**—consistent with two-characters-back and previous-character heads, respectively. The absolute offsets remain inferred.
- **Directional, induction-like composition:** Let
  \[
  C_k=\frac{\operatorname{tr}(M_{\rm later}B_{\rm earlier})}
  {\|M_{\rm later}\|_F\|B_{\rm earlier}\|_F},
  \qquad
  C_q=\frac{\operatorname{tr}(M_{\rm later}B_{\rm earlier}^{\top})}
  {\|M_{\rm later}\|_F\|B_{\rm earlier}\|_F}.
  \]
  The strongest relevant pairs are:
  - L0H5 → L2H1: **\(C_k=0.0776,\ C_q=-0.0398\)**
  - L2H2 → L3H1: **\(0.1104,\ -0.0164\)**
  - L2H2 → L3H5: **\(0.1836,\ -0.0354\)**

  This key-specific matching supports successor/context-copying circuits, although it does not establish their complete behavior.
- **Shared routing in layer 1:** The leading three output directions of L0H4 account for **53–68%** of key-side QK energy in L1H1–L1H5. Their query-bias-induced key preferences also align strongly. This suggests shared positional routing, but does not independently establish locality.

## Heads

Here, \(r_{90}\) is the number of singular values accounting for 90% of \(M\)'s squared norm.

L0H0: **unclear** — \(r_{90}=2\); low-dimensional selector, but no identifiable distance.  
L0H1: **local** — phase relationship with L0H5 suggests a two-character shift.  
L0H2: **unclear** — \(r_{90}=5\); mixed positional and content-selective weight directions.  
L0H3: **unclear** — \(r_{90}=3\); structured selector without a reliable lag signature.  
L0H4: **broad** — exceptionally weak QK product and strong, low-dimensional OV aggregation.  
L0H5: **previous** — strongest one-character-shift candidate; also supplies the L2H1 matching circuit.

L1H0: **unclear** — weaker shared-routing component; distance and selectivity remain ambiguous.  
L1H1: **local** — tentative; 63% key-side energy reads L0H4’s leading output subspace.  
L1H2: **local** — tentative; corresponding fraction 63%, with substantial downstream matching composition.  
L1H3: **local** — tentative; strongest shared-routing fraction, 68%.  
L1H4: **local** — tentative; 65% shared-routing fraction and aligned key preference.  
L1H5: **local** — tentative; 53% shared-routing fraction; exact offset unresolved.

L2H0: **unclear** — distributed QK structure, \(r_{90}=39\), without a decisive routing signature.  
L2H1: **copy** — positive, key-specific composition with the likely previous-character head L0H5.  
L2H2: **unclear** — important precursor to both late copy candidates; its own attention distance is unverified.  
L2H3: **unclear** — some shared positional-carrier dependence, insufficient to distinguish local from broad.  
L2H4: **unclear** — mixed routing structure; no decisive predecessor or induction signature.  
L2H5: **unclear** — distributed QK structure, \(r_{90}=42\).

L3H0: **unclear** — modest composition signals, not enough to establish copying or locality.  
L3H1: **copy** — strong directional composition with L2H2: \(C_k=0.1104\).  
L3H2: **unclear** — distributed selector without a clear distance or copying signature.  
L3H3: **unclear** — weaker QK magnitude alone cannot establish broad attention.  
L3H4: **unclear** — no sufficiently diagnostic composition.  
L3H5: **copy** — strongest induction-like candidate: \(C_k=0.1836\), versus \(C_q=-0.0354\).

## Reasoning

The likely short-offset character routing and directional copying circuits favor **as-is text** substantially over independently shuffled lines.

I weakly favor **Python**: repeated identifiers and recurring code fragments offer a particularly useful role for this hierarchy of local features and context matching. English is a substantial alternative; these circuits are not code-specific. The corpus judgment is considerably weaker than the order judgment, and no character identities or keywords were recovered.

corpus: python  
order: as is  
probabilities: python=0.56 romaji=0.13 english=0.31; as is=0.96 shuffled=0.04