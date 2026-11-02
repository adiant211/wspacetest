## Observations

These are **weight-based hypotheses, not measured attention patterns**.

- **L0H3 looks like an offset-routing head.** Its \(W_q^\top W_k/8\) has stable rank **14.93**, unlike the nearly rank-one L0H1/H2/H4/H5. After incorporating LN scales, it has substantial complex eigenvalue pairs, including **\(0.1366\pm0.1327i\)**. Its dominant QK and OV-input subspaces are almost orthogonal: top-eight overlap **0.010–0.011**. This suggests position-based selection carrying character information.
- **L0H0 also looks positional:** approximately **79%/82%** of its scaled QK query/key energy lies in the position-like subspace identified from L0H3.
- **L3H3 has a distinctive induction-compatible connection to L0H3.** For their scaled QK–OV composition, restricted to L0H3’s leading read directions, trace/Frobenius norm is **2.730**, versus **−0.597 to 0.551** for the other L3 heads. Its symmetric part has positive/negative eigenvalue sums **0.4513/−0.0362**. This supports matching information previously transported by L0H3.
- Several heads share strong, low-dimensional routing machinery. For example, `ln1_b.1[39]=-1.5676`, `[58]=1.3651`, `[99]=-1.3355`; L1 query-bias-induced key directions have pairwise cosines **0.823–0.873**. These establish shared routing features, **not their attention distances**.
- The relatively concentrated early value spectra weakly favor a restricted, unevenly used alphabet. However, clustering did **not** recover convincing labeled characters or a clear vowel–consonant structure.

## Heads

Here, \(A\) denotes the LN-scale-adjusted QK matrix; quoted stable ranks without that qualification refer to raw QK.

L0H0: **local** — tentative; strong positional-subspace overlap and a multidimensional value channel suggest nearby-character routing.  
L0H1: **unclear** — QK stable rank 1.28; OV closely resembles H2, but no distance is established.  
L0H2: **unclear** — QK stable rank 1.36; OV cosine with H1 is 0.801, indicating shared low-dimensional machinery.  
L0H3: **previous** — strongest candidate; position-like rotational QK structure separated from its character-carrying value directions. Exact offset remains inferred.  
L0H4: **unclear** — QK stable rank 1.56; shared low-rank output directions do not distinguish local from distant routing.  
L0H5: **unclear** — weakest L0 QK norm, 0.210, but strongest OV norm, 29.436; consistent with coarse aggregation or a reference-position mechanism.

L1H0: **unclear** — strong shared key preference; bias-derived key coefficient norm 3.975, without an identified lag.  
L1H1: **unclear** — corresponding norm 3.689; no convincing predecessor-matching composition.  
L1H2: **unclear** — corresponding norm 3.051; predominantly shared routing structure.  
L1H3: **unclear** — corresponding norm 3.390; positional range cannot be recovered from this alone.  
L1H4: **unclear** — corresponding norm 2.328; particularly weak direct matching composition with L0H3.  
L1H5: **unclear** — corresponding norm 3.895; shares the common key preference, but its range is unresolved.

L2H0: **unclear** — scaled QK stable rank 5.61; strong shared key-bias component, not a demonstrated lag.  
L2H1: **unclear** — scaled QK stable rank 13.91; negligible direct predecessor-matching score.  
L2H2: **unclear** — scaled QK stable rank 11.02; mixed content/routing structure without clear induction evidence.  
L2H3: **unclear** — scaled QK stable rank 10.45; weaker alignment with the common key preference.  
L2H4: **unclear** — scaled QK stable rank 7.43; some positive predecessor composition, insufficient for a copy label.  
L2H5: **unclear** — scaled QK stable rank 13.54; negligible direct predecessor-matching score.

L3H0: **unclear** — predecessor-composition trace/norm −0.597; not the positive matching signature seen in H3.  
L3H1: **unclear** — corresponding statistic −0.090; attention range remains unidentified.  
L3H2: **unclear** — corresponding statistic 0.184; weak evidence for direct induction.  
L3H3: **copy** — strongest induction candidate; distinctive positive matching composition with the likely previous-character head.  
L3H4: **unclear** — weak QK alongside strong OV suggests aggregation, but does not establish broad attention.  
L3H5: **unclear** — predecessor-composition statistic −0.152; neither a clear copy circuit nor an identified positional offset.

## Reasoning

The likely previous-character circuit and its later induction-compatible reader favor **ordered text** over line-shuffled text. They suggest short-range character processing plus retrieval useful for repeating earlier sequences.

**Romaji is my weaker corpus preference:** the concentrated early character-information channels are compatible with its uneven alphabet usage and strong local spelling constraints. English remains quite plausible; nothing recovered identifies Japanese syllables or excludes Python decisively. The corpus probabilities therefore reflect a tentative judgment, not a calibrated classifier.

corpus: romaji  
order: as is  
probabilities: python=0.14 romaji=0.55 english=0.31; as is=0.97 shuffled=0.03