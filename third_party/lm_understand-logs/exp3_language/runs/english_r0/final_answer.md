## Observations

These are **weight-based hypotheses, not measured attention patterns**. Missing embeddings prevent assigning character identities or establishing exact attention distances.

- **A distinctive positional transport head:** L0H2 has Q/K norms **30.818/28.300**, versus roughly **3–7** for other L0 heads. Its \(Q^\top K/8\) has leading singular values **7.485, 7.160, 6.948** and substantial complex eigenvalue pairs. This suggests a directional positional mechanism, plausibly previous-character transport.
- **Shared positional-summary coordinates:** L0H0/H1/H4/H5 write **68.6%/51.9%/57.4%/41.4%** of their gain-weighted OV energy into coordinate **222**. Their corresponding read directions strongly overlap L0H2’s Q/K subspace. These coordinates receive distinctive treatment: `ln2_w.0[222]=0.05277`, whereas `ln1_w.1[222]=1.12390` and `ln1_b.1[222]=-1.47125`.
- **A strong candidate induction circuit:** L3H1’s QK product composed with L0H2’s OV product has positive, identity-like structure. Restricted to the leading 28-dimensional L0H2 value-read subspace, **trace/nuclear-norm = 0.831**, or **0.767** including learned normalization gains. Other L3 heads are near zero by comparison. This supports matching a current feature against a feature transported from an earlier position.
- **Substantial representational breadth:** `W_proj.3` has 27th/28th singular values **7.461/7.166**, versus **3.139** at the 32nd. This mildly supports a broadly used alphabet, but does not identify the corpus.

## Heads

Below, **register fraction** means the fraction of the algebraic key-bias vector’s squared norm in coordinates **72, 78, 222**. These are candidate positional registers; fractions do not directly measure attention distance.

L0H0: **broad** — weak, nearly rank-one QK; position-summary writing concentrated at coordinate 222.  
L0H1: **broad** — smallest gain-weighted QK norm, **0.030**; shared positional-summary outputs.  
L0H2: **previous** — unusually strong, multidimensional directional QK; transport composes specifically with L3H1. Exact offset remains inferred.  
L0H3: **unclear** — strongly overlaps the positional subspace, but its spectrum does not establish a particular distance.  
L0H4: **broad** — low-rank QK and positional-summary writing; **57.4%** OV energy at coordinate 222.  
L0H5: **broad** — gain-weighted QK stable rank **1.04**; writes principally to shared summary coordinates.

L1H0: **local** — tentative recency mechanism; register fraction **92.5%**, key-bias norm **1.357**.  
L1H1: **local** — same mechanism; register fraction **94.4%**, key-bias norm **2.285**.  
L1H2: **local** — strong shared positional read; register fraction **95.2%**, key-bias norm **2.588**.  
L1H3: **local** — register fraction **94.2%**; also supplies features aligned with L3H1’s matching circuit.  
L1H4: **local** — **70.7%** of QK energy lies in the three key-register columns; bias register fraction **94.5%**.  
L1H5: **local** — register fraction **91.9%**; key-bias direction closely matches other L1 heads.

L2H0: **local** — tentative content-sensitive recency; register fraction **89.4%**, key-bias norm **1.699**.  
L2H1: **unclear** — weaker positional bias and substantial content structure; strong composition with L3H1 does not establish its own attention distance.  
L2H2: **local** — register fraction **84.9%**; relatively strong key-register QK contribution, **21.0%**.  
L2H3: **local** — register fraction **91.2%**, retaining the shared recency mechanism alongside higher-rank content reads.  
L2H4: **local** — register fraction **90.5%**, key-bias norm **1.577**.  
L2H5: **local** — register fraction **88.7%**, key-bias norm **1.509**.

L3H0: **unclear** — high-rank content matching, weak register dependence, and no convincing earlier-head induction composition.  
L3H1: **copy** — uniquely strong positive composition with L0H2; additional matching alignment with L1H2/H3 and L2H1.  
L3H2: **unclear** — mixed positional/content reads; positional bias exists but does not establish locality.  
L3H3: **unclear** — similar mixed structure; no convincing copy signature.  
L3H4: **unclear** — moderate register dependence and content-sensitive QK; attention range unresolved.  
L3H5: **broad** — tentative: weak positional-register dependence, relatively weak QK, and strong value/output transport suggest broader aggregation.

## Reasoning

The combination of directional transport, extensive positional/context processing, and a specific induction-like composition strongly favors **unshuffled text**. Shuffling would largely remove the useful successor relationships that this circuit appears organized to exploit.

Corpus identification is substantially weaker. The broad character-related subspace mildly disfavors romaji’s more restricted usual letter repertoire. Both English and Python remain plausible; these analyses reveal no clear code-specific signature. My best judgment is **English**, with considerably less confidence than the order judgment.

corpus: english  
order: as is  
probabilities: python=0.30 romaji=0.12 english=0.58; as is=0.98 shuffled=0.02