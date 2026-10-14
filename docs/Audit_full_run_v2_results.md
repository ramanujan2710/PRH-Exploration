# Full audit, all 12 Aristotelian LLMs (engine v3), 11 Oct 2026

Setup: Huh et al. / Aristotelian pipeline (95th-percentile clip + L2, max over all layer pairs, permutation null on the max; confirmed against their code). WIT-1024, 12 LLMs (BLOOMZ 560m–7b1, OpenLLaMA 3/7/13b, LLaMA 7/13/30b bf16 + 65b 4-bit) × 17 ViTs, 200 permutations. LLaMA-65B in 4-bit has bpb 1.661, worse than LLaMA-7B, so trends use parameter size; capability-based slopes are in the earlier 11-LLM analysis.

## 1. We reproduce their headline
Pearson correlation with LLM size rank, averaged per ViT: raw CKA **0.85** (paper 0.86), calibrated CKA **0.43** (paper 0.45), mutual k-NN **0.83** (paper ~0.85). With the same statistic, unbiased CKA with held-out layer = **0.69**.

## 2. Why calibration kills the trend
- τ = √(max_l r_LLM · max_l r_ViT) / n, with r = 1.0000 and MAE 0.002 over 612 pairs × 3 preprocessings. It is set by the LLM's highest-effective-rank layer (BLOOMZ: embedding layer).
- Per doubling of LLM size (family fixed effects, 95% CI over ViTs): τ **+0.0150** [0.0128, 0.0169] vs raw CKA **+0.0091** [0.0076, 0.0106]. The threshold grows faster than the score, so the calibrated trend turns **negative**: −0.0024 [−0.0031, −0.0017].
- Measured without a threshold: unbiased CKA with held-out layer **+0.0043** [0.0037, 0.0049]; mutual k-NN +0.0035 [0.0031, 0.0039]; held-out calibrated +0.0018 [0.0004, 0.0033].

## 3. What is true at the top end (LLaMA 7 → 13 → 30 → 65B)
- Unbiased held-out: 0.336, 0.333, 0.328, 0.327 (−0.0029 per doubling, CI excludes 0). Global alignment plateaus or slightly declines.
- Mutual k-NN: 0.144, 0.143, 0.145, 0.147 (+0.0011 per doubling). Local alignment is nearly flat too.
- Raw CKA rises (0.427 → 0.446) only because the chance offset rises (max layer rank 257 → 418). Calibrated falls (0.296 → 0.275) because τ rises (0.187 → 0.237).

## Claim for Paper A
The Aristotelian calibrated threshold is the chance level of the highest-rank layer, which grows faster with model size than the alignment itself, so it converts a positive global trend into a negative one. Measured with a threshold-free estimator, global alignment rises with scale across the model set (driven by the weaker models) and saturates within the strongest family, where local alignment also nearly saturates. Neither "global converges" nor "only local converges" describes the data; both saturate at the top.

## Open items
- Their Fig. 12 (calibrated ≈ debiased) is single-layer; our mechanism needs the multi-layer max. Cite and distinguish.
- Possible fp32/TF32 issue in their unbiased-CKA runs (unverified).
- Family and scale are confounded; LLaMA-65B is 4-bit.
- Next: E3 Rosetta yardstick; modern LLMs (Llama-3, OLMo, Mistral) to test saturation.
