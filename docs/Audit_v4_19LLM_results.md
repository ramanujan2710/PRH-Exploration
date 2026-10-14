# PRH audit v4: 19 LLMs × 17 ViTs (exact bf16), status 11 Oct 2026

Run: full, WIT-1024, 200 permutations, engine v3. Missing: Llama-3-70B (next session, RUN_ONLY).

## Established (strong)
- Threshold mechanism: the calibration threshold τ (max over all layer pairs) equals the chance level of the highest-effective-rank layer pair. Predicted vs observed r = 0.99999, MAE 0.002 across all 19 LLMs, incl. new families (OLMo, Gemma, Mistral, Llama-3). τ rises with size: 0.13 (BLOOMZ-560m) → 0.237 (LLaMA-65B).
- Replication of the Aristotelian statistic (original 12 LLMs): size-rank r raw 0.87, calibrated 0.47, mKNN 0.85 (paper: .86/.45/.85).
- LLaMA-65B 4-bit vs exact bf16: all metrics within 0.002 (cka_cal 0.2752 vs 0.2767), so earlier results stand.
- E1 closed form for chance-level CKA vs effective rank matches simulation.

## Moderate
- Global alignment trend after fixing the estimator (unbiased CKA, layer pair chosen on held-out half): Pearson vs capability 0.26 (Huh preprocessing), 0.33 (std), 0.64 (no clipping), vs calibrated aggregate 0.32/0.38/0.57.
- Effect sizes are tiny under Huh's clipping (held-out unbiased CKA, weakest vs strongest LLM: 0.269 vs 0.277). Preprocessing (clipping massive-activation dimensions) moves the result more than the estimator does.
- Local mKNN trend survives calibration (Pearson 0.71), consistent with their local claim.

## Weak / open
- Saturation outside LLaMA. Hinge on log2 params with family FE: held-out unbiased CKA slope +0.0008 below vs −0.0039 above the median; mKNN +0.0048 vs +0.0013.
  - Per family: Mistral flat (+0.0002, CI spans 0); LLaMA flat; OLMo rising (1B→7B); Gemma drops (−0.09 per doubling, possibly a massive-activation artifact: top-1 variance share 0.50/0.78 vs >0.95 for most).
  - Only 2 sizes per new family. The capability axis flips sign with vs without family FE.
- Bootstrap resamples ViTs only (17 clusters), so CIs ignore LLM-level sampling and are too narrow. Needs a two-way (LLM × ViT) bootstrap.
- One dataset (WIT-1024).
- E3 (does the corrected metric predict unpaired Rosetta alignability?): smoke passed 15/15, full run pending.
