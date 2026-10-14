# E1: Planted-signal test of the Aristotelian calibration (run 10 Oct 2026)

Setup: synthetic vision/language embeddings with a shared latent; exact population linear CKA held fixed or rising; n = 1,024 unless noted; 200 permutations, alpha = 0.05 (Aristotelian recipe).

## Findings
- **Null check:** with no shared signal, all calibrated scores are about 0. Their calibration does its stated job.
- **Width penalty: conditional.** With high effective rank (uniform noise), calibrated CKA falls from 0.30 to 0.16 as width goes 256 to 8,192 while the truth stays at 0.30. A true rise of +0.10 shows up as -0.04. With power-law spectra (effective rank about 40-75), there is no penalty. Debiased CKA (unbiased HSIC) matches the truth in every setting.
- **Closed form (validated, r = 0.954, MAE 0.005):** raw CKA ≈ (CKA_true + sqrt(r_x r_y)/n) / sqrt((1 + r_x/n)(1 + r_y/n)), where r = (tr Σ)² / ||Σ||²_F is the effective rank. Calibration removes the additive term, not the shrinkage. The penalty matters once r/n ≳ 0.5.
- **Depth penalty: not supported.** With correlated layers (AR ρ = 0.9), the max-over-layers threshold doesn't grow from 8 to 80 layers.
- **Power asymmetry: not supported.** CKA's detection power ≥ mutual k-NN's at all widths.
- **Small-bandwidth RBF-CKA is degenerate.** At 0.1-0.25 × median distance, raw ≈ 1 and calibrated ≈ 0, even with a planted 0.30 alignment. It recovers at ≥ 1 × median.
- **Outlier dimensions:** a few high-variance dimensions with no shared signal cut true global alignment sharply (0.30 to 0.02 with 2 dims at 30× in the high-rank case). Local mutual k-NN degrades less (29% retained vs 7%). Per-dimension standardization restores alignment.
- **Sample size:** in the high-rank case, calibrated CKA approaches the truth as n grows (0.175 at n = 512 → 0.257 at n = 4,096).

## Decisive next step
Measure effective rank (participation ratio) and outlier-dimension magnitudes of the real features (Huh et al. 12 LLMs × 17 vision models, WIT 1,024), then use the closed form to predict how much of the Aristotelian flattening the calibration explains.
