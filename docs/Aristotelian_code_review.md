# Aristotelian paper: official code review (11 Oct 2026)

Repo: github.com/mlbio-epfl/Aristotelian (commit 785054b, 24 Jun 2026). Linked from the authors' lab page (brbiclab.epfl.ch/projects/aristotelian).

## What their paper runs used
- **Preprocessing:** `PRH_Q_OUTLIER = 0.95` (scripts/experiments/sections/prh.py), i.e. Huh et al.'s 95th-percentile clip, then L2-normalize. The low-level function default is q = 1.0, but the paper runner passes 0.95. **So our `huh` variant is their setting; the no-clipping result does not describe their pipeline.**
- **Permutations and threshold:** 500 permutations, α = 0.05, order-statistic τ (same formula as ours).
- **Aggregation:** max over all layer pairs; the null is the max over all layer pairs under the same permutation (layerwise_engine.compute_alignment_gated_cka_cached). Identical to our engine.
- **Unbiased CKA:** also run with the same calibration, in fp32, with TF32 enabled on GPU and +1e-6 in the denominator. That's the regime where we saw garbage values for near-constant layers (ViT block-0). Since a max is taken over layer pairs, such values could affect their unbiased-CKA results. **Hypothesis, unverified.**
- **Trend statistic:** not in the repo. Plots order LLMs by parameter size.

## Reproducing their numbers on our full run (11 LLMs, huh, 200 permutations)
Pearson correlation with LLM *size* rank: averaged per ViT, raw CKA 0.84 / calibrated 0.50 / mutual k-NN 0.82; on LLM means, 0.86 / 0.57 / 0.82. Paper: 0.86 → 0.45, mutual k-NN ≈ 0.85. **Close replication.** Using capability (bpb) instead of size, the numbers are higher (raw 0.96–0.99, calibrated 0.81–0.85).

## What they already address
Fig. 12 (scripts/analyses/cka_debiasing_comparison.py) compares calibrated vs debiased CKA (Song 2012) and dep-cols CKA, but on **single-layer** synthetic data (rank-10 signal, isotropic noise). It does not test the max-over-layers aggregate with layers of very different effective rank, which is where our mechanism operates. Cite it and state the distinction explicitly.
