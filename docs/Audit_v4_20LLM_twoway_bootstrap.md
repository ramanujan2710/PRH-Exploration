# Audit v4 final: 20 LLMs × 17 ViTs + two-way bootstrap (results zip _7, 11 Oct 2026)

All 20 LLMs in exact bf16. Llama-3-70B: GPU+CPU offload, batch 12, 20 min, bpb 1.548 (best of all 20). WIT-1024, 200 perms, Huh preprocessing.

## Llama-3 8B → 70B (second family with a wide size range)
| | 8B | 70B |
|---|---|---|
| bpb | 1.636 | 1.548 |
| unbiased CKA, held-out layer | 0.334 | 0.338 |
| mKNN | 0.144 | 0.148 |
| raw CKA (max) | 0.423 | 0.448 |
| calibrated CKA | 0.295 | 0.283 |
| τ | 0.182 | 0.230 |

This is the same pattern as LLaMA 7→65B. Corrected global and local scores are ~flat over 3 doublings. Raw CKA rises and calibrated CKA falls, both tracking τ (effective rank), not alignment.

## Two-way bootstrap (resample LLMs AND ViTs, B = 2000)
Hinge on log2 params (hinge at 7.0B), with family fixed effects, 95% CI:

| Metric | Slope below hinge | Slope above hinge | Above − below |
|---|---|---|---|
| unbiased CKA, held-out | +0.0006 [−0.061, +0.014] | −0.0022 [−0.028, +0.001] | [−0.028, +0.060] |
| mKNN | +0.0048 [−0.004, +0.008] | +0.0013 [−0.001, +0.003] | [−0.007, +0.005] |

- ViT-only CIs (the previous method) were roughly 10× too narrow.
- **No above-vs-below difference excludes 0 for any metric or axis.** With 20 LLMs, a saturation kink is not statistically established. The plateau is descriptive only (top families cluster at unbiased CKA 0.33–0.35, mKNN ≈ 0.145).
- The capability-axis hinge is unstable (CIs span ±1). Do not report it.

Trend vs capability (Pearson over LLM means, two-way CI):

| Metric | All 20 LLMs | Without Gemma-7B (exploratory) |
|---|---|---|
| mKNN | 0.73 [0.40, 0.97] | 0.73 [0.36, 0.98] |
| raw CKA | 0.54 [0.29, 0.99] | 0.67 [0.29, 0.99] |
| calibrated CKA | 0.37 [0.14, 0.78] | 0.51 [0.09, 0.83] |
| unbiased CKA, held-out | 0.30 [0.04, 0.90] | 0.41 [−0.03, 0.94] |

- r(unbiased held-out) − r(calibrated) = −0.07 [−0.28, +0.12]: no evidence that either estimator shows a stronger global trend.

## Implications for Paper A
- Robust: the τ mechanism (r = 0.99999, deterministic), replication, and the local mKNN trend (positive, CI excludes 0).
- Not supported at this sample size: "saturation" as a statistical claim; "the corrected estimator restores a stronger global trend".
- The case that calibrated-CKA trends are confounded by effective rank stands on the mechanism (τ rising with size inside every family, e.g. Llama-3 0.182 → 0.230), not on trend correlations.
- E3 (functional yardstick) is now the decisive experiment for which metric to trust.
