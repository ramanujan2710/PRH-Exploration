# Audit v4: newer LLM families + exact bf16 LLaMA-65B (results zip _6, 11 Oct 2026)

Setup: WIT-1024 (Huh et al.), 17 ViTs, engine v3, 200 perms. 19 LLMs, all exact bf16; Mixtral and LLaMA-65B ran with GPU+CPU offload, which the smoke test showed is bit-exact. Llama-3-70B is still pending.

## Per-LLM means over 17 ViTs (Huh preprocessing)
| LLM | bpb | params (B) | raw CKA (max) | calibrated CKA | τ | unbiased CKA, held-out layer | mKNN |
|---|---|---|---|---|---|---|---|
| BLOOMZ-560m → 7b1 | 2.07 → 1.84 | 0.6 → 7.1 | 0.337 → 0.384 | 0.233 → 0.245 | 0.135 → 0.184 | 0.269 → 0.312 | 0.098 → 0.118 |
| OpenLLaMA-3b / 7b / 13b | 1.72 / 1.68 / 1.66 | 3.4 / 6.7 / 13 | 0.413 / 0.426 / 0.429 | 0.297 / 0.304 / 0.298 | | 0.334 / 0.346 / 0.338 | 0.137 / 0.141 / 0.144 |
| LLaMA-7b / 13b / 30b / 65b | 1.65 / 1.62 / 1.58 / 1.56 | 6.7 / 13 / 32.5 / 65 | 0.427 / 0.433 / 0.441 / 0.447 | 0.296 / 0.289 / 0.284 / 0.277 | 0.187 → 0.237 | 0.336 / 0.333 / 0.329 / 0.329 | 0.144 / 0.143 / 0.145 / 0.148 |
| Llama-3-8B | 1.64 | 8.0 | 0.423 | 0.295 | 0.182 | 0.334 | 0.144 |
| Mistral-7B / Mixtral-8x7B | 1.70 / 1.64 | 7.2 / 46.7 | 0.416 / 0.428 | 0.265 / 0.269 | 0.207 / 0.218 | 0.317 / 0.317 | 0.140 / 0.145 |
| OLMo-1B / 7B | 1.64 / 1.52 | 1.2 / 6.9 | 0.319 / 0.378 | 0.206 / 0.246 | | 0.243 / 0.277 | 0.099 / 0.121 |
| Gemma-2b / 7b | 1.82 / 1.76 | 2.5 / 8.5 | 0.386 / 0.244 | 0.275 / 0.093 | | 0.312 / 0.145 | 0.128 / 0.113 |

## Findings
1. **Mechanism replicates across all families.** τ is predicted by the effective-rank formula with r = 0.99999 (MAE 0.002). τ rises with size inside every family, and the calibrated CKA trend follows τ, not alignment.
2. **4-bit vs bf16 LLaMA-65B.** Alignment scores agree to within 0.002 (raw CKA 0.4458 vs 0.4467; unbiased held-out 0.3265 vs 0.3285), so the earlier 12-LLM results stand. But bpb moves from 1.661 (nf4) to 1.557 (bf16): 4-bit had made 65B look less capable than 30B. bf16 fixes the capability axis.
3. **Saturation holds outside the LLaMA family.** Families already near the top are flat:
   - LLaMA 7→65B: unbiased held-out CKA −0.007, mKNN +0.004.
   - Mistral 7B → Mixtral 47B: unbiased CKA 0.3166 → 0.3170.
   - OpenLLaMA 7→13B: 0.346 → 0.338.
   - Llama-3-8B sits at the same level (0.334).
   - Families that start low still climb (OLMo 1→7B +0.035; BLOOMZ +0.04 over 0.6→7B).
   - The picture is a common ceiling near unbiased CKA ≈ 0.33–0.35 and mKNN ≈ 0.145 for these ViTs on WIT-1024.
   - Pre-registered hinge on log2 params, family fixed effects: unbiased held-out CKA slope +0.0008 below the median size and −0.0039 above; mKNN +0.0048 / +0.0013.
4. **Capability does not order alignment across families.** OLMo-7B has the best bpb (1.52) but aligns clearly worse than LLaMA-7B, Llama-3-8B and OpenLLaMA. The family matters more than the language-modelling loss.
5. **Gemma-7B is an outlier; it should be investigated before use.**
   - Its global scores collapse (unbiased CKA 0.145) while mKNN barely drops (0.113 vs Gemma-2b 0.128).
   - Layer diagnostics: one dimension holds 70–78% of the variance in layers 1–25, max/median |activation| reaches ~6,800, and raw effective rank is ≈1.01. Its best raw-CKA layer is layer 1.
   - This looks like massive activations dominating global (CKA) geometry while leaving local (kNN) geometry intact.
   - Possible cause to test: the BOS/attention-sink token under left padding, mixed in by mean pooling. Robustness check: pool without BOS.
   - The capability-axis hinge results are driven by Gemma-7B and should be re-fit without it.

## Pending
- Llama-3-70B (`RUN_ONLY = "NousResearch/Meta-Llama-3-70B"`).
- Gemma-7B check: pool without BOS; re-fit the hinge excluding Gemma.
