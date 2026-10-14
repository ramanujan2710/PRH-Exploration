# E2-prompt: prompted readouts across all 20 LLMs (12 Oct 2026, final)

Setup: E2 pipeline unchanged (WIT-1024, 17 ViTs, all layer pairs, permutation null on the max, 200 permutations, exact bf16).
- Readouts: last token of a template. `eol` = PromptEOL; `eol_alt` = a second summarising prompt; `neutral` = the same frame without a summary request; `repeat` = the caption twice.
- Mean-pooled baseline = E2.
- `eol` on all 20 LLMs; controls on 16 (≤ 40 GB of weights).
- LLaMA-65B and Llama-3-70B ran in separate sessions: CPU offload, batch 8, 17–19 min each.

Checker: all pass. The only FAIL is the logged first 65B OOM attempt, which was resolved by the rerun.
- Every prompted readout sits on the template's final token for every caption.
- Threshold formula r = 0.99998; null 10/10.

## Pre-registered results (primary: calibrated CKA, Huh clipping, layer search; two-way bootstrap LLM × ViT; 20 LLMs)
| Test | Result | Verdict |
|---|---|---|
| **R1** slope per doubling, eol − mean (family + ViT FE) | eol −0.001, mean −0.006; Δ +0.004 [−0.018, +0.039] | **Not supported** |
| **R2** Pearson with capability, eol − mean | eol −0.07, mean +0.37; Δ −0.44 [−1.24, +0.07] | **Not supported** (eol has no capability trend) |
| **R3** eol − neutral (16 LLMs) | **+0.146 [0.112, 0.177]** | Supported: the summary request matters |
| R3 eol_alt − eol | +0.017 [−0.010, +0.041] | Supported (two summarising prompts agree) |
| R3 eol − repeat | +0.068 [0.041, 0.090] | Summary > just seeing the caption twice |
| eol − mean, level (huh, 18 LLMs) | +0.024 [−0.005, +0.053] | ≈ equal with clipping |
| eol − mean, level (noclip, 18 LLMs) | **+0.134 [0.094, 0.174]** | Prompt far better in raw geometry |
| neutral − mean (huh) | −0.115 [−0.165, −0.062] | A non-summary last token loses global structure |

Noclip slope: mean pooling −0.018 per doubling vs eol −0.000 (Δ +0.018 [−0.006, +0.053]). Mean pooling's raw-geometry decline with size disappears under the prompt.

## Per family (calibrated CKA): no rise with scale under the prompt
| Family | mean (huh) | eol (huh) | mean (noclip) | eol (noclip) |
|---|---|---|---|---|
| LLaMA 7 → 13 → 30 → 65B | 0.30 → 0.29 → 0.28 → 0.28 | 0.30 → 0.29 → 0.24 → 0.26 | 0.23 → 0.23 → 0.12 → 0.10 | 0.30 → 0.29 → 0.24 → 0.25 |
| Llama-3 8 → 70B | 0.30 → 0.28 | 0.28 → 0.26 | 0.19 → 0.12 | 0.28 → 0.25 |
| Mistral 7B → Mixtral | 0.27 → 0.27 | 0.28 → 0.25 | 0.18 → 0.11 | 0.30 → 0.27 |
| OpenLLaMA 3 → 7 → 13B | 0.30 → 0.30 → 0.30 | 0.33 → 0.33 → 0.28 | 0.14 → 0.14 → 0.13 | 0.33 → 0.33 → 0.28 |
| BLOOMZ 0.6 → 7.1B | 0.23 → 0.25 | 0.24 → 0.34 (1.7B) → 0.29 | 0.08 → 0.11 | 0.25 → 0.35 → 0.28 |
| OLMo 1 → 7B | 0.21 → 0.25 | 0.28 → 0.27 | 0.22 → 0.25 | 0.29 → 0.29 |
| Gemma 2 → 7B | 0.28 → 0.09 | 0.15 → 0.26 (the prompt fails on Gemma-2B) | 0.13 → 0.04 | 0.05 → 0.27 |

## Conclusions
1. **The E6 Qwen3 result (prompt readout rising 0.23 → 0.37 with scale) does not generalise.** Across 20 LLMs / 7 families, no readout shows convergence with scale beyond ~1B (pre-registered R1/R2 null). The headline "global readout restores convergence" is dropped.
2. Robust: a **summarising prompt** gives a global readout whose alignment with vision is (a) far above a neutral last token (+0.15), (b) equal to mean pooling after clipping, and (c) **much higher than mean pooling in raw geometry** (+0.13). It is also immune to the massive-activation decline that makes mean-pooled raw alignment fall with model size (e.g. LLaMA-65B 0.10 vs 0.25).
3. What this means for the PRH:
   - Global cross-modal alignment is real and well above chance (calibrated CKA ≈ 0.25–0.34 for models ≥ 1B).
   - It is roughly constant across 1B → 70B and across families, consistent with a plateau, not continued convergence.
   - Apparent scale trends in either direction are measurement artefacts: the threshold rising with effective rank (E1/E2), and massive activations under raw mean pooling.
4. Next:
   - Decide the paper framing ("measurement, not convergence").
   - Rosetta unpaired test of prompt vs mean readouts (done: E7).
