# E3 interim results

## Interim 2 (12 Oct 2026, 09:20 UTC): 349/420 alignments, 77/84 combos with 4–5 seeds
Seven combos have 0 alignments: worker 10 stalled on its first task (COCO dinov2-g14 × qwen3-8b-gen, seed 1) since 19:05 UTC on 11 Oct. Static sharding put all 5 seeds of those 7 combos on that worker. A rescue script re-splits that shard over 6 processes.
- Median across-seed SD of FOSCTTM: 0.04 (max 0.15).
- Runtime per alignment (median, 12 in parallel): COCO 51 min, SPC 20, DOCCI 13, DCI 8.

### Spearman with mean FOSCTTM (77 combos; CI = bootstrap over 21 model pairs)
| Predictor | ρ | 95% CI | R² |
|---|---|---|---|
| Rosetta CKA, n = 10k | −0.879 | [−0.923, −0.781] | 0.72 |
| unbiased CKA, Rosetta prep, n = 1,024 | −0.877 | [−0.922, −0.777] | 0.72 |
| Aristotelian raw CKA | −0.789 | [−0.854, −0.675] | 0.61 |
| Aristotelian calibrated CKA | −0.789 | [−0.857, −0.672] | 0.61 |
| unbiased CKA, their prep (fp32 = fp64) | −0.791 | [−0.857, −0.676] | 0.61 |
| mKNN raw = calibrated | −0.749 | [−0.821, −0.626] | 0.56 |
| Rosetta mKNN / TSI / QSI | −0.59 / −0.51 / −0.35 | | |

### Tests
| Test | Result | Reading |
|---|---|---|
| H0: Rosetta CKA vs paper (ρ −0.90, R² 0.73) | ρ −0.879, R² 0.72 | **Replicated** |
| H1: \|ρ\| unbiased (their prep) − calibrated | +0.002 [−0.002, +0.006] | **Null**: calibration of a fixed pair is harmless |
| H1b: raw − calibrated | −0.000 [−0.014, +0.016] | |
| H2: calibrated − raw mKNN | +0.000 | As predicted |
| S1: n = 10k vs n = 1,024 (Rosetta prep) | +0.002 [−0.002, +0.008] | Sample size irrelevant |
| Preprocessing (pooled): Rosetta prep (center + L2) − Huh/Aristotelian prep (95th-percentile clip + L2) | +0.086 [+0.030, +0.153] | See the decoder correction below |
| Unbiased CKA (their prep) − mKNN | +0.042 [−0.021, +0.105] | |

## Decoder vs encoder language models (key finding, 12 Oct)
The language models are mpnet and Qwen3-Embedding-8B (encoders/embedding models) and Qwen3-8B generative (decoder).

| LM | mean FOSCTTM | Rosetta-prep CKA | clipped CKA |
|---|---|---|---|
| mpnet | 0.212 | 0.541 | 0.524 |
| Qwen3-Embedding-8B | 0.204 | 0.583 | 0.567 |
| Qwen3-8B generative | **0.394** | **0.357** | 0.501 |

- The decoder aligns much worse unpaired: worse than mpnet in 17/21 matched (dataset, vision) cells (p = 2e-4) and worse than Qwen3-Embedding in 18/21 (p = 2e-5).
- Exception: on COCO it is comparable (0.20 vs 0.14–0.17). DOCCI/DCI/SPC: 0.46–0.49 vs 0.12–0.29.
- **The pooled preprocessing effect is this decoder effect.**
  - Within encoders: Rosetta prep −0.809 vs clipped −0.811 vs calibrated −0.808, so no difference.
  - Within the decoder: −0.612 vs −0.594 vs −0.590.
  - Clipping inflates the decoder's CKA (0.50 vs 0.36), hiding its alignment deficit. Likely mechanism: massive activations dominate the decoder's unclipped geometry, and that is the geometry the aligner sees.
- Correction to interim 1: "clipping removes real signal in general" is not supported. "Clipping masks a decoder-specific geometry problem" is.

## Interpretation
- For one fixed embedding pair, calibration does not hurt predictive validity (H1 null). The Aristotelian failure mode is the max-over-layers aggregate threshold growing with effective rank / model size (E1/E2), which E3 does not exercise.
- The Aristotelian paper (arXiv 2602.14486) does not attribute weak alignment to decoder architectures.
- New opportunity: unpaired alignment is substantially worse for a generative decoder LLM. Fixing that (outlier-/spectrum-aware preprocessing before Wasserstein Procrustes) is a candidate method contribution.
  - Pilot on DCI/DOCCI, where the gap is largest.
