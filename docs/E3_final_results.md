# E3 final results: which similarity score predicts real unpaired alignability? (12 Oct 2026)

Complete: 420/420 alignments (84 combos × 5 seeds), 84 Rosetta geometry, 84 Aristotelian metric files (GPU, n = 1,024, K = 500, 20 subsets).
- Outcome: Rosetta Wasserstein-Procrustes FOSCTTM on all 40,504 COCO val pairs, mean over 5 seeds.
- Combos: 4 training corpora × 7 vision × 3 language models (mpnet, Qwen3-Embedding-8B, Qwen3-8B generative).
- Rosetta commit fdfad84; Aristotelian commit 785054b.

Checker: all pass except "FOSCTTM within [0, 0.5]". 39/420 runs are worse than chance (> 0.5), 34 of them the generative decoder on DCI/DOCCI/SPC. This is real (systematically wrong maps), not a pipeline error; the check's bound was too strict.

## Predictive validity (84 combos; CI = cluster bootstrap over 21 model pairs)
| Predictor | Spearman ρ | 95% CI | R² |
|---|---|---|---|
| Rosetta CKA (unbiased, fp64, n = 10k) | **−0.895** | [−0.930, −0.810] | **0.752** |
| Unbiased CKA, Rosetta prep, n = 1,024 | −0.894 | [−0.929, −0.806] | 0.754 |
| Aristotelian raw CKA (clip + L2) | −0.796 | [−0.859, −0.691] | 0.616 |
| Aristotelian calibrated CKA | −0.790 | [−0.857, −0.679] | 0.623 |
| Unbiased CKA, their prep (fp32 = fp64) | −0.792 | [−0.858, −0.680] | 0.622 |
| Mutual k-NN raw = calibrated (n = 1,024) | −0.772 | [−0.831, −0.664] | 0.606 |
| Rosetta mKNN / TSI / QSI (n = 10k) | −0.65 / −0.59 / −0.44 | | 0.43 / 0.40 / 0.20 |

## Pre-registered tests
| Test | Δ\|ρ\| [95% CI] | Verdict |
|---|---|---|
| H0: replicate Rosetta (paper R² 0.73, ρ −0.90) | R² 0.752, ρ −0.895 | **Replicated** |
| H1: unbiased − calibrated CKA (same prep, n = 1,024) | +0.002 [−0.002, +0.006] | **Null**: calibrating a fixed pair is harmless |
| H1b: raw − calibrated CKA | +0.006 [−0.007, +0.023] | Null |
| H2: calibration doesn't change mKNN | −0.000 [−0.002, 0.000] | Confirmed |
| S1: n = 10k vs n = 1,024 (Rosetta prep) | +0.002 [−0.001, +0.007] | Sample size irrelevant |
| S2: fp32 vs fp64 | 0.000 | Precision irrelevant |

Exploratory contrasts:
- Rosetta prep − Huh/Aristotelian clipped prep: **+0.102 [+0.042, +0.170]**.
- Rosetta CKA − calibrated: +0.105 [+0.046, +0.173].
- Unbiased CKA (their prep) − mKNN: +0.020 [−0.036, +0.077].

## Decoder vs encoder language models
| LM | mean FOSCTTM | Rosetta-prep CKA | clipped CKA | calibrated CKA | mKNN |
|---|---|---|---|---|---|
| mpnet | 0.212 | 0.541 | 0.524 | 0.521 | 0.258 |
| Qwen3-Embedding-8B | 0.206 | 0.583 | 0.567 | 0.565 | 0.283 |
| Qwen3-8B generative | **0.399** | **0.347** | 0.494 | 0.490 | 0.219 |

- The generative decoder is worse than mpnet in 24/28 matched (corpus, vision) cells (Wilcoxon p = 6e-6), and worse than Qwen3-Embedding in 23/28 (p = 3e-6).
- Per corpus (decoder vs encoders): DOCCI 0.47 vs 0.27–0.28; DCI 0.50 vs 0.28–0.29; SPC 0.46 vs 0.12. COCO is the exception: 0.17 vs 0.14–0.17.
- The pooled preprocessing effect comes from the decoder.
  - Within encoders: Rosetta prep −0.817, clipped −0.820, calibrated −0.817.
  - Within the decoder: −0.574, −0.554, −0.553.
  - Clipping inflates the decoder's CKA (0.35 → 0.49), hiding its deficit.
- Within the decoder, mKNN predicts best (−0.70 vs −0.57 for CKA): local structure is what is left.

## Within-corpus Spearman (n = 21 each)
| Corpus | Rosetta CKA | calibrated CKA | mKNN |
|---|---|---|---|
| COCO | −0.93 | −0.76 | −0.57 |
| SPC | −0.88 | −0.73 | −0.77 |
| DOCCI | −0.74 | −0.67 | −0.73 |
| DCI | −0.70 | −0.59 | −0.54 |

## Conclusions
1. Rosetta's headline (unbiased CKA predicts unpaired alignability) replicates exactly. It holds at n = 1,024 too.
2. The Aristotelian calibration neither helps nor hurts for a fixed embedding pair (pre-registered H1 null). Their failure mode is the max-over-layers threshold (E1/E2), which E3 does not exercise.
3. Huh/Aristotelian clipping costs ~0.10 in |ρ|. This is entirely by masking the generative decoder's poor geometry: it makes the decoder look as aligned as the encoders when it aligns twice as badly.
4. The generative decoder aligns far worse unpaired than encoder/embedding models, including the embedding model from the same Qwen3 family. This supports Mayug's view: global structure is shared when the language side has a global representation.
