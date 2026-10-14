# CVPR 2027 main-track plan (drafted 12 Oct 2026; revised 12 Oct after the E3 decoder finding and Mayug's input)

> Status 13 Oct: E6, E2-prompt and E7 have since run. E7 shows a prompt alone does not fix decoders for unpaired alignment, so C3 via prompting failed. A *trained* global head (M1) is still untested. The current framing is an analysis/measurement paper; see README.

Deadlines (AoE): abstract 10 Nov, paper 16 Nov, supplementary 23 Nov. Not extended. That leaves ~5 weeks.

## Pitch (revised)
"Global cross-modal structure is shared, if the language side has a global representation."
- The Aristotelian claim (only local structure converges) comes from (a) an aggregation threshold that scales with effective rank, and (b) measuring decoder LLM hidden states, which are not global sentence representations.
- With encoder/embedding models, global CKA survives calibration and predicts unpaired alignability.
- Turning a decoder into an encoder (a learned global embedding, text-only training) restores both.
- Mayug's observation is the core hypothesis. Credit him accordingly.

## Evidence so far
- E1/E2: the max-over-layers threshold equals the chance level of the highest-effective-rank layer pair (r = 0.99999, 20 LLMs). It rises with size inside every family.
- E3 interim (77/84 combos), the same base family in two forms:

  | LM | FOSCTTM | Rosetta-prep CKA | calibrated CKA (n = 1,024) |
  |---|---|---|---|
  | Qwen3-8B generative | 0.39 | 0.36 | 0.50 |
  | Qwen3-Embedding-8B | 0.20 | 0.58 | 0.57 |
  | mpnet | 0.21 | 0.54 | 0.52 |

  The decoder is worse in 18/21 matched cells.
- Correction: the earlier "clipping costs 0.09 |ρ|" result is driven entirely by the decoder (no effect within encoders). Clipping inflates the decoder's CKA and hides its deficit. Drop the old claim "outlier dims carry cross-modal signal".
- To check: how Rosetta builds the "qwen3-8b-gen" embedding (pooling / prompt), and whether Qwen3-Embedding-8B is trained from the same base.

## Contributions → evidence
| # | Contribution | Experiment | Status |
|---|---|---|---|
| C1 | Diagnosis: calibrated PRH trend measures effective rank (aggregation threshold) | E1, E2 | Done |
| C2 | Global structure is shared for global (encoder) representations: calibrated global CKA high and rising with scale; predicts unpaired FOSCTTM | E3 (done); **E6**: audit Qwen3-Embedding 0.6B/4B/8B vs Qwen3 0.6B/4B/8B decoders (+ another encoder family) on WIT-1024 × 17 ViTs | E6 done (rising-with-scale part not supported) |
| C3 | Method: decoder → encoder via a learned global embedding trained on text only (no image pairs, so not circular) restores calibrated global CKA and Rosetta unpaired alignment | **M1**: Qwen3-8B (+ Llama-3-8B) with a text-only contrastive objective (SimCSE-style / LLM2Vec-style; LoRA or pooling head) → E2-style audit + Rosetta on DCI/DOCCI | Prompt-only version failed (E7); trained version open |
| C4 | Practical: predict alignability before training; model-pairing rule (regret vs oracle) | E3 data (+ E3b adapter retrieval with Mayug) | Analysis |

## Novelty risks
- LLM2Vec / NV-Embed / Qwen3-Embedding already turn decoders into encoders for text retrieval. Our claim is about cross-modal convergence and unpaired alignability, which they do not test.
- Rosetta already uses embedding models. Our contribution is explaining why decoders fail, and the fix.
- Need a literature check before 18 Oct.

## Timeline
- **Week 1 (12–18 Oct):** finish E3 (final analysis); build + run E6; check the qwen3-8b-gen construction; literature check; Mayug meeting (Wed): align on C2/C3 and his prior results.
- **Week 2 (19–25 Oct):** M1 text-only decoder→encoder training (1–2 decoders) + Rosetta pilot on DCI/DOCCI. **Go/no-go on C3 by 25 Oct.**
- **Week 3 (26 Oct–1 Nov):** M1 at scale (all 4 corpora × 7 vision × 3 seeds); E6 extended; C4 analysis; figures.
- **Week 4 (2–8 Nov):** write; ablations; register abstract by 10 Nov.
- **Week 5 (9–16 Nov):** polish, internal review, submit. Supplementary by 23 Nov.

## Fallback if C3 fails
Submit C1 + C2 + C4 as an analysis paper (weaker main-track odds), or move it to ICML.
