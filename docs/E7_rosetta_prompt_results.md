# E7 (Rosetta–prompt): prompted decoder readouts in Rosetta's unpaired aligner (13 Oct 2026)

Setup: Rosetta's code unchanged (commit fdfad84).
- Wasserstein Procrustes; FOSCTTM on all 40,504 COCO val pairs.
- DCI and DOCCI × 7 vision models × 3 seeds (the first 3 of E3's).
- New final-layer caption embeddings for Qwen3-8B and Llama-3-8B: `avg` = token mean; `eol` = PromptEOL last token. Captions from Rosetta's own loaders and in their row order.
- Baselines from E3 (same seeds): Qwen3-8B generative token pooling (Rosetta's "gen"), Qwen3-Embedding-8B, mpnet.

Checker: 19/19 PASS.
- Rosetta's exact shape and caption-slot pattern for all files; values finite.
- Paired CKA > 0, shuffled ≈ 0.
- All 168 alignments + geometry present; FOSCTTM valid on 40,504 pairs.

## Mean FOSCTTM (lower = better)
| | qwen3-8b-gen | qwen3-8b-avg | qwen3-8b-eol | llama3-8b-avg | llama3-8b-eol | qwen3-embedding-8b | mpnet |
|---|---|---|---|---|---|---|---|
| DCI | 0.505 | 0.473 | 0.466 | 0.534 | 0.435 | 0.273 | 0.292 |
| DOCCI | 0.476 | 0.437 | 0.452 | 0.368 | 0.451 | 0.297 | 0.278 |

## Pre-registered tests (14 cells; CI = bootstrap over 7 vision models)
| Test | Result | Verdict |
|---|---|---|
| Q1 Qwen3 eol < Rosetta gen | Δ −0.032 [−0.064, −0.005]; better in 8/14 cells; Wilcoxon p = 0.36 | Marginal (CI excludes 0, but not consistent across cells) |
| Q2 eol closes ≥ 50% of the gen → embedding gap | median 18% (2/14 cells ≥ 50%) | **Not supported** |
| Q3 eol < avg | Qwen3 +0.004 [−0.056, +0.052]; Llama-3 −0.008 [−0.064, +0.067] | **Null** |
| Q4 Llama-3 eol within 0.05 of embedding models | +0.158 vs Qwen3-Embedding; +0.158 vs mpnet | **Not supported** |

## Geometry vs outcome (Rosetta scores, n = 10k)
| | CKA | mKNN | FOSCTTM |
|---|---|---|---|
| mpnet | 0.483 | 0.134 | 0.285 |
| Qwen3-Embedding-8B | 0.545 | 0.169 | 0.285 |
| Llama-3-8B eol | 0.444 | 0.094 | 0.443 |
| Llama-3-8B avg | **0.483** | 0.099 | **0.451** |
| Qwen3-8B avg | 0.353 | 0.069 | 0.455 |
| Qwen3-8B eol | 0.391 | 0.063 | 0.459 |
| Qwen3-8B gen | 0.208 | 0.041 | 0.491 |

- Llama-3-8B avg has the **same global CKA as mpnet (0.48) but far worse unpaired alignment (0.45 vs 0.29)**. Its local neighbourhood agreement (mKNN) is lower.
- On these 98 cells (long-caption corpora, with many decoder readouts), mKNN predicts FOSCTTM slightly better than CKA (Spearman −0.61 vs −0.56), and CKA adds nothing once mKNN is known.
  - This contrasts with E3's 84-combo grid, where CKA was best. Subset-specific; exploratory.

## Conclusions
1. **No training-free fix:** neither PromptEOL nor mean pooling makes decoder LLMs usable for unpaired alignment. All decoder readouts are 0.44–0.49 vs 0.28–0.29 for embedding models. The method contribution for CVPR (C3 / M1) fails.
2. Higher global CKA from a better readout does not translate into unpaired alignability. Decoders lack the local neighbourhood agreement that embedding models have.
3. Implication: the paper is an analysis/measurement paper. A trained global embedding (contrastive head / LLM2Vec-style) remains the only untested route to a method result.
