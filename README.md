# PRH audit: is cross-modal convergence real, and how should we measure it?

This repo re-examines the *Aristotelian* critique of the Platonic Representation Hypothesis (arXiv 2602.14486). The critique says only local structure converges across vision and language models. Global structure, it says, does not.

- **Target:** CVPR 2027 (abstract 10 Nov, paper 16 Nov). ICML (Jan) is the fallback.
- **Status:** 13 Oct 2026.

All heavy runs were done on Colab G4 (RTX PRO 6000, 95 GB) from the notebooks in `notebooks/`. Every result in `results/` is the raw output of those runs.

## Headline findings

| # | Claim | Strength | Evidence |
|---|---|---|---|
| 1 | The Aristotelian calibrated scaling trend is **confounded by effective rank**. The calibration threshold τ (max over all layer pairs) is exactly the chance level of the highest-effective-rank layer pair (r = 0.99999, 20 LLMs). τ rises with model size in every family, so calibrated CKA falls even when alignment does not. | Strong (deterministic mechanism) | E1, E2 |
| 2 | We replicate their numbers: size-rank r 0.87 / 0.47 / 0.85 vs the paper's 0.86 / 0.45 / 0.85. | Strong | E2 |
| 3 | Global alignment is real (calibrated CKA ≈ 0.25–0.34 for LLMs ≥ 1B) but **flat beyond ~1B**. This holds for every readout (mean pooling, PromptEOL, embedding models). Saturation is descriptive only: the two-way bootstrap cannot establish a kink. | Moderate | E2, E2-prompt, E6 |
| 4 | Rosetta's result replicates: unbiased CKA predicts unpaired alignability (ρ −0.895, R² 0.75). The Aristotelian calibration of a *fixed* pair is harmless (H1 null). Huh-style clipping costs ~0.10 in \|ρ\|, entirely by hiding the decoder's bad geometry. | Strong | E3 |
| 5 | **Generative decoder LLMs are poor partners for unpaired cross-modal alignment, whatever the readout.** FOSCTTM is 0.40–0.49 for the decoder vs 0.21–0.29 for embedding models (Qwen3 and Llama-3; mean, generative pooling and PromptEOL). | Strong | E3, E7 |
| 6 | Decoders do contain global structure in their intermediate layers. Under the Aristotelian pipeline itself, they match embedding models. The deficit is at the output: massive activations plus the readout. A prompt (PromptEOL) fixes the raw geometry (+0.13 CKA) but **not** unpaired alignment. | Moderate | E6, E2-prompt, E7 |

### Mayug's claim, checked

> "Global structure (CKA even after calibration) is present if you find a good global representation: use language encoders, or turn a decoder into an encoder by learning a global embedding."

**Mostly true:**
- With embedding models, calibrated global CKA is clearly present.
- They beat decoders at the final layer (+0.05 clipped, +0.14 to +0.18 unclipped).
- They align about 2× better unpaired.

**Refinements:**
- (a) The structure is *inside* decoders too (intermediate layers). It is hidden at the output.
- (b) A free prompt does not turn a decoder into a usable encoder. A *trained* embedding model does.
- (c) Better readouts do not make alignment rise with scale.

## Repo layout

```
notebooks/            Colab notebooks (final versions; archive/ = earlier audit versions)
  E2_audit_v4_20LLMs.ipynb          20 LLMs x 17 ViTs, exact bf16, RUN_ONLY / RUN_EXCEPT flags
  E2_audit_v4_llama3_70b.ipynb      Llama-3-70B with GPU + CPU offload
  E3_rosetta_yardstick.ipynb        Rosetta unpaired aligner: 84 combos x 5 seeds
  E6_global_representations.ipynb   decoder vs embedding x MAE vs DINOv2 (Qwen3 0.6/4/8B)
  E2prompt_prompted_readouts.ipynb  PromptEOL and controls across 20 LLMs
  E7_rosetta_prompt.ipynb           prompted decoder readouts in Rosetta's aligner
src/                  modules the notebooks are built from (cells_*.py + build_*.py -> .ipynb)
  prh_engine.py       alignment engine: permutation null, unbiased CKA, held-out layer, mutual k-NN
  prh_extract_v4.py   exact-bf16 extraction, GPU/CPU placement, per-model subprocess
  e2p_extract.py, e6_extract.py, e7_text.py, e3_worker.py
e1_simulation/        E1 planted-signal simulation (CPU): scripts, results, figures
analysis/             local analyses (two-way bootstrap, E3 interim, PDF report builder)
results/              raw outputs returned from Colab
  e2_audit/           alignment_all.csv (20 LLMs), diagnostics, two-way bootstrap, figures
  e2_prompt/          main_18llm/, llama_65b/, llama3_70b/
  e3/                 420 alignments, geometry, GPU metrics, tests
  e6/, e7/
report/               PRH_results_2026-10-12.pdf (E1, E2, E3, E6) + figures
docs/                 write-ups per experiment, the code review, and the CVPR plan
```

## Experiments at a glance

| Exp | Question | Write-up |
|---|---|---|
| E1 | Does the Aristotelian calibration recover planted alignment? | `docs/E1_simulation_results.md` |
| E2 | Full audit, 20 LLMs × 17 ViTs: replication, τ mechanism, trends, two-way bootstrap | `docs/Audit_*.md` |
| E3 | Which similarity score predicts real unpaired alignability (Rosetta)? | `docs/E3_final_results.md` |
| E6 | Decoder vs embedding models, MAE vs DINOv2 | `docs/E6_results.md` |
| E2-prompt | Does a global (PromptEOL) readout restore convergence with scale? | `docs/E2prompt_results.md` |
| E7 | Does PromptEOL make decoders usable for unpaired alignment? | `docs/E7_rosetta_prompt_results.md` |

## Reproducing

1. Open a notebook in Colab on a large GPU (G4 / A100 80 GB). Set `SMOKE_TEST = True` first; every notebook ends with a checker cell.
2. Put the Hugging Face token in Colab **Secrets** as `HF_TOKEN`. Never paste it into a cell. It is needed for Gemma / Llama-3.
3. E3 and E7 clone Rosetta (commit fdfad84) and use its embedding cache. E2 / E6 / E2-prompt use the WIT-1024 set from Huh et al.
4. `analysis/figs.py` has a hard-coded results path at the top. Point it at `results/`.

## Open items (next steps)

- A second matched decoder / embedding pair (e.g. LLM2Vec-Llama-3).
- E7 on all 4 corpora (COCO, SPC).
- A second dataset for E2.
- Compare with Mayug's own setup.
- A trained text-only global head on a decoder (the only untested route to a method contribution).
