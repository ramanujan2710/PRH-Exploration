# E6: global representations, decoder vs encoder × MAE vs DINOv2 (12 Oct 2026; v2 final)

Setup: WIT-1024, E2 engine (v3), 200 permutations, huh (clip + L2) and noclip (L2).
- Language: Qwen3-{0.6B, 4B, 8B}-Base decoders (readouts mean / last / PromptEOL prompt), Qwen3-Embedding-{0.6B, 4B, 8B} (last = native, mean), mpnet (mean = native, cls).
- Vision: MAE-B/L/H and DINOv2-B/L/g (cls, patch-mean).
- Both the layer search (Aristotelian pipeline) and the final layer (the model's own output).

Checker v2: **16/16 PASS**. v1 bug: fixed-token readouts have a constant layer 0, so CKA = 0/0. v2 drops zero-variance layers; exactly layer 0 was dropped for the 7 affected sets. Threshold formula r = 0.99999.

## Pre-registered predictions: verdicts
| | Test | Result | Verdict |
|---|---|---|---|
| **P1 primary** | embedding(last) − decoder(mean), **layer search, huh**, DINOv2 cls | calibrated −0.001 [−0.010, +0.006] (5/9); held-out unbiased +0.000 [−0.002, +0.004] (6/9) | **Not supported** (null) |
| P1 secondary | same, final layer, huh | calibrated **+0.046** (9/9); unbiased +0.044 (9/9); mKNN +0.020 (9/9) | Supported |
| P1 secondary | same, noclip (layer search / final) | held-out unbiased **+0.184** (9/9) / calibrated +0.143 (9/9) | Supported, large |
| P2 | slope per doubling, layer search huh, DINOv2 cls: embedding vs decoder-mean | held-out unbiased +0.011 vs +0.011; calibrated +0.005 vs +0.003 (CI incl. 0) | **Not supported** (equal slopes) |
| P3 | encoder − decoder gap with MAE (huh layer search / noclip) | +0.007 calibrated (6/9) / +0.14 (9/9) | Supported: "like with like" rejected |
| P4 | decoder prompt − mean | final +0.050 (9/9); layer search calibrated +0.044 (6/9). Last token − mean: −0.09 final, −0.12 layers (0/9) | Supported; bare last token is the worst readout |

## The unexpected result: PromptEOL decoders, under the Aristotelian pipeline itself
Layer search, huh, DINOv2 cls, mean over 3 DINOv2 sizes:

| LM / readout | calibrated CKA 0.6B / 4B / 8B | held-out unbiased | τ (8B) |
|---|---|---|---|
| Qwen3 Base, mean (E2/Huh readout) | 0.253 / 0.271 / 0.260 | 0.327 / 0.357 / 0.369 | 0.226 |
| Qwen3 Base, **PromptEOL** | 0.231 / **0.318 / 0.367** | 0.307 / 0.404 / **0.436** | 0.159 |
| Qwen3-Embedding, last | 0.254 / 0.241 / 0.285 | 0.333 / 0.335 / 0.385 | 0.203 |
| mpnet, mean | 0.284 | 0.377 | 0.180 |

Scale slope per doubling:
- PromptEOL calibrated **+0.036** [0.031, 0.039], held-out unbiased +0.035.
- Embedding: +0.005 / +0.011.
- Decoder-mean: +0.003 (CI incl. 0) / +0.011.

**With a prompted global readout, calibrated global alignment rises steeply with scale, even with the Aristotelian calibration.** The prompt readout's τ is also lower (0.159 vs 0.226), because its highest-rank layer has lower rank.

> Update (E2-prompt, 20 LLMs): this scale trend is Qwen3-specific and does **not** generalise. See `E2prompt_results.md`.

## Takeaways
1. Mayug's hypothesis needs refining.
   - Under the Aristotelian pipeline (clip + layer search), decoders' intermediate layers already align as well as embedding models.
   - The decoder deficit is in (a) the raw, unclipped geometry (massive activations) and (b) the final-layer / mean / last-token readouts.
   - This explains E3: Rosetta used raw final embeddings, where the deficit is large.
2. Headline candidate (later refuted by E2-prompt): **PromptEOL readouts restore a strong, calibrated convergence-with-scale trend** (0.23 → 0.37 over 0.6B → 8B). Embedding models and mean-pooled decoders are nearly flat.
3. "Like with like" (MAE ↔ decoders) is rejected.
4. Caveats: one family (Qwen3), 3 sizes, n = 1,024, 3 vision models per family.
5. Next (done since):
   - **E2-prompt**: re-extract the 20 E2 LLMs with a PromptEOL readout and rerun the audit.
   - **E7 (Rosetta–PromptEOL)**: unpaired FOSCTTM as the method test.
