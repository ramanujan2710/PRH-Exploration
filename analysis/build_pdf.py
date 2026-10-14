import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether)
from PIL import Image as PILImage

D = os.path.dirname(os.path.abspath(__file__)); F = os.path.join(D, "figs")
pdfmetrics.registerFont(TTFont("DV", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DVB", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("DVI", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf"))
from reportlab.lib.fonts import addMapping
addMapping("DV", 0, 0, "DV"); addMapping("DV", 1, 0, "DVB"); addMapping("DV", 0, 1, "DVI"); addMapping("DV", 1, 1, "DVB")

INK, INK2, ACC, RULE, BAND = colors.HexColor("#0b0b0b"), colors.HexColor("#52514e"), colors.HexColor("#2a78d6"), colors.HexColor("#d9d8d3"), colors.HexColor("#f3f2ee")
st = {
    "title": ParagraphStyle("t", fontName="DVB", fontSize=19, leading=23, textColor=INK, spaceAfter=4),
    "sub": ParagraphStyle("s", fontName="DV", fontSize=9.5, leading=13, textColor=INK2, spaceAfter=10),
    "h1": ParagraphStyle("h1", fontName="DVB", fontSize=13.5, leading=17, textColor=INK, spaceBefore=10, spaceAfter=5),
    "h2": ParagraphStyle("h2", fontName="DVB", fontSize=10.5, leading=14, textColor=INK, spaceBefore=7, spaceAfter=3),
    "p": ParagraphStyle("p", fontName="DV", fontSize=9, leading=12.6, textColor=INK, spaceAfter=5, alignment=TA_LEFT),
    "b": ParagraphStyle("b", fontName="DV", fontSize=9, leading=12.6, textColor=INK, leftIndent=11, bulletIndent=2, spaceAfter=2.5),
    "cap": ParagraphStyle("c", fontName="DV", fontSize=7.8, leading=10.4, textColor=INK2, spaceBefore=2, spaceAfter=9),
    "cell": ParagraphStyle("cell", fontName="DV", fontSize=7.8, leading=10, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="DVB", fontSize=7.8, leading=10, textColor=INK),
    "box": ParagraphStyle("box", fontName="DV", fontSize=9, leading=12.6, textColor=INK),
}
W = A4[0] - 36 * mm

def P(t, s="p"): return Paragraph(t, st[s])
def B(items): return [Paragraph(x, st["b"], bulletText="•") for x in items]
def fig(name, cap, width=W):
    path = os.path.join(F, name); w, h = PILImage.open(path).size
    return KeepTogether([Image(path, width=width, height=width * h / w), P(cap, "cap")])
def table(rows, widths, head=True, bold_rows=()):
    data = [[Paragraph(str(c), st["cellb"] if (head and i == 0) or i in bold_rows else st["cell"]) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * W for w in widths], repeatRows=1 if head else 0)
    sty = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK2),
           ("LINEBELOW", (0, 1), (-1, -1), 0.3, RULE), ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
           ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    if head: sty.append(("BACKGROUND", (0, 0), (-1, 0), BAND))
    t.setStyle(TableStyle(sty)); return t
def box(items):
    t = Table([[ [P(x, "box") for x in items] ]], colWidths=[W])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), BAND), ("LINEBEFORE", (0, 0), (0, -1), 2.5, ACC),
                           ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                           ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    return t

def footer(c, d):
    c.saveState(); c.setFont("DV", 7.5); c.setFillColor(INK2)
    c.drawString(18 * mm, 10 * mm, "PRH project: results to date (12 Oct 2026)")
    c.drawRightString(A4[0] - 18 * mm, 10 * mm, f"{d.page}"); c.restoreState()

s = []
s += [P("Revisiting the Platonic Representation Hypothesis", "title"),
      P("Results to date: E1 simulation, E2 audit of 20 LLMs, E3 Rosetta yardstick, E6 global representations · 12 October 2026", "sub")]

s += [P("Summary", "h1"), box([
    "<b>1. Their calibration threshold measures dimensionality.</b> The Aristotelian calibrated test takes a maximum over all layer pairs. "
    "Its threshold equals the chance level of the highest-effective-rank layer pair (r = 0.99999 across 20 LLMs × 17 vision models), "
    "and it rises with model size inside every family. So their calibrated scaling trend partly measures dimensionality, not convergence.",
    "<b>2. Calibrating a single fixed pair is harmless.</b> On Rosetta's benchmark (420 unpaired alignments), calibrated, raw and unbiased CKA "
    "predict real alignment equally well (pre-registered H1: null). Rosetta's own result replicates almost exactly (ρ = −0.895, R² = 0.75).",
    "<b>3. Generative decoders hide their global structure.</b> Unpaired alignment error is 0.40 for the generative decoder vs 0.21 for its embedding-model "
    "counterpart. Clipping massive activations masks this: the decoder's CKA rises from 0.35 to 0.49.",
    "<b>4. A prompted global readout brings back convergence with scale.</b> On Qwen3 (0.6B → 8B), a training-free PromptEOL readout gives calibrated CKA 0.23 → 0.37 "
    "under the Aristotelian pipeline itself. Mean pooling (the readout Huh et al. and the Aristotelian paper use) stays flat at 0.25–0.27. "
    "This is so far one model family; replicating it across 20 LLMs is the next experiment.",
])]

s += [P("Experiments at a glance", "h2"), table([
    ["", "Question", "Scale", "Status"],
    ["E1", "Does the calibrated test recover a known planted alignment?", "Simulation, n = 1,024, widths 256–8,192", "Done"],
    ["E2", "Replication and mechanism on real models; trends with scale", "20 LLMs × 17 ViTs, WIT-1,024, 200 permutations, exact bf16", "Done"],
    ["E3", "Which similarity score predicts real unpaired alignability?", "84 combinations × 5 seeds = 420 Rosetta alignments", "Done"],
    ["E6", "Does a global language representation change the picture?", "7 language models × 3 readouts × 6 ViTs × 2 readouts", "Done"],
    ["E2-prompt", "Does the prompted readout restore convergence across 20 LLMs?", "Re-extraction with PromptEOL", "Next"],
], [0.09, 0.43, 0.36, 0.12])]

# ---------------- E1
s += [PageBreak(), P("E1 · Simulation: the calibrated score depends on effective rank", "h1"),
      P("We planted a shared latent signal with a known population CKA of 0.30 and varied the feature width, using the Aristotelian recipe "
        "(n = 1,024, 200 permutations, α = 0.05). With no shared signal, every calibrated score is about 0, so the calibration does its stated job. "
        "But with high-effective-rank features, the calibrated score falls as width grows, while the truth stays fixed."),
      fig("e1_width.png", "Figure 1. Calibrated CKA (orange) underestimates a fixed true alignment of 0.30 more and more as width grows when features have high effective rank (left), "
          "falling from 0.30 to 0.16. With a low-rank, power-law spectrum (right) there is no penalty. Unbiased CKA (green) matches the truth in every setting. Means over replicates."),
      P("Closed form (validated against the simulation, r = 0.954, mean error 0.005): raw CKA ≈ (CKA<sub>true</sub> + √(r<sub>x</sub>r<sub>y</sub>)/n) / √((1 + r<sub>x</sub>/n)(1 + r<sub>y</sub>/n)), "
        "where r is the effective rank. Calibration removes the additive chance term but not the shrinkage, which matters once r/n ≳ 0.5."),
      P("Other E1 results:", "h2")] + B([
      "Depth penalty (more layers to search) is not supported: with correlated layers the max-over-layers threshold does not grow from 8 to 80 layers.",
      "Mutual k-NN is not more powerful than CKA at detecting a shared signal.",
      "A few high-variance dimensions with no shared signal sharply cut global alignment (0.30 → 0.02); local mutual k-NN degrades less.",
      "Small-bandwidth RBF-CKA is degenerate: raw ≈ 1 and calibrated ≈ 0 even with a planted 0.30 alignment."])

# ---------------- E2
s += [PageBreak(), P("E2 · Real-model audit: 20 LLMs × 17 vision models", "h1"),
      P("We used the Huh et al. / Aristotelian pipeline (WIT-1,024; 95th-percentile clipping plus L2 normalisation; maximum over all layer pairs; "
        "permutation null on the maximum; 200 permutations), confirmed against the official Aristotelian code. All 20 LLMs ran in exact bf16; "
        "the largest models used CPU offload, which is bit-identical to an all-GPU run. LLaMA-65B in 4-bit and in bf16 agree within 0.002 on every metric."),
      P("Their headline reproduces", "h2"),
      table([["Size-rank correlation (original 12 LLMs)", "Ours", "Aristotelian paper"],
             ["Raw CKA", "0.87", "0.86"], ["Calibrated CKA", "0.47", "0.45"], ["Mutual k-NN", "0.85", "0.85"]], [0.5, 0.25, 0.25]),
      Spacer(1, 8),
      fig("e2_tau.png", "Figure 2. The calibration threshold τ is predicted exactly by the chance level of the highest-effective-rank layer pair on each side "
          "(r = 0.99999, mean error 0.002, all model pairs and preprocessings). It is set by dimensionality alone.", width=0.52 * W),
      fig("e2_families.png", "Figure 3. Per family (mean over 17 vision models, Huh preprocessing). The threshold τ rises with size in every family (second panel), so raw CKA rises "
          "and calibrated CKA falls in step with it, e.g. Llama-3 8B → 70B: τ 0.18 → 0.23, calibrated CKA 0.30 → 0.28. "
          "Held-out unbiased CKA is nearly flat at the top (Llama-3 0.334 → 0.338). Gemma-7B is an outlier dominated by a massive-activation dimension.")]
s += [P("Statistics with a two-way bootstrap (resampling LLMs and vision models)", "h2"),
      table([["Claim", "Evidence", "Verdict"],
             ["Threshold tracks effective rank and rises with size", "r = 0.99999; rises in all 7 families", "Strong"],
             ["Local alignment (mutual k-NN) rises with capability", "r = 0.73 [0.40, 0.97]", "Solid"],
             ["Calibrated / unbiased CKA rise with capability", "0.37 [0.14, 0.78] / 0.30 [0.04, 0.90]; difference −0.07 [−0.28, +0.12]", "Weak; no difference between estimators"],
             ["Saturation (kink at 7B)", "Above − below slope CI includes 0 for every metric", "Not established (descriptive only)"],
             ["Earlier CIs (vision models only)", "About 10× too narrow", "Corrected"]], [0.38, 0.40, 0.22]),
      Spacer(1, 8),
      fig("e2_vision.png", "Figure 4. Vision-side objective matters: MAE (masked reconstruction) aligns worst with the LLMs, DINOv2 (global self-distillation) better, "
          "and CLIP best (it was trained with language, so partly by construction).", width=0.62 * W)]

# ---------------- E3
s += [PageBreak(), P("E3 · Rosetta yardstick: which score predicts real unpaired alignment?", "h1"),
      P("Outcome: FOSCTTM of Rosetta's unpaired Wasserstein-Procrustes aligner (their code, unchanged) on all 40,504 MS COCO validation pairs, averaged over 5 seeds. "
        "Combinations: 4 training corpora × 7 vision models × 3 language models (mpnet, Qwen3-Embedding-8B, Qwen3-8B generative). "
        "Aristotelian measurements used their code at n = 1,024, K = 500, averaged over 20 random subsets. All 420 alignments completed."),
      fig("e3_predictors.png", "Figure 5. Predictive validity of each score (|Spearman ρ| with mean FOSCTTM over 84 combinations; 95% cluster-bootstrap CI over the 21 model pairs). "
          "Blue: unbiased CKA with Rosetta preprocessing; orange: the Aristotelian pipeline (clipping); grey: neighbourhood and topology scores.", width=0.86 * W),
      table([["Pre-registered test", "Δ|ρ| [95% CI]", "Verdict"],
             ["H0: replicate Rosetta (paper R² 0.73, ρ −0.90)", "R² 0.752, ρ −0.895", "Replicated"],
             ["H1: unbiased − calibrated CKA (same preprocessing)", "+0.002 [−0.002, +0.006]", "Null: calibrating a fixed pair is harmless"],
             ["H1b: raw − calibrated CKA", "+0.006 [−0.007, +0.023]", "Null"],
             ["H2: calibration does not change mutual k-NN", "−0.000 [−0.002, 0.000]", "Confirmed"],
             ["S1: n = 10,000 vs n = 1,024", "+0.002 [−0.001, +0.007]", "Sample size irrelevant"],
             ["Exploratory: Rosetta − Aristotelian preprocessing", "+0.102 [+0.042, +0.170]", "Clipping costs ~0.10 (see below)"]], [0.45, 0.25, 0.30])]
s += [P("The generative decoder aligns about twice as badly, and clipping hides it", "h2"),
      table([["Language model", "Mean FOSCTTM", "CKA (Rosetta prep)", "CKA (clipped)", "Calibrated CKA"],
             ["mpnet (encoder)", "0.212", "0.541", "0.524", "0.521"],
             ["Qwen3-Embedding-8B", "0.206", "0.583", "0.567", "0.565"],
             ["Qwen3-8B generative (decoder)", "0.399", "0.347", "0.494", "0.490"]], [0.34, 0.15, 0.17, 0.17, 0.17]),
      Spacer(1, 6),
      fig("e3_decoder.png", "Figure 6. Each point is one combination. With Rosetta preprocessing (left) the decoder's poor alignment shows up as low CKA. "
          "Clipping (right) lifts the decoder's CKA to the encoders' level while its alignment stays poor, which accounts for the whole 0.10 loss in predictive power. "
          "Within the encoders, every CKA variant predicts equally well (ρ ≈ −0.82). The decoder is worse in 23–24 of 28 matched comparisons (p < 1e-5); COCO is the one corpus where it is comparable.")]

# ---------------- E6
s += [PageBreak(), P("E6 · Global representations: decoders vs encoders, MAE vs DINOv2", "h1"),
      P("We tested Mayug's hypothesis (global structure survives calibration if the language side is a global representation) with the E2 engine on WIT-1,024. "
        "Language: Qwen3-Base decoders at 0.6B / 4B / 8B, read out by mean pooling, last token, or a PromptEOL prompt "
        "(the last token of <i>This sentence : \"…\" means in one word:\"</i>); Qwen3-Embedding at the same sizes; mpnet. "
        "Vision: MAE-B/L/H and DINOv2-B/L/g, each read by [CLS] and by patch mean. The checker passed 16/16."),
      fig("e6_readouts.png", "Figure 7. Alignment with DINOv2 ([CLS], mean of 3 sizes) by language readout and size. Left: under the Aristotelian pipeline, the PromptEOL readout "
          "rises steeply with size (calibrated CKA 0.23 → 0.32 → 0.37), while mean pooling and the embedding models stay nearly flat. Middle: at the final layer, embedding models "
          "beat mean-pooled decoders in 9/9 comparisons. Right: in the raw, unclipped geometry the gap is large and mean-pooled decoders get worse with size."),
      table([["Pre-registered prediction", "Result", "Verdict"],
             ["P1 (primary): embedding > decoder-mean, layer search + clipping", "calibrated −0.001 [−0.010, +0.006]; held-out unbiased +0.000", "Not supported"],
             ["P1 (secondary): same, final layer / no clipping", "+0.046 (9/9) / +0.14 to +0.18 (9/9)", "Supported"],
             ["P2: embedding models converge faster with scale", "slopes equal (+0.011 vs +0.011 per doubling)", "Not supported"],
             ["P3: encoder > decoder also with MAE (vs \"like with like\")", "+0.007 (6/9) clipped; +0.14 (9/9) unclipped", "Supported; \"like with like\" rejected"],
             ["P4: prompt / last-token readout ≥ mean", "prompt +0.05 (9/9); bare last token −0.09 (0/9)", "Prompt supported; last token is worst"],
             ["Exploratory: PromptEOL slope with scale", "+0.036 per doubling [0.031, 0.039] vs +0.003 for mean pooling", "Strong, one family"]], [0.38, 0.40, 0.22]),
      Spacer(1, 6),
      P("Reading: decoders do carry global structure in their intermediate layers, about as much as embedding models once massive activations are clipped. "
        "The deficit is in the readout (mean pooling, last token, final layer) and in the raw geometry. Asking the decoder for a global summary through a prompt "
        "exposes the structure, and it is the only readout under which alignment grows clearly with scale.")]

# ---------------- synthesis
s += [PageBreak(), P("What this adds up to", "h1"),
      table([["Claim", "Evidence", "Strength"],
             ["Calibrated PRH trend confounds alignment with dimensionality", "E1 closed form; E2 τ = chance level (r = 0.99999), rising in every family", "Very strong"],
             ["Calibrating a fixed pair is harmless", "E3 H1 null (+0.002 [−0.002, +0.006])", "Strong (clean null)"],
             ["Decoders hide global structure in their readout and raw geometry", "E3 decoder vs embedding (0.40 vs 0.21); E6 final layer and unclipped", "Strong, one model family for the paired comparison"],
             ["Clipping masks the decoder deficit", "E3 (+0.10 Δ|ρ|, entirely from the decoder); E6 unclipped gaps 3× larger", "Strong"],
             ["A global (prompted) readout restores convergence with scale", "E6 PromptEOL: 0.23 → 0.37 calibrated", "Promising, one family"],
             ["Global convergence saturates at the top", "E2 plateau, not significant under two-way bootstrap", "Descriptive only"]], [0.38, 0.42, 0.20]),
      P("Next steps", "h2")] + B([
      "<b>E2-prompt (decisive):</b> re-extract all 20 E2 LLMs with the PromptEOL readout and rerun the audit with the two-way bootstrap. Include prompt controls: a second summarising "
      "prompt, a neutral prompt of the same length, and the caption repeated twice. Add a second dataset (COCO or DCI captions).",
      "<b>Rosetta–PromptEOL:</b> unpaired alignment with prompted decoder embeddings. Does a training-free readout close the 0.40 → 0.21 FOSCTTM gap?",
      "<b>Mechanism:</b> effective rank and massive-activation share of the prompted readout; a learned text-only global embedding as a comparison to the prompt.",
      "<b>Mayug meeting (Wednesday):</b> compare setups (readouts, clipping, layer search) and agree framing and credit.",
      "<b>CVPR 2027:</b> abstract 10 Nov, paper 16 Nov. Go/no-go on the headline after E2-prompt and Rosetta–PromptEOL, around 25 Oct."]) + [
      P("Caveats", "h2")] + B([
      "WIT-1,024 is a single dataset of 1,024 pairs; E3 uses Rosetta's corpora, but E2 and E6 do not yet have a second dataset.",
      "The E6 readout results come from one LLM family (Qwen3) at three sizes, with 3 vision models per family, so confidence intervals are rough.",
      "With 20 LLMs, the two-way bootstrap leaves trend and saturation claims wide; only the mechanism and the mutual k-NN trend are tight."])

doc = SimpleDocTemplate(os.path.join(D, "PRH_results_2026-10-12.pdf"), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                        topMargin=16 * mm, bottomMargin=18 * mm, title="PRH results to date", author="PRH project")
doc.build(s, onFirstPage=footer, onLaterPages=footer)
print("ok")
