# ===================== E2-prompt: prompted last-token readouts for decoder LLMs (exact bf16) =====================
# Each caption is wrapped in a template; the readout is the hidden state of the LAST token at every layer
# (embeddings + each block; the last entry is after the final norm). Left padding, so position -1 is always real.
# Weights: exactly as in the E2 audit (bf16, GPU or GPU+CPU offload via prh_extract_v4.plan_placement / load_llm).
import gc, os, torch
from prh_extract_v4 import load_tokenizer, download_llm, load_llm

PROMPTS = {
    # main readout: PromptEOL (Jiang et al. 2023)
    "eol": 'This sentence : "{}" means in one word:"',
    # second summarising prompt, different wording
    "eol_alt": 'This image caption : "{}" can be summarized in one word:"',
    # control: same frame and length, but asks for a continuation, not a summary
    "neutral": 'This sentence : "{}" was written down and then',
    # control: caption repeated, so the last token has read the whole caption once already (no instruction)
    "repeat": '{}\n{}',
}

def fill(template, caption):
    return template.replace("{}", caption)

@torch.no_grad()
def extract_prompted(name, texts, plan, readouts, offload_dir="/content/offload", batch_size=32, act_budget_gb=7.0):
    path = download_llm(name)
    tok = load_tokenizer(name, path); tok.padding_side = "left"
    model = load_llm(path, plan, offload_dir)
    base = model.base_model                              # transformer without the LM head: no logits needed
    c = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    L, D = c.num_hidden_layers, c.hidden_size
    I = getattr(c, "intermediate_size", None) or 4 * D
    H = getattr(c, "num_attention_heads", 1)
    n_params = sum(p.numel() for p in model.parameters())
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    # some tokenizers append an EOS token; the readout must be the template's own final token, so step back over it
    probe = tok("a b")["input_ids"]
    pos = -2 if (tok.eos_token_id is not None and len(probe) > 1 and probe[-1] == tok.eos_token_id) else -1
    out = {}
    for r in readouts:
        inp = [fill(PROMPTS[r], t) for t in texts]
        T = max(len(x) for x in tok(inp)["input_ids"])
        per_seq = T * ((L + 1) * D * 2 + 4 * I * 2 + 8 * D * 2) + H * T * T * 4
        # Offloaded models (65B/70B) also stream layer weights through the GPU; use half the activation budget there.
        budget = act_budget_gb if plan["mode"] == "gpu" else act_budget_gb / 2
        bs = max(1, min(batch_size, int(budget * 1e9 // per_seq)))
        print(f"  [{r}] {T} tokens max, ~{per_seq / 1e6:.0f} MB per caption -> batch size {bs}", flush=True)
        feats, last_ids = [], []
        for i in range(0, len(inp), bs):
            b = tok(inp[i:i + bs], padding="longest", return_tensors="pt")
            ids, am = b["input_ids"].to(dev), b["attention_mask"].to(dev)
            assert bool(am[:, -1].all()), "left padding expected"
            hs = base(input_ids=ids, attention_mask=am, output_hidden_states=True).hidden_states
            feats.append(torch.stack([h[:, pos].float().cpu() for h in hs], 1).to(torch.bfloat16))
            last_ids.append(ids[:, pos].cpu())
            del hs
        last_ids = torch.cat(last_ids)
        out[r] = dict(feats=torch.cat(feats), readout=r, template=PROMPTS[r], num_params=int(n_params),
                      precision="bf16", placement=plan["mode"], batch_size=bs, max_tokens=T,
                      last_token=tok.decode(last_ids[:1]), last_token_unique=int(last_ids.unique().numel()), readout_position=pos)
    del model, base; gc.collect(); torch.cuda.empty_cache()
    return out
