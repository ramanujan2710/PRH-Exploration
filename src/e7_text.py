# ===================== E7: decoder-LLM caption embeddings in Rosetta's file format =====================
# Readouts of the FINAL layer (the model's own output; unpaired alignment cannot choose a layer with pairs):
#   avg = mean over the caption's tokens (plain text, no template)
#   eol = last token of 'This sentence : "<caption>" means in one word:"' (PromptEOL, as in E2-prompt / E6)
# Positions count only real tokens (left padding), as in Rosetta's own MeanPooling, so padding never changes a text's
# embedding. File: [items, texts, d] bfloat16, NaN-padded, rows in Rosetta's corpus order (modalities.image_captions
# .datasets.texts), written to <storage>/embeddings/<dataset>/language/<name>.pt.
import gc, os, torch
from transformers import AutoModel, AutoTokenizer

PROMPT_EOL = 'This sentence : "{}" means in one word:"'

def _dtype_kw():
    import transformers
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    return {"dtype": torch.bfloat16} if (major, minor) >= (4, 56) else {"torch_dtype": torch.bfloat16}

@torch.no_grad()
def embed_corpus(checkpoint, corpus_texts, readouts, token_budget=32768, max_batch=256, device="cuda"):
    """corpus_texts: list (items) of lists (captions). Returns {readout: Tensor[items, T, d] bf16 NaN-padded}."""
    tok = AutoTokenizer.from_pretrained(checkpoint); tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(checkpoint, **_dtype_kw()).to(device).eval()
    probe = tok("a b")["input_ids"]
    pos = -2 if (tok.eos_token_id is not None and len(probe) > 1 and probe[-1] == tok.eos_token_id) else -1
    flat = [(i, j, t) for i, caps in enumerate(corpus_texts) for j, t in enumerate(caps)]
    T = max(len(c) for c in corpus_texts); D = model.config.hidden_size
    out = {}
    for r in readouts:
        inputs = [PROMPT_EOL.format(t) if r == "eol" else t for _, _, t in flat]
        ids = tok(inputs, add_special_tokens=True)["input_ids"]
        order = sorted(range(len(ids)), key=lambda k: len(ids[k]))           # length-sorted batches: little padding
        res = torch.full((len(corpus_texts), T, D), float("nan"), dtype=torch.bfloat16)
        k = 0
        while k < len(order):
            L = len(ids[order[min(k + max_batch, len(order)) - 1]])
            n = max(1, min(max_batch, token_budget // max(L, 1)))
            batch = order[k:k + n]; k += n
            b = tok.pad({"input_ids": [ids[q] for q in batch]}, return_tensors="pt").to(device)
            am = b["attention_mask"]
            positions = (am.long().cumsum(-1) - 1).masked_fill(am == 0, 1)
            h = model(input_ids=b["input_ids"], attention_mask=am, position_ids=positions).last_hidden_state
            if r == "avg":
                w = am.unsqueeze(-1).float(); v = (h.float() * w).sum(1) / w.sum(1)
            else:
                assert bool(am[:, pos].all()), "left padding expected"
                v = h[:, pos].float()
            v = v.cpu().to(torch.bfloat16)
            for row, q in enumerate(batch):
                i, j, _ = flat[q]; res[i, j] = v[row]
        out[r] = res
        last = tok.decode([tok(PROMPT_EOL.format("x"))["input_ids"][pos]]) if r == "eol" else None
        print(f"  {checkpoint} [{r}]: {tuple(res.shape)}; eol final token = {last!r}", flush=True)
    del model; gc.collect(); torch.cuda.empty_cache()
    return out

def write_file(storage, dataset, name, tensor):
    path = os.path.join(storage, "embeddings", dataset, "language", f"{name}.pt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(tensor, path + ".tmp"); os.replace(path + ".tmp", path)
    return path
