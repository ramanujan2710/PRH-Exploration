# ===================== E6 feature extraction: several readouts per model, every layer =====================
# Language: hidden states of every layer (embeddings + each block; the last entry is after the final norm, i.e. the
#           model's own output), read out per caption as
#             mean   = mean over non-pad tokens (Huh et al. / E2 protocol)
#             last   = last token (left padding, so position -1 is always the caption's last real token)
#             prompt = last token of a PromptEOL template (Jiang et al. 2023), a training-free decoder->encoder readout
#             cls    = first token (encoders only)
# Vision:   output of every transformer block, read out as cls (token 0, = E2 / Huh et al.) and mean over patch tokens
#           (prefix tokens excluded), timm transforms.
import gc, math, os, shutil
import torch, transformers

DEV = "cuda" if torch.cuda.is_available() else "cpu"
PROMPT_EOL = 'This sentence : "{}" means in one word:"'

def _dtype_kw(dtype):
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    return {"dtype": dtype} if (major, minor) >= (4, 56) else {"torch_dtype": dtype}

def hf_cache_clear(name):
    from huggingface_hub import constants
    shutil.rmtree(os.path.join(constants.HF_HUB_CACHE, "models--" + name.replace("/", "--")), ignore_errors=True)
    shutil.rmtree(getattr(constants, "HF_XET_CACHE", os.path.join(constants.HF_HOME, "xet")), ignore_errors=True)

@torch.no_grad()
def extract_text(name, texts, kind, readouts, batch_size=32, act_budget_gb=6.0):
    """kind: 'decoder' | 'embedding' (decoder-architecture embedding model, last-token pooling) | 'encoder' (BERT-like).
    Returns {readout: dict(feats=N x L x D bf16, ...)} from the SAME weights."""
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    left = kind in ("decoder", "embedding")
    if left:
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if left else torch.float32            # encoders (mpnet) run in fp32, as sentence-transformers does
    model = AutoModel.from_pretrained(name, **_dtype_kw(dtype)).to(DEV).eval()
    n_params = sum(p.numel() for p in model.parameters())
    c = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    L, D = c.num_hidden_layers, c.hidden_size
    out = {}
    groups = {"prompt": [r for r in readouts if r == "prompt"], "plain": [r for r in readouts if r != "prompt"]}
    for grp, rs in groups.items():
        if not rs: continue
        inp = [PROMPT_EOL.format(t) for t in texts] if grp == "prompt" else list(texts)
        lens = [len(x) for x in tok(inp, add_special_tokens=True)["input_ids"]]
        T = max(lens)
        per_seq = T * (L + 1) * D * 4 * 2                       # all hidden states, fp32 pooling copies
        bs = max(1, min(batch_size, int(act_budget_gb * 1e9 // per_seq)))
        acc = {r: [] for r in rs}
        for i in range(0, len(inp), bs):
            b = tok(inp[i:i + bs], padding="longest", return_tensors="pt").to(DEV)
            am = b["attention_mask"]
            if left:
                assert bool(am[:, -1].all()), "left padding expected: last position must be a real token"
            hs = model(**b, output_hidden_states=True).hidden_states   # tuple of L+1 tensors B x T x D
            w = am.unsqueeze(-1).float()
            for r in rs:
                if r == "mean":
                    v = [((h.float() * w).sum(1) / w.sum(1)) for h in hs]
                elif r in ("last", "prompt"):
                    v = [h[:, -1].float() for h in hs]
                elif r == "cls":
                    v = [h[:, 0].float() for h in hs]
                else:
                    raise ValueError(r)
                acc[r].append(torch.stack(v, 1).cpu().to(torch.bfloat16))   # B x (L+1) x D
            del hs
        for r in rs:
            out[r] = dict(feats=torch.cat(acc[r]), num_params=int(n_params), kind=kind, readout=r,
                          dtype=str(dtype), batch_size=bs, max_tokens=T)
    # Self-check for the checker: the last hidden_states entry equals the model's own output (last_hidden_state)
    b = tok(list(texts[:8]), padding="longest", return_tensors="pt").to(DEV)
    o = model(**b, output_hidden_states=True)
    same = bool(torch.equal(o.hidden_states[-1], o.last_hidden_state))
    for r in out: out[r]["final_is_model_output"] = same
    del model; gc.collect(); torch.cuda.empty_cache()
    return out

@torch.no_grad()
def extract_vision2(name, images, batch_size=64):
    """Returns {'cls': ..., 'mean': ...}, each N x L x D bf16 (one entry per transformer block)."""
    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
    model = timm.create_model(name, pretrained=True).eval().to(DEV)
    n_params = sum(p.numel() for p in model.parameters())
    tf = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    npre = int(getattr(model, "num_prefix_tokens", 1))
    store = {}
    def hook(k):
        def f(m, i, o):
            store[k] = (o[:, 0, :].float().cpu(), o[:, npre:, :].float().mean(1).cpu())
        return f
    hooks = [blk.register_forward_hook(hook(k)) for k, blk in enumerate(model.blocks)]
    cls, mean = [], []
    for i in range(0, len(images), batch_size):
        x = torch.stack([tf(im.convert("RGB")) for im in images[i:i + batch_size]]).to(DEV)
        store.clear(); model(x)
        cls.append(torch.stack([store[k][0] for k in range(len(model.blocks))], 1))
        mean.append(torch.stack([store[k][1] for k in range(len(model.blocks))], 1))
    for h in hooks: h.remove()
    del model; gc.collect(); torch.cuda.empty_cache()
    base = dict(num_params=int(n_params), num_prefix_tokens=npre)
    return {"cls": dict(feats=torch.cat(cls).to(torch.bfloat16), readout="cls", **base),
            "mean": dict(feats=torch.cat(mean).to(torch.bfloat16), readout="mean", **base)}
