# ===================== Feature extraction (Huh et al. protocol), exact bf16 =====================
# Language: every hidden state (embeddings + each block), mean-pooled over non-pad tokens,
#           left padding to the longest caption in the dataset, bf16 weights. NO quantization:
#           models larger than the GPU run with accelerate CPU/disk offload, which streams the bf16
#           weights to the GPU layer by layer, so the arithmetic is identical to an all-GPU run.
# Vision:   CLS token after every transformer block (= Huh's "blocks.{i}.add_1" node), timm transforms.
import gc, glob, math, os, shutil, time
import torch, transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

def _dtype_kw(dtype):
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    return {"dtype": dtype} if (major, minor) >= (4, 56) else {"torch_dtype": dtype}

def load_tokenizer(name, path=None):
    path = path or name
    try:
        tok = AutoTokenizer.from_pretrained(path)
    except Exception as e:                      # e.g. OpenLLaMA fast-tokenizer conversion issues
        print("  fast tokenizer failed, using slow:", repr(e)[:120])
        tok = AutoTokenizer.from_pretrained(path, use_fast=False)
    if "huggyllama" in name:
        tok.pad_token = tok.unk_token           # Huh et al. set "[PAD]" (not in vocab); padded positions are masked either way
    elif tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok

def weights_gb(name):
    """Exact checkpoint size from the Hub metadata (safetensors if present, else .bin)."""
    from huggingface_hub import HfApi
    info = HfApi().model_info(name, files_metadata=True)
    files = [(s.rfilename, s.size or 0) for s in info.siblings]
    st = [sz for f, sz in files if f.endswith(".safetensors")]
    bn = [sz for f, sz in files if f.endswith(".bin") and "training_args" not in f]
    return (sum(st) if st else sum(bn)) / 1e9, ("safetensors" if st else "bin")

def plan_placement(name, gpu_headroom_gb=14.0, cpu_headroom_gb=16.0, disk_headroom_gb=8.0):
    """Decide GPU / GPU+CPU / GPU+CPU+disk placement for exact bf16 loading, or a reason to skip.
    Memory budgets are computed in GiB (2**30 bytes), the unit accelerate's max_memory uses; disk in GB.
    14 GiB GPU headroom = 7 GiB activations (extract_llm sizes the batch to fit) + CUDA contexts, the layer
    being streamed in from CPU during offload, and allocator fragmentation. Offload is bit-exact
    (verified on the GPU by the smoke test), so erring towards offload only costs time."""
    import psutil
    GiB = 2 ** 30
    w, fmt = weights_gb(name)                                   # GB (1e9), from the Hub metadata
    w_gib = w * 1e9 / GiB
    gpu = (torch.cuda.get_device_properties(0).total_memory / GiB) if torch.cuda.is_available() else 0.0
    cpu = psutil.virtual_memory().available / GiB
    disk = shutil.disk_usage("/").free / 1e9
    plan = dict(weights_gb=round(w, 1), weights_gib=round(w_gib, 1), format=fmt, gpu_gib=round(gpu, 1),
                cpu_free_gib=round(cpu, 1), disk_free_gb=round(disk, 1))
    if disk < w + disk_headroom_gb:
        return dict(plan, mode="skip", reason=f"download needs {w + disk_headroom_gb:.0f} GB disk, {disk:.0f} GB free")
    gpu_budget, cpu_budget = max(gpu - gpu_headroom_gb, 0), max(cpu - cpu_headroom_gb, 0)
    if w_gib <= gpu_budget:
        return dict(plan, mode="gpu", max_memory=None)
    mm = {0: f"{int(gpu_budget)}GiB", "cpu": f"{int(cpu_budget)}GiB"}
    if w_gib <= int(gpu_budget) + int(cpu_budget):
        return dict(plan, mode="gpu+cpu", max_memory=mm)
    spill_gb = (w_gib - int(gpu_budget) - int(cpu_budget)) * GiB / 1e9
    if disk - w - disk_headroom_gb > spill_gb * 1.1:
        return dict(plan, mode="gpu+cpu+disk", max_memory=mm)
    return dict(plan, mode="skip", reason=f"needs {spill_gb:.0f} GB disk offload beyond the download; not enough disk")

MODEL_DIR = "/content/hf_models"

def local_model_dir(name):
    return os.path.join(MODEL_DIR, name.replace("/", "--"))

def download_llm(name):
    """Download exactly one copy of the weights (the format weights_gb counted) into a dedicated folder.
    Skips duplicate formats (Meta's original/*.pth, .msgpack, .h5), which would double the disk needed."""
    from huggingface_hub import snapshot_download
    _, fmt = weights_gb(name)
    weights = ["*.safetensors"] if fmt == "safetensors" else ["*.bin"]
    return snapshot_download(name, local_dir=local_model_dir(name),
                             allow_patterns=["*.json", "*.model", "*.tiktoken", "*.txt"] + weights,
                             ignore_patterns=["original/*", "*.pth", "*.pt", "*.msgpack", "*.h5", "*.onnx",
                                              "training_args.bin"])

def load_llm(path, plan, offload_dir):
    kw = dict(device_map="auto", **_dtype_kw(torch.bfloat16))
    if plan["mode"] != "gpu":
        # JSON turns the GPU key 0 into "0"; accelerate needs the integer device index
        kw["max_memory"] = {(int(k) if str(k).isdigit() else k): v for k, v in plan["max_memory"].items()}
    if plan["mode"] == "gpu+cpu+disk":
        os.makedirs(offload_dir, exist_ok=True); kw["offload_folder"] = offload_dir
    return AutoModelForCausalLM.from_pretrained(path, **kw).eval()

@torch.no_grad()
def extract_llm(name, texts, plan, offload_dir="/content/offload", batch_size=32, act_budget_gb=7.0):
    path = download_llm(name)                   # one explicit folder per model, deleted by the notebook afterwards
    tok = load_tokenizer(name, path)
    model = load_llm(path, plan, offload_dir)
    n_params = sum(p.numel() for p in model.parameters())
    tokens = tok(texts, padding="longest", return_tensors="pt")
    T = tokens["input_ids"].shape[1]; V = model.get_output_embeddings().weight.shape[0]
    # bf16 logits + their fp32 copy for the loss: 6 bytes per (token, vocab) entry; keep under the budget
    # Batch size from the activation memory of THIS model, so it fits in the GPU headroom left by the weights.
    # Per token: hidden states of every layer (kept until the forward ends because output_hidden_states=True),
    # logits in bf16 + their fp32 copy for the loss, MLP intermediates, residual/attention scratch; per sequence:
    # attention scores (fp32 upper bound). The batch size never changes the per-caption math (no batch statistics).
    c = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    L, D = c.num_hidden_layers, c.hidden_size
    I = getattr(c, "intermediate_size", None) or 4 * D
    H = getattr(c, "num_attention_heads", 1)
    per_seq = T * ((L + 1) * D * 2 + V * 6 + 4 * I * 2 + 8 * D * 2) + H * T * T * 4
    bs = max(1, min(batch_size, int(act_budget_gb * 1e9 // per_seq)))
    print(f"  activation estimate {per_seq / 1e6:.0f} MB per caption -> batch size {bs}", flush=True)
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"   # accelerate hooks move inputs to each module's device
    feats, losses, bpbs = [], [], []
    crit = torch.nn.CrossEntropyLoss(reduction="none")
    for i in range(0, len(texts), bs):
        ids = tokens["input_ids"][i:i + bs].to(dev).long()
        am = tokens["attention_mask"][i:i + bs].to(dev).long()
        out = model(input_ids=ids, attention_mask=am, output_hidden_states=True)
        m = am[:, :-1].float()
        logits = out.logits[:, :-1].float().to(m.device)
        tok_loss = m * crit(logits.permute(0, 2, 1), ids[:, 1:].to(m.device))
        losses.append((tok_loss.sum(-1) / m.sum(-1)).cpu())
        nbytes = torch.tensor([len(t.encode("utf-8")) for t in texts[i:i + bs]], dtype=torch.float32)
        bpbs.append((tok_loss.sum(-1).cpu() * math.log2(math.e)) / nbytes)
        pooled = []
        for h in out.hidden_states:
            w = am.to(h.device).unsqueeze(-1).to(h.dtype)
            pooled.append(((h * w).sum(1) / w.sum(1)).float().cpu())                   # per layer: B x D
        feats.append(torch.stack(pooled, 1).to(torch.bfloat16))                         # B x L x D
        del out, logits
    res = dict(feats=torch.cat(feats), num_params=int(n_params),
               loss=torch.cat(losses).mean().item(), bpb=torch.cat(bpbs).mean().item(),
               precision="bf16", placement=plan["mode"], weights_gb=plan["weights_gb"], batch_size=bs, seq_len=T)
    del model; gc.collect(); torch.cuda.empty_cache()
    shutil.rmtree(offload_dir, ignore_errors=True)
    return res

@torch.no_grad()
def extract_vision(name, images, batch_size=64, pretrained=True):
    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
    model = timm.create_model(name, pretrained=pretrained).eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"; model = model.to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    tf = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    store = {}
    hooks = [blk.register_forward_hook(lambda m, i, o, k=k: store.__setitem__(k, o[:, 0, :].float().cpu()))
             for k, blk in enumerate(model.blocks)]
    feats = []
    for i in range(0, len(images), batch_size):
        x = torch.stack([tf(im.convert("RGB")) for im in images[i:i + batch_size]]).to(dev)
        store.clear(); model(x)
        feats.append(torch.stack([store[k] for k in range(len(model.blocks))], 1))
    for h in hooks: h.remove()
    res = dict(feats=torch.cat(feats).to(torch.bfloat16), num_params=int(n_params))
    del model; gc.collect(); torch.cuda.empty_cache()
    return res

def clear_hf_cache(name=None):
    """Delete downloaded LLM weights: the model's own folder (or all of them if name is None), any copy in the
    Hugging Face hub cache (located via huggingface_hub itself, wherever HF_HOME points), and the xet chunk cache."""
    from huggingface_hub import constants
    hub = constants.HF_HUB_CACHE
    if name is None:
        shutil.rmtree(MODEL_DIR, ignore_errors=True)
        for d in glob.glob(os.path.join(hub, "models--*")):
            shutil.rmtree(d, ignore_errors=True)
    else:
        shutil.rmtree(local_model_dir(name), ignore_errors=True)
        shutil.rmtree(os.path.join(hub, "models--" + name.replace("/", "--")), ignore_errors=True)
    shutil.rmtree(getattr(constants, "HF_XET_CACHE", os.path.join(constants.HF_HOME, "xet")), ignore_errors=True)
