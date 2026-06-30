import os
import gc
import json
from pathlib import Path
from itertools import combinations

import torch
from transformers import AutoConfig, AutoModelForCausalLM

# -----------------------------
# Config
# -----------------------------
model_names = [
    "Qwen/Qwen2.5-3B-Instruct",
    "anonymous/AnonMT_Hindi_English",
    "anonymous/AnonMT_Bengali_English",
    "anonymous/AnonMT_Tamil_English",
    "anonymous/AnonMT_Telugu_English",
]

lang = "hi"
data_dir = "data_qwen_flores_indic_en"

N = 128
max_length = 512
pool = "mean"

use_cpu_offload = False
torch.set_grad_enabled(False)

# -----------------------------
# Device and dtype
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cuda" and torch.cuda.is_bf16_supported():
    dtype = torch.bfloat16
elif device == "cuda":
    dtype = torch.float16
else:
    dtype = torch.float32

print(f"device={device}, dtype={dtype}, N={N}, max_length={max_length}")

# -----------------------------
# Linear CKA utilities
# -----------------------------
def center(x: torch.Tensor) -> torch.Tensor:
    return x - x.mean(dim=0, keepdim=True)

@torch.no_grad()
def linear_cka(X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-12) -> float:
    X = center(X)
    Y = center(Y)

    xty = X.T @ Y
    num = (xty * xty).sum()

    xtx = X.T @ X
    yty = Y.T @ Y
    denom = torch.sqrt((xtx * xtx).sum() * (yty * yty).sum()).clamp_min(eps)

    return (num / denom).item()

# -----------------------------
# Collect layer representations
# -----------------------------
@torch.no_grad()
def collect_layer_reps_from_ids(model, input_ids, attention_mask):
    """
    Returns a list where reps[l] has shape [N, D]
    """
    model.eval()

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )

    reps = []
    hidden_states = out.hidden_states  # num_layers + 1

    for h in hidden_states:
        mask = attention_mask.unsqueeze(-1).to(h.dtype)
        summed = (h * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        rep = (summed / denom).detach().cpu()
        reps.append(rep)

    del out
    return reps

# -----------------------------
# Load data and build sequences
# -----------------------------
ids = torch.load(f"{data_dir}/id.{lang}.valid.nemo")

cfg = AutoConfig.from_pretrained(model_names[0], trust_remote_code=True)
seq_len = min(getattr(cfg, "max_position_embeddings", max_length), max_length)

total = (ids.numel() // seq_len) * seq_len
input_ids = ids[:total].reshape(-1, seq_len)[:N].contiguous()

attention_mask = (input_ids != 0).long()

print("Using sequences:", input_ids.size())

# -----------------------------
# Run models and cache representations
# -----------------------------
cache = {}

for name in model_names:
    print("\nRunning model:", name)

    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map="auto" if (device == "cuda" and use_cpu_offload) else None,
    )

    if not use_cpu_offload:
        model = model.to(device)

    reps = collect_layer_reps_from_ids(
        model,
        input_ids.to(device) if not use_cpu_offload else input_ids,
        attention_mask.to(device) if not use_cpu_offload else attention_mask,
    )

    cache[name] = reps

    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

# -----------------------------
# Pairwise layer-wise CKA + dump
# -----------------------------
output_dir = Path("cka_outputs")
output_dir.mkdir(parents=True, exist_ok=True)

results = {
    "language": lang,
    "num_samples": N,
    "seq_len": seq_len,
    "models": model_names,
    "pairs": {}
}

pairs = list(combinations(model_names, 2))

for a, b in pairs:
    reps_a = cache[a]
    reps_b = cache[b]
    L = min(len(reps_a), len(reps_b))

    pair_key = f"{a}|||{b}"
    results["pairs"][pair_key] = {}

    print("\nPAIR:", a, "vs", b)

    for layer in range(L):
        X = reps_a[layer].to(torch.float32)
        Y = reps_b[layer].to(torch.float32)

        cka = linear_cka(X, Y)
        results["pairs"][pair_key][f"layer_{layer:02d}"] = float(cka)

        print(f"  layer {layer:02d}: CKA={cka:.4f}")

# -----------------------------
# Save JSON
# -----------------------------
output_path = output_dir / f"{lang}.json"
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved CKA results to {output_path}")
