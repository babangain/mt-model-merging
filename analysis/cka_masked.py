# #!/usr/bin/env python3
# """
# Masked (span-restricted) layerwise Linear CKA for Qwen-style MT models.

# This script expects the data produced by your masking script:
#   - input_ids.{lang}.pt   [N, T]  (padded to seq_len)
#   - src_mask.{lang}.pt    [N, T]  bool mask for Indic sentence tokens
#   - tgt_mask.{lang}.pt    [N, T]  bool mask for English target tokens
#   - lengths.{lang}.pt     [N]     (optional, not required)

# It computes per-layer pooled representations using ONLY the masked tokens (src or tgt),
# then computes pairwise layerwise Linear CKA across models.


# python cka_masked.py \
#   --lang hi \
#   --data_dir data_masked_qwen/indic_en \
#   --tokenizer_name Qwen/Qwen2.5-3B-Instruct \
#   --models \
#     Qwen/Qwen2.5-3B-Instruct \
#     anonymous/AnonMT_Hindi_English \
#     anonymous/AnonMT_Bengali_English \
#     anonymous/AnonMT_Tamil_English \
#     anonymous/AnonMT_Telugu_English \
#   --N 128 \
#   --max_length 512 \
#   --span both \
#   --out_dir cka_outputs_masked

# """

# import os
# import gc
# import json
# import argparse
# from pathlib import Path
# from itertools import combinations
# from typing import Dict, List

# import torch
# from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM


# # -----------------------------
# # Linear CKA utilities
# # -----------------------------
# def _center(x: torch.Tensor) -> torch.Tensor:
#     return x - x.mean(dim=0, keepdim=True)


# @torch.no_grad()
# def linear_cka(X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-12) -> float:
#     """
#     Linear CKA between X and Y, where:
#       X: [N, D], Y: [N, D]
#     """
#     X = _center(X)
#     Y = _center(Y)

#     xty = X.T @ Y
#     num = (xty * xty).sum()

#     xtx = X.T @ X
#     yty = Y.T @ Y
#     denom = torch.sqrt((xtx * xtx).sum() * (yty * yty).sum()).clamp_min(eps)

#     return (num / denom).item()


# # -----------------------------
# # Masked pooling over tokens
# # -----------------------------
# @torch.no_grad()
# def collect_layer_reps_masked(
#     model,
#     input_ids: torch.Tensor,          # [N, T]
#     attention_mask: torch.Tensor,     # [N, T] (1 for real tokens)
#     token_mask: torch.Tensor,         # [N, T] (bool, span tokens)
# ) -> List[torch.Tensor]:
#     """
#     Returns reps: list length (num_layers + 1),
#     where reps[l] is [N, D], pooled by mean over tokens where (attention_mask & token_mask) is True.
#     """
#     model.eval()

#     out = model(
#         input_ids=input_ids,
#         attention_mask=attention_mask,
#         output_hidden_states=True,
#         use_cache=False,
#     )

#     hidden_states = out.hidden_states  # tuple: (embeddings, layer1, ..., layerL)
#     reps: List[torch.Tensor] = []

#     # Build final pooling mask once
#     pool_mask = (attention_mask.to(torch.bool) & token_mask.to(torch.bool))  # [N, T]
#     denom = pool_mask.sum(dim=1)  # [N]
#     if (denom == 0).any():
#         # This should not happen if we filter valid rows before calling this function.
#         # Still keep it safe.
#         denom = denom.clamp_min(1)

#     for h in hidden_states:
#         # h: [N, T, D]
#         m = pool_mask.unsqueeze(-1).to(h.dtype)  # [N, T, 1]
#         summed = (h * m).sum(dim=1)              # [N, D]
#         rep = (summed / denom.unsqueeze(-1).to(h.dtype)).detach().cpu()
#         reps.append(rep)

#     del out
#     return reps


# def pick_dtype(device: str) -> torch.dtype:
#     if device == "cuda" and torch.cuda.is_bf16_supported():
#         return torch.bfloat16
#     if device == "cuda":
#         return torch.float16
#     return torch.float32


# def load_masked_tensors(data_dir: str, lang: str):
#     input_ids = torch.load(os.path.join(data_dir, f"input_ids.{lang}.pt"))
#     src_mask = torch.load(os.path.join(data_dir, f"src_mask.{lang}.pt"))
#     tgt_mask = torch.load(os.path.join(data_dir, f"tgt_mask.{lang}.pt"))
#     lengths_path = os.path.join(data_dir, f"lengths.{lang}.pt")
#     lengths = torch.load(lengths_path) if os.path.exists(lengths_path) else None
#     return input_ids, src_mask, tgt_mask, lengths


# def filter_and_truncate(
#     input_ids: torch.Tensor,
#     span_mask: torch.Tensor,
#     pad_id: int,
#     N: int,
#     max_length: int,
# ) -> Dict[str, torch.Tensor]:
#     """
#     Applies:
#       - truncate to max_length
#       - build attention_mask from pad_id
#       - filter rows with at least 1 masked token inside attention_mask
#       - keep up to N rows
#     """
#     input_ids = input_ids[:, :max_length].contiguous()
#     span_mask = span_mask[:, :max_length].contiguous()

#     attention_mask = (input_ids != pad_id).long()

#     pool_mask = (attention_mask.to(torch.bool) & span_mask.to(torch.bool))
#     valid = pool_mask.sum(dim=1) > 0
#     input_ids = input_ids[valid]
#     span_mask = span_mask[valid]
#     attention_mask = attention_mask[valid]

#     if input_ids.size(0) == 0:
#         raise RuntimeError("No valid rows after filtering for this span mask.")
#     if input_ids.size(0) < 2:
#         raise RuntimeError(f"Too few valid rows ({input_ids.size(0)}) for CKA. Need at least 2.")

#     input_ids = input_ids[:N].contiguous()
#     span_mask = span_mask[:N].contiguous()
#     attention_mask = attention_mask[:N].contiguous()

#     return {
#         "input_ids": input_ids,
#         "span_mask": span_mask,
#         "attention_mask": attention_mask,
#     }


# def run_span(
#     span_name: str,
#     model_names: List[str],
#     tokenizer_name: str,
#     device: str,
#     dtype: torch.dtype,
#     use_cpu_offload: bool,
#     input_ids: torch.Tensor,
#     attention_mask: torch.Tensor,
#     span_mask: torch.Tensor,
# ) -> Dict:
#     """
#     Runs all models, caches masked reps, then computes pairwise layerwise CKA.
#     """
#     cache: Dict[str, List[torch.Tensor]] = {}

#     for name in model_names:
#         print(f"\n[{span_name}] Running model: {name}")

#         model = AutoModelForCausalLM.from_pretrained(
#             name,
#             torch_dtype=dtype,
#             trust_remote_code=True,
#             low_cpu_mem_usage=True,
#             device_map="auto" if (device == "cuda" and use_cpu_offload) else None,
#         )

#         if not use_cpu_offload:
#             model = model.to(device)

#         reps = collect_layer_reps_masked(
#             model,
#             input_ids.to(device) if not use_cpu_offload else input_ids,
#             attention_mask.to(device) if not use_cpu_offload else attention_mask,
#             span_mask.to(device) if not use_cpu_offload else span_mask,
#         )
#         cache[name] = reps

#         del model
#         gc.collect()
#         if device == "cuda":
#             torch.cuda.empty_cache()

#     results_span = {"pairs": {}}
#     pairs = list(combinations(model_names, 2))

#     for a, b in pairs:
#         reps_a = cache[a]
#         reps_b = cache[b]
#         L = min(len(reps_a), len(reps_b))

#         pair_key = f"{a}|||{b}"
#         results_span["pairs"][pair_key] = {}

#         print(f"\n[{span_name}] PAIR: {a} vs {b}")

#         for layer in range(L):
#             X = reps_a[layer].to(torch.float32)
#             Y = reps_b[layer].to(torch.float32)

#             cka = linear_cka(X, Y)
#             results_span["pairs"][pair_key][f"layer_{layer:02d}"] = float(cka)
#             print(f"  layer {layer:02d}: CKA={cka:.4f}")

#     return results_span


# def main() -> None:
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--lang", type=str, required=True, help="hi | bn | ta | te")
#     ap.add_argument("--data_dir", type=str, required=True, help="e.g. data_masked_qwen/indic_en")
#     ap.add_argument("--tokenizer_name", type=str, required=True, help="e.g. Qwen/Qwen2.5-3B-Instruct")
#     ap.add_argument(
#         "--models",
#         type=str,
#         nargs="+",
#         required=True,
#         help="List of model names (space separated)",
#     )
#     ap.add_argument("--N", type=int, default=128)
#     ap.add_argument("--max_length", type=int, default=512)
#     ap.add_argument("--span", type=str, default="both", choices=["src", "tgt", "both"])
#     ap.add_argument("--out_dir", type=str, default="cka_outputs_masked")
#     ap.add_argument("--use_cpu_offload", action="store_true")
#     args = ap.parse_args()

#     torch.set_grad_enabled(False)

#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     dtype = pick_dtype(device)

#     print(f"device={device}, dtype={dtype}, N={args.N}, max_length={args.max_length}, span={args.span}")

#     tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True, trust_remote_code=True)
#     if tokenizer.pad_token_id is None:
#         tokenizer.pad_token = tokenizer.eos_token
#     pad_id = int(tokenizer.pad_token_id)

#     input_ids_all, src_mask_all, tgt_mask_all, _lengths = load_masked_tensors(args.data_dir, args.lang)

#     cfg = AutoConfig.from_pretrained(args.models[0], trust_remote_code=True)
#     model_max_pos = int(getattr(cfg, "max_position_embeddings", args.max_length))
#     max_length = min(args.max_length, model_max_pos)
#     if max_length != args.max_length:
#         print(f"Note: clipped max_length to model max_position_embeddings = {max_length}")

#     out_dir = Path(args.out_dir)
#     out_dir.mkdir(parents=True, exist_ok=True)

#     base_meta = {
#         "language": args.lang,
#         "num_samples_requested": args.N,
#         "max_length": max_length,
#         "pad_token_id": pad_id,
#         "models": args.models,
#         "tokenizer_name": args.tokenizer_name,
#         "device": device,
#         "dtype": str(dtype),
#     }

#     results = {"meta": base_meta, "spans": {}}

#     if args.span in ("src", "both"):
#         pack = filter_and_truncate(
#             input_ids=input_ids_all,
#             span_mask=src_mask_all,
#             pad_id=pad_id,
#             N=args.N,
#             max_length=max_length,
#         )
#         print(f"\n[src] Using sequences: {pack['input_ids'].size()} (after filtering)")
#         results["spans"]["src"] = run_span(
#             span_name="src",
#             model_names=args.models,
#             tokenizer_name=args.tokenizer_name,
#             device=device,
#             dtype=dtype,
#             use_cpu_offload=args.use_cpu_offload,
#             input_ids=pack["input_ids"],
#             attention_mask=pack["attention_mask"],
#             span_mask=pack["span_mask"],
#         )
#         results["spans"]["src"]["num_samples_used"] = int(pack["input_ids"].size(0))

#     if args.span in ("tgt", "both"):
#         pack = filter_and_truncate(
#             input_ids=input_ids_all,
#             span_mask=tgt_mask_all,
#             pad_id=pad_id,
#             N=args.N,
#             max_length=max_length,
#         )
#         print(f"\n[tgt] Using sequences: {pack['input_ids'].size()} (after filtering)")
#         results["spans"]["tgt"] = run_span(
#             span_name="tgt",
#             model_names=args.models,
#             tokenizer_name=args.tokenizer_name,
#             device=device,
#             dtype=dtype,
#             use_cpu_offload=args.use_cpu_offload,
#             input_ids=pack["input_ids"],
#             attention_mask=pack["attention_mask"],
#             span_mask=pack["span_mask"],
#         )
#         results["spans"]["tgt"]["num_samples_used"] = int(pack["input_ids"].size(0))

#     out_path = out_dir / f"{args.lang}.masked_cka.json"
#     with open(out_path, "w") as f:
#         json.dump(results, f, indent=2)

#     print(f"\nSaved masked CKA results to {out_path}")


# if __name__ == "__main__":
#     main()



























#!/usr/bin/env python3
"""
Masked (span-restricted) layerwise Linear CKA for Qwen-style MT models (LOWER GPU MEM).

What changed vs your version (to reduce GPU usage):
- No `output_hidden_states=True` (that keeps ALL layer activations alive).
- Uses forward hooks to pool each layer output on-the-fly, immediately moves pooled reps to CPU.
- Runs in micro-batches (`--batch_size`) to reduce peak activation memory.
- Stores only [N, D] per layer on CPU (float32), not [N, T, D] per layer on GPU.

Expected files (produced by your masking script):
  - input_ids.{lang}.pt   [N, T]
  - src_mask.{lang}.pt    [N, T] bool
  - tgt_mask.{lang}.pt    [N, T] bool
  - lengths.{lang}.pt     [N] (optional)

Example:
python cka_masked.py \
  --lang te \
  --data_dir data_masked_qwen/indic_en \
  --tokenizer_name Qwen/Qwen2.5-3B-Instruct \
  --models \
    Qwen/Qwen2.5-3B-Instruct \
    anonymous/AnonMT_Hindi_English \
    anonymous/AnonMT_Bengali_English \
    anonymous/AnonMT_Tamil_English \
    anonymous/AnonMT_Telugu_English \
  --N 128 \
  --max_length 512 \
  --span both \
  --batch_size 8 \
  --out_dir cka_outputs_masked


python cka_masked.py \
  --lang te \
  --data_dir data_masked_qwen/en_indic \
  --tokenizer_name Qwen/Qwen2.5-3B-Instruct \
  --models \
    Qwen/Qwen2.5-3B-Instruct \
    anonymous/AnonMT_English_Hindi \
    anonymous/AnonMT_English_Bengali \
    anonymous/AnonMT_English_Tamil \
    anonymous/AnonMT_English_Telugu \
  --N 128 \
  --max_length 512 \
  --span both \
  --batch_size 8 \
  --out_dir cka_outputs_masked_en_indic
"""

import os
import gc
import json
import argparse
from pathlib import Path
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM


# -----------------------------
# Linear CKA utilities
# -----------------------------
def _center(x: torch.Tensor) -> torch.Tensor:
    return x - x.mean(dim=0, keepdim=True)


@torch.no_grad()
def linear_cka(X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-12) -> float:
    """
    Linear CKA between X and Y, where:
      X: [N, D], Y: [N, D]
    """
    X = _center(X)
    Y = _center(Y)

    xty = X.T @ Y
    num = (xty * xty).sum()

    xtx = X.T @ X
    yty = Y.T @ Y
    denom = torch.sqrt((xtx * xtx).sum() * (yty * yty).sum()).clamp_min(eps)

    return (num / denom).item()


def pick_dtype(device: str) -> torch.dtype:
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device == "cuda":
        return torch.float16
    return torch.float32


# -----------------------------
# I/O: masked tensors
# -----------------------------
def load_masked_tensors(data_dir: str, lang: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    input_ids = torch.load(os.path.join(data_dir, f"input_ids.{lang}.pt"))
    src_mask = torch.load(os.path.join(data_dir, f"src_mask.{lang}.pt"))
    tgt_mask = torch.load(os.path.join(data_dir, f"tgt_mask.{lang}.pt"))
    lengths_path = os.path.join(data_dir, f"lengths.{lang}.pt")
    lengths = torch.load(lengths_path) if os.path.exists(lengths_path) else None
    return input_ids, src_mask, tgt_mask, lengths


def filter_and_truncate(
    input_ids: torch.Tensor,
    span_mask: torch.Tensor,
    pad_id: int,
    N: int,
    max_length: int,
) -> Dict[str, torch.Tensor]:
    """
    Applies:
      - truncate to max_length
      - build attention_mask from pad_id
      - filter rows with at least 1 masked token inside attention_mask
      - keep up to N rows
    """
    input_ids = input_ids[:, :max_length].contiguous()
    span_mask = span_mask[:, :max_length].contiguous()

    attention_mask = (input_ids != pad_id).long()

    pool_mask = (attention_mask.to(torch.bool) & span_mask.to(torch.bool))
    valid = pool_mask.sum(dim=1) > 0
    input_ids = input_ids[valid]
    span_mask = span_mask[valid]
    attention_mask = attention_mask[valid]

    if input_ids.size(0) == 0:
        raise RuntimeError("No valid rows after filtering for this span mask.")
    if input_ids.size(0) < 2:
        raise RuntimeError(f"Too few valid rows ({input_ids.size(0)}) for CKA. Need at least 2.")

    input_ids = input_ids[:N].contiguous()
    span_mask = span_mask[:N].contiguous()
    attention_mask = attention_mask[:N].contiguous()

    return {
        "input_ids": input_ids,
        "span_mask": span_mask,
        "attention_mask": attention_mask,
    }


# -----------------------------
# Low-memory layerwise masked pooling via hooks
# -----------------------------
class LayerwiseMaskedPooler:
    """
    Collects pooled reps [B, D] for:
      - embedding output (as "layer_00" conceptually)
      - each transformer block output

    It pools mean over tokens where (attention_mask & token_mask) is True, per sample.
    Pooled reps are immediately moved to CPU float32 to keep GPU mem low.
    """

    def __init__(self):
        self.handles = []
        self._active = False

        # Set at runtime per batch
        self.pool_mask: Optional[torch.Tensor] = None   # [B, T] bool
        self.denom: Optional[torch.Tensor] = None       # [B] int64
        self.layer_cpu_chunks: Optional[List[List[torch.Tensor]]] = None  # list over layers, each list over batches

    def _pool_and_store(self, h: torch.Tensor, layer_idx: int) -> None:
        # h: [B, T, D]
        assert self.pool_mask is not None
        assert self.denom is not None
        assert self.layer_cpu_chunks is not None

        pm = self.pool_mask.unsqueeze(-1).to(dtype=h.dtype)  # [B, T, 1]
        summed = (h * pm).sum(dim=1)                         # [B, D]
        rep = (summed / self.denom.unsqueeze(-1).to(dtype=h.dtype))  # [B, D]

        # Move only [B, D] to CPU as float32
        self.layer_cpu_chunks[layer_idx].append(rep.detach().to("cpu", dtype=torch.float32))

    def _hook_factory(self, layer_idx: int):
        def hook(_module, _inputs, output):
            # Some blocks return tuple (hidden, ...) in HF
            h = output[0] if isinstance(output, (tuple, list)) else output
            if not torch.is_tensor(h):
                return
            # Expect [B, T, D]
            if h.dim() != 3:
                return
            self._pool_and_store(h, layer_idx)
        return hook

    def attach(self, model) -> int:
        """
        Attach hooks. Returns num_layers_total = 1 + num_transformer_blocks
        """
        if self._active:
            raise RuntimeError("Pooler already active/attached.")

        # Qwen-style in transformers: model.model.embed_tokens, model.model.layers
        if not hasattr(model, "model"):
            raise RuntimeError("Unexpected model structure: missing `.model`")

        core = model.model
        if not hasattr(core, "embed_tokens"):
            raise RuntimeError("Unexpected model structure: missing `model.model.embed_tokens`")
        if not hasattr(core, "layers"):
            raise RuntimeError("Unexpected model structure: missing `model.model.layers`")

        layers = list(core.layers)
        num_blocks = len(layers)

        # We treat embeddings as layer 0, blocks as 1..num_blocks
        total_layers = 1 + num_blocks
        self.layer_cpu_chunks = [[] for _ in range(total_layers)]

        # Hook embeddings output
        self.handles.append(core.embed_tokens.register_forward_hook(self._hook_factory(layer_idx=0)))

        # Hook each transformer block output
        for i, blk in enumerate(layers, start=1):
            self.handles.append(blk.register_forward_hook(self._hook_factory(layer_idx=i)))

        self._active = True
        return total_layers

    def set_batch_masks(self, attention_mask: torch.Tensor, token_mask: torch.Tensor) -> None:
        # both [B, T]
        pool_mask = (attention_mask.to(torch.bool) & token_mask.to(torch.bool))
        denom = pool_mask.sum(dim=1)  # [B]
        # Safety
        denom = denom.clamp_min(1)
        self.pool_mask = pool_mask
        self.denom = denom

    def detach(self) -> None:
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles = []
        self._active = False
        self.pool_mask = None
        self.denom = None

    def finalize(self) -> List[torch.Tensor]:
        """
        Returns reps: list length total_layers, each is [N, D] on CPU float32
        """
        assert self.layer_cpu_chunks is not None
        reps: List[torch.Tensor] = []
        for chunks in self.layer_cpu_chunks:
            if len(chunks) == 0:
                raise RuntimeError("No chunks collected for a layer. Check hooks/model structure.")
            reps.append(torch.cat(chunks, dim=0).contiguous())
        return reps


@torch.no_grad()
def collect_layer_reps_masked_lowmem(
    model,
    input_ids: torch.Tensor,          # [N, T] CPU
    attention_mask: torch.Tensor,     # [N, T] CPU
    token_mask: torch.Tensor,         # [N, T] CPU bool
    device: str,
    batch_size: int,
) -> List[torch.Tensor]:
    """
    Returns reps: list length (num_layers + 1),
    where reps[l] is [N, D], pooled by mean over masked tokens.
    Low GPU memory: uses hooks and micro-batches, avoids output_hidden_states.
    """
    model.eval()

    pooler = LayerwiseMaskedPooler()
    total_layers = pooler.attach(model)

    N = input_ids.size(0)

    try:
        for start in range(0, N, batch_size):
            end = min(N, start + batch_size)

            ids_b = input_ids[start:end]
            att_b = attention_mask[start:end]
            msk_b = token_mask[start:end]

            # Send only batch to GPU
            if device == "cuda":
                ids_b = ids_b.to(device, non_blocking=True)
                att_b = att_b.to(device, non_blocking=True)
                msk_b = msk_b.to(device, non_blocking=True)
            else:
                # keep on CPU
                pass

            pooler.set_batch_masks(attention_mask=att_b, token_mask=msk_b)

            # forward: hooks capture outputs and pool
            _ = model(
                input_ids=ids_b,
                attention_mask=att_b,
                use_cache=False,
                output_hidden_states=False,
                output_attentions=False,
                return_dict=False,
            )

            # free batch tensors sooner
            del ids_b, att_b, msk_b, _
            if device == "cuda":
                torch.cuda.empty_cache()

        reps = pooler.finalize()
        if len(reps) != total_layers:
            raise RuntimeError(f"Hook layers mismatch: collected {len(reps)} vs expected {total_layers}")
        return reps

    finally:
        pooler.detach()


def _maybe_set_pad_token(tokenizer) -> int:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return int(tokenizer.pad_token_id)


def load_model(
    name: str,
    dtype: torch.dtype,
    device: str,
    use_cpu_offload: bool,
):
    # device_map="auto" can reduce GPU but may slow down; still useful on tight VRAM.
    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map="auto" if (device == "cuda" and use_cpu_offload) else None,
    )
    if not use_cpu_offload:
        model = model.to(device)
    model.eval()
    return model


def run_span(
    span_name: str,
    model_names: List[str],
    tokenizer_name: str,
    device: str,
    dtype: torch.dtype,
    use_cpu_offload: bool,
    input_ids: torch.Tensor,          # CPU
    attention_mask: torch.Tensor,     # CPU
    span_mask: torch.Tensor,          # CPU bool
    batch_size: int,
) -> Dict:
    """
    Runs all models, collects low-mem masked reps on CPU, then computes pairwise layerwise CKA.
    """
    cache: Dict[str, List[torch.Tensor]] = {}

    for name in model_names:
        print(f"\n[{span_name}] Running model: {name}")

        model = load_model(name=name, dtype=dtype, device=device, use_cpu_offload=use_cpu_offload)

        reps = collect_layer_reps_masked_lowmem(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_mask=span_mask,
            device=device if not use_cpu_offload else ("cuda" if torch.cuda.is_available() else "cpu"),
            batch_size=batch_size,
        )
        cache[name] = reps

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results_span = {"pairs": {}}
    pairs = list(combinations(model_names, 2))

    for a, b in pairs:
        reps_a = cache[a]
        reps_b = cache[b]
        L = min(len(reps_a), len(reps_b))

        pair_key = f"{a}|||{b}"
        results_span["pairs"][pair_key] = {}

        print(f"\n[{span_name}] PAIR: {a} vs {b}")

        for layer in range(L):
            # Already CPU float32
            X = reps_a[layer]
            Y = reps_b[layer]
            cka = linear_cka(X, Y)
            results_span["pairs"][pair_key][f"layer_{layer:02d}"] = float(cka)
            print(f"  layer {layer:02d}: CKA={cka:.4f}")

    return results_span


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", type=str, required=True, help="hi | bn | ta | te")
    ap.add_argument("--data_dir", type=str, required=True, help="e.g. data_masked_qwen/indic_en")
    ap.add_argument("--tokenizer_name", type=str, required=True, help="e.g. Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--models", type=str, nargs="+", required=True, help="List of model names")
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--span", type=str, default="both", choices=["src", "tgt", "both"])
    ap.add_argument("--batch_size", type=int, default=8, help="Micro-batch size for lower GPU memory")
    ap.add_argument("--out_dir", type=str, default="cka_outputs_masked")
    ap.add_argument("--use_cpu_offload", action="store_true", help="Use HF device_map=auto offload (CUDA only)")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = pick_dtype(device)

    print(
        f"device={device}, dtype={dtype}, N={args.N}, max_length={args.max_length}, "
        f"span={args.span}, batch_size={args.batch_size}, cpu_offload={args.use_cpu_offload}"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True, trust_remote_code=True)
    pad_id = _maybe_set_pad_token(tokenizer)

    input_ids_all, src_mask_all, tgt_mask_all, _lengths = load_masked_tensors(args.data_dir, args.lang)

    cfg = AutoConfig.from_pretrained(args.models[0], trust_remote_code=True)
    model_max_pos = int(getattr(cfg, "max_position_embeddings", args.max_length))
    max_length = min(args.max_length, model_max_pos)
    if max_length != args.max_length:
        print(f"Note: clipped max_length to model max_position_embeddings = {max_length}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_meta = {
        "language": args.lang,
        "num_samples_requested": args.N,
        "num_samples_used_src": None,
        "num_samples_used_tgt": None,
        "max_length": max_length,
        "pad_token_id": pad_id,
        "models": args.models,
        "tokenizer_name": args.tokenizer_name,
        "device": device,
        "dtype": str(dtype),
        "batch_size": args.batch_size,
        "use_cpu_offload": bool(args.use_cpu_offload),
    }

    results = {"meta": base_meta, "spans": {}}

    if args.span in ("src", "both"):
        pack = filter_and_truncate(
            input_ids=input_ids_all,
            span_mask=src_mask_all,
            pad_id=pad_id,
            N=args.N,
            max_length=max_length,
        )
        print(f"\n[src] Using sequences: {pack['input_ids'].size()} (after filtering)")
        results["spans"]["src"] = run_span(
            span_name="src",
            model_names=args.models,
            tokenizer_name=args.tokenizer_name,
            device=device,
            dtype=dtype,
            use_cpu_offload=args.use_cpu_offload,
            input_ids=pack["input_ids"],
            attention_mask=pack["attention_mask"],
            span_mask=pack["span_mask"],
            batch_size=args.batch_size,
        )
        results["spans"]["src"]["num_samples_used"] = int(pack["input_ids"].size(0))
        results["meta"]["num_samples_used_src"] = int(pack["input_ids"].size(0))

    if args.span in ("tgt", "both"):
        pack = filter_and_truncate(
            input_ids=input_ids_all,
            span_mask=tgt_mask_all,
            pad_id=pad_id,
            N=args.N,
            max_length=max_length,
        )
        print(f"\n[tgt] Using sequences: {pack['input_ids'].size()} (after filtering)")
        results["spans"]["tgt"] = run_span(
            span_name="tgt",
            model_names=args.models,
            tokenizer_name=args.tokenizer_name,
            device=device,
            dtype=dtype,
            use_cpu_offload=args.use_cpu_offload,
            input_ids=pack["input_ids"],
            attention_mask=pack["attention_mask"],
            span_mask=pack["span_mask"],
            batch_size=args.batch_size,
        )
        results["spans"]["tgt"]["num_samples_used"] = int(pack["input_ids"].size(0))
        results["meta"]["num_samples_used_tgt"] = int(pack["input_ids"].size(0))

    out_path = out_dir / f"{args.lang}.masked_cka.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved masked CKA results to {out_path}")


if __name__ == "__main__":
    main()
