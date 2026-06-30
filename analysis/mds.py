#!/usr/bin/env python3
"""
Geometric analysis for en->Indic Qwen MT models (LOW GPU MEM):
1) Layerwise principal-angle curves (target span only)
2) 2D subspace maps (MDS) from principal-angle subspace distances

Assumptions:
- You already have masked tensors from your masking script:
    input_ids.{lang}.pt   [N, T]
    src_mask.{lang}.pt    [N, T] bool
    tgt_mask.{lang}.pt    [N, T] bool
- We only care about en-indic and span="tgt" (as requested)

Model names:
  base: Qwen/Qwen2.5-3B-Instruct
  ft:   anonymous/AnonMT_English_{Hindi,Bengali,Tamil,Telugu}

Outputs:
- JSON with per-layer principal angles and subspace distances for each model pair
- PNG plots:
    - principal_angle_curve_layerwise.png
    - subspace_map_layer_XX.png (for selected layers)
    - optional: pairwise_distance_heatmap_layer_XX.png

Example:
python mds.py \
  --lang ta \
  --data_dir data_masked_qwen/en_indic \
  --tokenizer_name Qwen/Qwen2.5-3B-Instruct \
  --N 256 \
  --max_length 512 \
  --batch_size 8 \
  --plot_heatmaps \
  --k 64 \
  --layers_to_map 0 12 24 34 35 36 \
  --out_dir geom_outputs_en_indic

Notes:
- This script pools hidden states per layer by mean over masked target tokens.
- Uses hooks and micro-batches; stores only [N, D] per layer on CPU float32.
"""

import os
import gc
import json
import argparse
from pathlib import Path
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

# sklearn only for MDS (very small dependency)
from sklearn.manifold import MDS


# -----------------------------
# Utilities
# -----------------------------
def pick_dtype(device: str) -> torch.dtype:
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device == "cuda":
        return torch.float16
    return torch.float32


def _maybe_set_pad_token(tokenizer) -> int:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return int(tokenizer.pad_token_id)


def _center_np(X: np.ndarray) -> np.ndarray:
    return X - X.mean(axis=0, keepdims=True)


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
        raise RuntimeError(f"Too few valid rows ({input_ids.size(0)}) for geometry. Need at least 2.")

    input_ids = input_ids[:N].contiguous()
    span_mask = span_mask[:N].contiguous()
    attention_mask = attention_mask[:N].contiguous()

    return {
        "input_ids": input_ids,
        "span_mask": span_mask,
        "attention_mask": attention_mask,
    }


# -----------------------------
# Low-memory masked pooling via hooks
# -----------------------------
class LayerwiseMaskedPooler:
    """
    Pools [B, T, D] -> [B, D] per layer by mean over masked tokens.
    Stores pooled reps on CPU float32 as chunks, later concatenated -> [N, D].
    """

    def __init__(self):
        self.handles = []
        self._active = False

        self.pool_mask: Optional[torch.Tensor] = None   # [B, T] bool
        self.denom: Optional[torch.Tensor] = None       # [B] int64
        self.layer_cpu_chunks: Optional[List[List[torch.Tensor]]] = None  # list over layers

    def _pool_and_store(self, h: torch.Tensor, layer_idx: int) -> None:
        assert self.pool_mask is not None
        assert self.denom is not None
        assert self.layer_cpu_chunks is not None

        pm = self.pool_mask.unsqueeze(-1).to(dtype=h.dtype)  # [B, T, 1]
        summed = (h * pm).sum(dim=1)                         # [B, D]
        rep = summed / self.denom.unsqueeze(-1).to(dtype=h.dtype)  # [B, D]

        self.layer_cpu_chunks[layer_idx].append(rep.detach().to("cpu", dtype=torch.float32))

    def _hook_factory(self, layer_idx: int):
        def hook(_module, _inputs, output):
            h = output[0] if isinstance(output, (tuple, list)) else output
            if not torch.is_tensor(h):
                return
            if h.dim() != 3:
                return
            self._pool_and_store(h, layer_idx)
        return hook

    def attach(self, model) -> int:
        if self._active:
            raise RuntimeError("Pooler already active/attached.")
        if not hasattr(model, "model"):
            raise RuntimeError("Unexpected model structure: missing `.model`")

        core = model.model
        if not hasattr(core, "embed_tokens"):
            raise RuntimeError("Unexpected model structure: missing `model.model.embed_tokens`")
        if not hasattr(core, "layers"):
            raise RuntimeError("Unexpected model structure: missing `model.model.layers`")

        layers = list(core.layers)
        num_blocks = len(layers)

        # embeddings = layer 0, blocks = 1..num_blocks
        total_layers = 1 + num_blocks
        self.layer_cpu_chunks = [[] for _ in range(total_layers)]

        self.handles.append(core.embed_tokens.register_forward_hook(self._hook_factory(layer_idx=0)))
        for i, blk in enumerate(layers, start=1):
            self.handles.append(blk.register_forward_hook(self._hook_factory(layer_idx=i)))

        self._active = True
        return total_layers

    def set_batch_masks(self, attention_mask: torch.Tensor, token_mask: torch.Tensor) -> None:
        pool_mask = (attention_mask.to(torch.bool) & token_mask.to(torch.bool))
        denom = pool_mask.sum(dim=1).clamp_min(1)
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
    Returns reps: list length (1 + num_blocks), each [N, D] on CPU float32.
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

            if device == "cuda":
                ids_b = ids_b.to(device, non_blocking=True)
                att_b = att_b.to(device, non_blocking=True)
                msk_b = msk_b.to(device, non_blocking=True)

            pooler.set_batch_masks(attention_mask=att_b, token_mask=msk_b)

            _ = model(
                input_ids=ids_b,
                attention_mask=att_b,
                use_cache=False,
                output_hidden_states=False,
                output_attentions=False,
                return_dict=False,
            )

            del ids_b, att_b, msk_b, _
            if device == "cuda":
                torch.cuda.empty_cache()

        reps = pooler.finalize()
        if len(reps) != total_layers:
            raise RuntimeError(f"Hook layers mismatch: collected {len(reps)} vs expected {total_layers}")
        return reps

    finally:
        pooler.detach()


def load_model(
    name: str,
    dtype: torch.dtype,
    device: str,
    use_cpu_offload: bool,
):
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


# -----------------------------
# Geometry: subspace bases, principal angles, distances
# -----------------------------
def compute_subspace_basis_from_reps(H: torch.Tensor, k: int) -> np.ndarray:
    """
    H: [N, D] CPU float32 torch tensor
    Returns Q: [D, k_eff] orthonormal basis (numpy float64)
    """
    X = H.cpu().numpy().astype(np.float64, copy=False)
    X = _center_np(X)
    # SVD: X = U S Vt, principal directions are rows of Vt
    # Use full_matrices=False for speed
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    V = Vt.T  # [D, r]
    k_eff = int(min(k, V.shape[1]))
    Q = V[:, :k_eff]
    # Orthonormalize
    Q, _ = np.linalg.qr(Q)
    return Q[:, :k_eff]


def principal_angles_deg(Qa: np.ndarray, Qb: np.ndarray) -> np.ndarray:
    """
    Qa: [D, ka], Qb: [D, kb], both orthonormal.
    Returns angles in degrees, length = min(ka,kb).
    """
    M = Qa.T @ Qb
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    return np.degrees(np.arccos(s))


def subspace_distance(Qa: np.ndarray, Qb: np.ndarray) -> float:
    """
    Distance in [0,1]: sqrt(mean(sin^2(theta_i))).
    0 means identical subspaces; 1 means orthogonal (on average).
    """
    ang = principal_angles_deg(Qa, Qb)
    rad = np.radians(ang)
    return float(np.sqrt(np.mean(np.sin(rad) ** 2)))


def pairwise_geometry_for_layer(
    bases: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, float]]:
    """
    bases: model_name -> Q [D,k]
    Returns:
      {
        "dist": { "a|||b": float },
        "median_angle": { "a|||b": float },
      }
    """
    out_dist = {}
    out_medang = {}

    names = list(bases.keys())
    for a, b in combinations(names, 2):
        Qa = bases[a]
        Qb = bases[b]
        ang = principal_angles_deg(Qa, Qb)
        d = subspace_distance(Qa, Qb)
        key = f"{a}|||{b}"
        out_dist[key] = d
        out_medang[key] = float(np.median(ang))
    return {"dist": out_dist, "median_angle": out_medang}


def build_distance_matrix(
    names: List[str],
    dist_map: Dict[str, float],
) -> np.ndarray:
    """
    dist_map uses keys "a|||b" for a<b pairs.
    Returns symmetric D [m,m]
    """
    m = len(names)
    D = np.zeros((m, m), dtype=np.float64)

    index = {n: i for i, n in enumerate(names)}
    for key, val in dist_map.items():
        a, b = key.split("|||")
        i = index[a]
        j = index[b]
        D[i, j] = val
        D[j, i] = val
    return D


def mds_2d(D: np.ndarray, random_state: int = 0) -> np.ndarray:
    mds = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=random_state,
        n_init=6,
        max_iter=2000,
    )
    return mds.fit_transform(D)


# -----------------------------
# Plotting
# -----------------------------
def _safe_label(name: str) -> str:
    # Shorten HF names for plots
    if name == "Qwen/Qwen2.5-3B-Instruct":
        return "Base"
    if name.startswith("anonymous/AnonMT_English_"):
        return name.split("anonymous/AnonMT_English_", 1)[1]
    return name


def plot_layerwise_principal_angle_curves(
    layers: List[int],
    pair_to_curve: Dict[str, Dict[str, List[float]]],
    out_path: Path,
    title: str,
) -> None:
    """
    pair_to_curve: pair_key -> {"median_angle": [..], "dist": [..]}
    Plots median principal angle curves per pair.
    """
    plt.figure()
    for pair_key, curves in pair_to_curve.items():
        label = pair_key
        a, b = pair_key.split("|||")
        label = f"{_safe_label(a)} vs {_safe_label(b)}"
        plt.plot(layers, curves["median_angle"], label=label)
    plt.xlabel("Layer")
    plt.ylabel("Median principal angle (deg)")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_subspace_map(
    coords: np.ndarray,
    names: List[str],
    out_path: Path,
    title: str,
) -> None:
    plt.figure()
    plt.scatter(coords[:, 0], coords[:, 1])
    for i, n in enumerate(names):
        plt.text(coords[i, 0], coords[i, 1], f" {_safe_label(n)}", fontsize=10)
    plt.xlabel("MDS-1")
    plt.ylabel("MDS-2")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_distance_heatmap(
    D: np.ndarray,
    names: List[str],
    out_path: Path,
    title: str,
) -> None:
    plt.figure()
    plt.imshow(D, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(names)), [_safe_label(n) for n in names], rotation=45, ha="right")
    plt.yticks(range(len(names)), [_safe_label(n) for n in names])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", type=str, required=True, help="hi | bn | ta | te")
    ap.add_argument("--data_dir", type=str, required=True, help="e.g. data_masked_qwen/en_indic")
    ap.add_argument("--tokenizer_name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--N", type=int, default=256)
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--k", type=int, default=64, help="Subspace rank for principal angles (top-k directions)")
    ap.add_argument("--layers_to_map", type=int, nargs="*", default=[0, 12, 24, 34, 35, 36],
                    help="Which layers to render 2D subspace maps for")
    ap.add_argument("--out_dir", type=str, default="geom_outputs_en_indic")
    ap.add_argument("--use_cpu_offload", action="store_true", help="HF device_map=auto offload (CUDA only)")
    ap.add_argument("--plot_heatmaps", action="store_true", help="Also save pairwise distance heatmaps per mapped layer")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = pick_dtype(device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fixed model list per your request
    base_model = "Qwen/Qwen2.5-3B-Instruct"
    model_names = [
        base_model,
        "anonymous/AnonMT_English_Hindi",
        "anonymous/AnonMT_English_Bengali",
        "anonymous/AnonMT_English_Tamil",
        "anonymous/AnonMT_English_Telugu",
    ]

    print(
        f"device={device}, dtype={dtype}, N={args.N}, max_length={args.max_length}, "
        f"batch_size={args.batch_size}, k={args.k}, cpu_offload={args.use_cpu_offload}"
    )
    print("Models:")
    for m in model_names:
        print("  -", m)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True, trust_remote_code=True)
    pad_id = _maybe_set_pad_token(tokenizer)

    input_ids_all, _src_mask_all, tgt_mask_all, _lengths = load_masked_tensors(args.data_dir, args.lang)

    cfg = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
    model_max_pos = int(getattr(cfg, "max_position_embeddings", args.max_length))
    max_length = min(args.max_length, model_max_pos)
    if max_length != args.max_length:
        print(f"Note: clipped max_length to model max_position_embeddings = {max_length}")

    # We only care about tgt span
    pack = filter_and_truncate(
        input_ids=input_ids_all,
        span_mask=tgt_mask_all,
        pad_id=pad_id,
        N=args.N,
        max_length=max_length,
    )
    input_ids = pack["input_ids"]
    attention_mask = pack["attention_mask"]
    tgt_mask = pack["span_mask"]

    print(f"\n[tgt] Using sequences: {tuple(input_ids.size())} (after filtering)")

    # 1) Collect layerwise pooled reps for each model
    reps_cache: Dict[str, List[torch.Tensor]] = {}
    total_layers: Optional[int] = None

    for name in model_names:
        print(f"\nCollecting reps (tgt span) for: {name}")
        model = load_model(name=name, dtype=dtype, device=device, use_cpu_offload=args.use_cpu_offload)

        reps = collect_layer_reps_masked_lowmem(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_mask=tgt_mask,
            device=device if not args.use_cpu_offload else ("cuda" if torch.cuda.is_available() else "cpu"),
            batch_size=args.batch_size,
        )

        if total_layers is None:
            total_layers = len(reps)
        else:
            total_layers = min(total_layers, len(reps))

        reps_cache[name] = reps

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    assert total_layers is not None
    layers = list(range(total_layers))
    print(f"\nCollected pooled reps for {len(model_names)} models; total_layers={total_layers}")

    # 2) Build per-layer subspace bases (top-k) for each model
    # bases[layer][model] = Q [D,k]
    print("\nComputing subspace bases per layer...")
    bases_by_layer: List[Dict[str, np.ndarray]] = []
    for layer_idx in layers:
        bases = {}
        for name in model_names:
            H = reps_cache[name][layer_idx]  # [N,D] CPU float32
            Q = compute_subspace_basis_from_reps(H, k=args.k)
            bases[name] = Q
        bases_by_layer.append(bases)

    # 3) Pairwise principal angles + distances per layer
    print("\nComputing pairwise geometry per layer...")
    per_layer = {}
    pair_to_curve: Dict[str, Dict[str, List[float]]] = {}

    pairs = list(combinations(model_names, 2))
    for a, b in pairs:
        key = f"{a}|||{b}"
        pair_to_curve[key] = {"median_angle": [], "dist": []}

    for layer_idx in layers:
        geom = pairwise_geometry_for_layer(bases_by_layer[layer_idx])
        per_layer[f"layer_{layer_idx:02d}"] = geom

        for a, b in pairs:
            key = f"{a}|||{b}"
            pair_to_curve[key]["median_angle"].append(geom["median_angle"][key])
            pair_to_curve[key]["dist"].append(geom["dist"][key])

    # 4) Plot: principal-angle curves (layerwise)
    curves_path = out_dir / f"{args.lang}.principal_angle_curve_layerwise.png"
    plot_layerwise_principal_angle_curves(
        layers=layers,
        pair_to_curve=pair_to_curve,
        out_path=curves_path,
        title=f"Target-span principal angles (en->Indic), lang={args.lang}, k={args.k}",
    )
    print(f"Saved principal-angle curves: {curves_path}")

    # 5) Plot: 2D subspace maps at selected layers (MDS over subspace distances)
    names = list(model_names)
    for L in args.layers_to_map:
        if L < 0 or L >= total_layers:
            print(f"Skipping map for layer {L}: out of range.")
            continue

        dist_map = per_layer[f"layer_{L:02d}"]["dist"]
        D = build_distance_matrix(names=names, dist_map=dist_map)
        coords = mds_2d(D, random_state=0)

        map_path = out_dir / f"{args.lang}.subspace_map_layer_{L:02d}.png"
        plot_subspace_map(
            coords=coords,
            names=names,
            out_path=map_path,
            title=f"2D subspace map (MDS) | tgt span | layer {L:02d} | k={args.k}",
        )
        print(f"Saved subspace map: {map_path}")

        if args.plot_heatmaps:
            hm_path = out_dir / f"{args.lang}.pairwise_distance_heatmap_layer_{L:02d}.png"
            plot_distance_heatmap(
                D=D,
                names=names,
                out_path=hm_path,
                title=f"Subspace distance heatmap | tgt span | layer {L:02d} | k={args.k}",
            )
            print(f"Saved distance heatmap: {hm_path}")

    # 6) Save JSON
    meta = {
        "language": args.lang,
        "data_dir": args.data_dir,
        "num_samples_requested": args.N,
        "num_samples_used_tgt": int(input_ids.size(0)),
        "max_length": max_length,
        "pad_token_id": pad_id,
        "models": model_names,
        "tokenizer_name": args.tokenizer_name,
        "device": device,
        "dtype": str(dtype),
        "batch_size": args.batch_size,
        "use_cpu_offload": bool(args.use_cpu_offload),
        "subspace_rank_k": int(args.k),
        "layers_total": int(total_layers),
        "layers_mapped": list(args.layers_to_map),
    }

    out_json = out_dir / f"{args.lang}.tgt_principal_angles_and_maps.json"
    payload = {
        "meta": meta,
        "per_layer": per_layer,  # contains dist + median_angle per layer, keyed by "a|||b"
    }
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved geometry JSON: {out_json}")


if __name__ == "__main__":
    main()