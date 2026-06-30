#!/usr/bin/env python3
"""
Neuron Usage Alignment (NUA) from already-saved vLLM activation dumps.

Assumes you have already saved per-model activation files via your script, e.g.:

  acts_<SAFE_NAME>/en_indic/src/activation.<lang>.src.pt
  acts_<SAFE_NAME>/en_indic/tgt/activation.<lang>.tgt.pt

Where each file contains a dict with at least:
  - "lang": str
  - "span": "src" or "tgt"
  - "n": int
  - "over_zero": Tensor[int32] of shape [num_layers, intermediate_size]

NUA compares, for each layer, the usage vectors u_l (intermediate_size,) between two models.
You can choose to normalize by n (masked token count) to get rates.

Outputs:
  - pairwise per-layer NUA (cosine similarity)
  - aggregated upper/mid/low layer NUA
  - optional heatmap matrices saved as .pt and .json
  - optional plots saved as .png (matplotlib)

Example:

python nua_en_indic.py \
  --acts_root ./ \
  --models \
    Qwen_Qwen2.5-3B-Instruct \
    baban_QwenTranslate_English_Hindi \
    baban_QwenTranslate_English_Bengali \
    baban_QwenTranslate_English_Tamil \
    baban_QwenTranslate_English_Telugu \
  --langs hi bn ta te \
  --span src \
  --normalize rate \
  --bands "low:0-11 mid:12-23 upper:24-35" \
  --out_dir nua_outputs_en_indic \
  --save_plots


Notes:
- The --models values are SAFE_NAME directory names (acts_<SAFE_NAME>/...).
  If you want, you can also pass raw model names and use --auto_safe_name.
"""

import os
import re
import json
import math
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch


def safe_name_from_model(model: str) -> str:
    # Mimic: sed 's/[^A-Za-z0-9._-]/_/g'
    return re.sub(r"[^A-Za-z0-9._-]", "_", model)


def parse_band_spec(spec: str) -> Dict[str, List[int]]:
    """
    Parse bands like:
      low:0-11 mid:12-23 upper:24-35
    Returns dict band_name -> list of layer indices.
    """
    bands: Dict[str, List[int]] = {}
    for chunk in spec.strip().split():
        name, rng = chunk.split(":")
        if "-" in rng:
            a, b = rng.split("-")
            a_i, b_i = int(a), int(b)
            if b_i < a_i:
                raise ValueError(f"Bad band range: {chunk}")
            bands[name] = list(range(a_i, b_i + 1))
        else:
            bands[name] = [int(rng)]
    return bands


def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / x.norm(p=2).clamp_min(eps)


def cosine_sim(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> float:
    x = x.to(torch.float32)
    y = y.to(torch.float32)
    num = (x * y).sum()
    denom = x.norm(p=2) * y.norm(p=2)
    denom = denom.clamp_min(eps)
    return (num / denom).item()


def load_activation_file(path: Path) -> Dict:
    obj = torch.load(path, map_location="cpu")
    if "over_zero" not in obj:
        raise KeyError(f"Missing 'over_zero' in {path}")
    if "n" not in obj:
        raise KeyError(f"Missing 'n' in {path}")
    return obj


def collect_model_lang_usage(
    acts_root: Path,
    safe_model: str,
    span: str,
    langs: List[str],
    normalize: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      U: [L, I] float32 usage vectors aggregated across langs
      n_total: [L] float32 total masked token count used in normalization reference (same for all layers)
    Strategy:
      - Load each activation.<lang>.<span>.pt
      - Get over_zero: [L, I]
      - Optionally convert counts to rates by dividing by n (masked tokens in that file)
      - Aggregate across languages by averaging rates (or summing counts then dividing by total n)

    normalize:
      - "count": use raw counts; aggregate by sum over languages
      - "rate": convert to per-token rate within each lang, then average across languages
      - "rate_weighted": convert to counts sum, divide by total n across languages (weighted rate)
    """
    per_lang = []
    ns = []
    for lang in langs:
        f = acts_root / f"acts_{safe_model}" / "en_indic" / span / f"activation.{lang}.{span}.pt"
        if not f.exists():
            raise FileNotFoundError(f"Missing activation file: {f}")
        obj = load_activation_file(f)
        over = obj["over_zero"].to(torch.float32)  # [L, I]
        n = float(obj["n"])
        if n <= 0:
            raise ValueError(f"Non-positive n in {f}: {n}")
        per_lang.append(over)
        ns.append(n)

    # Sanity: shapes
    L, I = per_lang[0].shape
    for t in per_lang[1:]:
        if t.shape != (L, I):
            raise ValueError("Mismatched shapes across languages for same model/span.")

    if normalize == "count":
        U = torch.stack(per_lang, dim=0).sum(dim=0)  # [L, I]
        n_total = torch.tensor([sum(ns)] * L, dtype=torch.float32)
        return U, n_total

    if normalize == "rate":
        # Per-lang rate then average (equal weight per language)
        rates = []
        for over, n in zip(per_lang, ns):
            rates.append(over / n)
        U = torch.stack(rates, dim=0).mean(dim=0)  # [L, I]
        n_total = torch.tensor([sum(ns)] * L, dtype=torch.float32)
        return U, n_total

    if normalize == "rate_weighted":
        # Sum counts then divide by total tokens across languages
        total_over = torch.stack(per_lang, dim=0).sum(dim=0)  # [L, I]
        total_n = sum(ns)
        U = total_over / float(total_n)
        n_total = torch.tensor([total_n] * L, dtype=torch.float32)
        return U, n_total

    raise ValueError(f"Unknown normalize mode: {normalize}")


def compute_pairwise_nua(
    model_to_U: Dict[str, torch.Tensor],
) -> Dict[str, Dict[str, float]]:
    """
    Returns:
      pair_key -> {layer_00: sim, ...}
    """
    models = list(model_to_U.keys())
    out: Dict[str, Dict[str, float]] = {}
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            a, b = models[i], models[j]
            Ua = model_to_U[a]  # [L, I]
            Ub = model_to_U[b]
            L = min(Ua.size(0), Ub.size(0))
            pair_key = f"{a}|||{b}"
            out[pair_key] = {}
            for l in range(L):
                sim = cosine_sim(Ua[l], Ub[l])
                out[pair_key][f"layer_{l:02d}"] = float(sim)
    return out


def summarize_bands(
    pairwise: Dict[str, Dict[str, float]],
    bands: Dict[str, List[int]],
) -> Dict[str, Dict[str, float]]:
    """
    For each pair, average similarity over each band.
    """
    out: Dict[str, Dict[str, float]] = {}
    for pair_key, layer_map in pairwise.items():
        out[pair_key] = {}
        # Convert layer_map to list for easy indexing
        # layer_map keys like layer_00
        for band_name, layers in bands.items():
            vals = []
            for l in layers:
                k = f"layer_{l:02d}"
                if k in layer_map:
                    vals.append(layer_map[k])
            if len(vals) == 0:
                out[pair_key][band_name] = float("nan")
            else:
                out[pair_key][band_name] = float(sum(vals) / len(vals))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts_root", type=str, required=True, help="Directory containing acts_<SAFE_NAME>/...")
    ap.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=True,
        help="SAFE_NAME(s) (directory names after acts_), or raw model ids if --auto_safe_name is set",
    )
    ap.add_argument("--auto_safe_name", action="store_true", help="Convert given --models entries into SAFE_NAME.")
    ap.add_argument("--langs", type=str, nargs="+", default=["hi", "bn", "ta", "te"])
    ap.add_argument("--span", type=str, choices=["src", "tgt"], required=True)
    ap.add_argument(
        "--normalize",
        type=str,
        default="rate_weighted",
        choices=["count", "rate", "rate_weighted"],
        help="How to normalize over_zero counts before comparing.",
    )
    ap.add_argument(
        "--bands",
        type=str,
        default="low:0-11 mid:12-23 upper:24-35",
        help="Layer bands for aggregation.",
    )
    ap.add_argument("--out_dir", type=str, default="nua_outputs")
    ap.add_argument("--save_plots", action="store_true", help="Save layerwise similarity plots (requires matplotlib).")
    args = ap.parse_args()

    acts_root = Path(args.acts_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bands = parse_band_spec(args.bands)

    safe_models = [safe_name_from_model(m) for m in args.models] if args.auto_safe_name else args.models

    # Load and aggregate usage vectors per model
    model_to_U: Dict[str, torch.Tensor] = {}
    meta: Dict[str, Dict] = {}

    for raw, sm in zip(args.models, safe_models):
        U, n_total = collect_model_lang_usage(
            acts_root=acts_root,
            safe_model=sm,
            span=args.span,
            langs=args.langs,
            normalize=args.normalize,
        )
        model_to_U[sm] = U  # [L, I] float32
        meta[sm] = {
            "raw_model_arg": raw,
            "safe_model": sm,
            "span": args.span,
            "langs": args.langs,
            "normalize": args.normalize,
            "num_layers": int(U.size(0)),
            "intermediate_size": int(U.size(1)),
            "total_masked_tokens_sum_over_langs": float(n_total[0].item()),
        }
        print(f"Loaded {sm}: U={tuple(U.shape)} total_masked_tokens={meta[sm]['total_masked_tokens_sum_over_langs']}")

    # Pairwise layerwise cosine similarities
    pairwise = compute_pairwise_nua(model_to_U)
    band_summary = summarize_bands(pairwise, bands)

    results = {
        "meta": {
            "acts_root": str(acts_root),
            "span": args.span,
            "langs": args.langs,
            "normalize": args.normalize,
            "bands": bands,
            "models": safe_models,
        },
        "per_model": meta,
        "pairs_layerwise": pairwise,
        "pairs_bands": band_summary,
    }

    out_json = out_dir / f"nua.{args.span}.{args.normalize}.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved NUA results: {out_json}")

    # Optional plots
    if args.save_plots:
        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            raise RuntimeError("matplotlib is required for --save_plots") from e

        # For each pair: plot similarity vs layer
        for pair_key, layer_map in pairwise.items():
            layers = sorted(int(k.split("_")[1]) for k in layer_map.keys())
            vals = [layer_map[f"layer_{l:02d}"] for l in layers]

            plt.figure()
            plt.plot(layers, vals)
            plt.xlabel("Layer")
            plt.ylabel("Neuron usage alignment (cosine)")
            plt.title(f"{pair_key} [{args.span}, {args.normalize}]")
            plt.ylim(0.0, 1.0)
            fig_path = out_dir / f"plot.{args.span}.{args.normalize}.{safe_name_from_model(pair_key)}.png"
            plt.savefig(fig_path, dpi=200, bbox_inches="tight")
            plt.close()

        print(f"Saved plots under: {out_dir}")

    # Print a compact summary table to stdout
    print("\n=== Band-averaged NUA (higher = more similar neuron usage) ===")
    band_names = list(bands.keys())
    header = "PAIR".ljust(100) + " " + " ".join([b.rjust(10) for b in band_names])
    print(header)
    print("-" * len(header))
    for pair_key, bmap in band_summary.items():
        row = pair_key[:100].ljust(100)
        for b in band_names:
            v = bmap.get(b, float("nan"))
            row += f" {v:10.4f}" if math.isfinite(v) else " " + "   nan   ".rjust(10)
        print(row)


if __name__ == "__main__":
    main()
