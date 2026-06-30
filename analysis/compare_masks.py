#!/usr/bin/env python3
"""
Analyze how activation-mask neuron selections change after fine-tuning vs base,
across layers, using only src.pt files.

Expected saved format (from your script):
  torch.load(path) -> {
      "languages": [..],
      "span": "src" or "tgt",
      "final_indices": List[ List[torch.Tensor] ]
          shape: [lang_num][num_layers], each tensor = 1D indices in [0, intermediate_size)
  }

This script:
- Loads base src.pt once.
- Loads per-language fine-tuned src.pt masks.
- For each language: compares FT vs base per layer:
    count_base, count_ft, gained, lost, overlap, jaccard
- Prints summary + (optional) saves plots.

Usage example:
  python compare_masks.py \
    --base_dir ./activation_mask/indic_en/Qwen_Qwen2.5-3B-Instruct \
    --ft_root ./activation_mask/indic_en \
    --ft_template "baban_QwenTranslate_{name}_English" \
    --langs "hi bn ta te" \
    --names "Hindi Bengali Tamil Telugu" \
    --out_dir ./mask_change_reports \
    --plot
"""

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch


@dataclass(frozen=True)
class LayerStats:
    base: int
    ft: int
    gained: int
    lost: int
    overlap: int
    union: int
    jaccard: float


def _as_set(x: torch.Tensor) -> Set[int]:
    if x.numel() == 0:
        return set()
    x = x.detach().cpu().to(torch.long).flatten()
    return set(int(v) for v in x.tolist())


def load_mask(path: str) -> Dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing file: {path}")
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise ValueError(f"Unexpected object at {path}: {type(obj)}")
    if "final_indices" not in obj:
        raise KeyError(f"{path} missing key 'final_indices'")
    if obj.get("span", None) not in (None, "src", "tgt"):
        raise ValueError(f"{path} has unexpected span={obj.get('span')}")
    return obj


def pick_lang_index(mask_obj: Dict, lang: str) -> int:
    langs = mask_obj.get("languages", None)
    if not isinstance(langs, (list, tuple)) or len(langs) == 0:
        raise ValueError("Mask file missing usable 'languages' list")
    if lang not in langs:
        raise KeyError(f"Language '{lang}' not found in mask languages={langs}")
    return int(langs.index(lang))


def per_layer_sets(mask_obj: Dict, lang: str) -> List[Set[int]]:
    li = pick_lang_index(mask_obj, lang)
    final_indices = mask_obj["final_indices"]  # [lang_num][num_layers]
    if li >= len(final_indices):
        raise IndexError("final_indices does not match languages length")
    per_layer_tensors = final_indices[li]
    if not isinstance(per_layer_tensors, (list, tuple)) or len(per_layer_tensors) == 0:
        raise ValueError("final_indices[lang] is not a non-empty list")
    return [_as_set(t) for t in per_layer_tensors]


def compare_layerwise(base_layers: List[Set[int]], ft_layers: List[Set[int]]) -> List[LayerStats]:
    if len(base_layers) != len(ft_layers):
        raise ValueError(f"Layer mismatch: base={len(base_layers)} vs ft={len(ft_layers)}")

    out: List[LayerStats] = []
    for A, B in zip(base_layers, ft_layers):
        inter = A & B
        uni = A | B
        gained = len(B - A)
        lost = len(A - B)
        j = (len(inter) / len(uni)) if len(uni) > 0 else 1.0
        out.append(
            LayerStats(
                base=len(A),
                ft=len(B),
                gained=gained,
                lost=lost,
                overlap=len(inter),
                union=len(uni),
                jaccard=float(j),
            )
        )
    return out


def summarize(stats: List[LayerStats]) -> Dict[str, float]:
    base_total = sum(s.base for s in stats)
    ft_total = sum(s.ft for s in stats)
    gained_total = sum(s.gained for s in stats)
    lost_total = sum(s.lost for s in stats)
    overlap_total = sum(s.overlap for s in stats)
    union_total = sum(s.union for s in stats)
    j_global = (overlap_total / union_total) if union_total > 0 else 1.0
    j_mean = sum(s.jaccard for s in stats) / max(1, len(stats))
    return {
        "base_total": float(base_total),
        "ft_total": float(ft_total),
        "gained_total": float(gained_total),
        "lost_total": float(lost_total),
        "overlap_total": float(overlap_total),
        "union_total": float(union_total),
        "jaccard_global": float(j_global),
        "jaccard_mean_layer": float(j_mean),
    }


def print_table(lang: str, stats: List[LayerStats]) -> None:
    print(f"\n=== {lang} (FT vs Base) per-layer ===")
    header = f"{'L':>3} | {'base':>6} {'ft':>6} | {'gained':>6} {'lost':>6} | {'overlap':>7} {'jaccard':>7}"
    print(header)
    print("-" * len(header))
    for i, s in enumerate(stats):
        print(
            f"{i:>3} | {s.base:>6} {s.ft:>6} | {s.gained:>6} {s.lost:>6} | {s.overlap:>7} {s.jaccard:>7.3f}"
        )
    summ = summarize(stats)
    print(
        "\nTotals:"
        f" base={int(summ['base_total'])}"
        f" ft={int(summ['ft_total'])}"
        f" gained={int(summ['gained_total'])}"
        f" lost={int(summ['lost_total'])}"
        f" global_jaccard={summ['jaccard_global']:.3f}"
        f" mean_layer_jaccard={summ['jaccard_mean_layer']:.3f}"
    )


def maybe_plot(
    out_path: str,
    lang: str,
    stats: List[LayerStats],
    title_prefix: str = "",
) -> None:
    import matplotlib.pyplot as plt  # only imported if --plot

    L = list(range(len(stats)))
    base = [s.base for s in stats]
    ft = [s.ft for s in stats]
    gained = [s.gained for s in stats]
    lost = [s.lost for s in stats]
    jacc = [s.jaccard for s in stats]

    # Plot counts
    plt.figure()
    plt.plot(L, base, label="base_count")
    plt.plot(L, ft, label="ft_count")
    plt.plot(L, gained, label="gained")
    plt.plot(L, lost, label="lost")
    plt.xlabel("Layer")
    plt.ylabel("Neuron count")
    plt.title(f"{title_prefix}{lang}: counts (base vs ft)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, f"{lang}_counts.png"), dpi=200)
    plt.close()

    # Plot Jaccard
    plt.figure()
    plt.plot(L, jacc, label="jaccard")
    plt.xlabel("Layer")
    plt.ylabel("Jaccard")
    plt.title(f"{title_prefix}{lang}: per-layer Jaccard (ft vs base)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, f"{lang}_jaccard.png"), dpi=200)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=str, required=True, help="Directory containing base src.pt")
    ap.add_argument("--ft_root", type=str, required=True, help="Root directory containing fine-tuned folders")
    ap.add_argument(
        "--ft_template",
        type=str,
        default="baban_QwenTranslate_{name}_English",
        help="Fine-tuned folder name template under ft_root",
    )
    ap.add_argument("--langs", type=str, default="hi bn ta te", help="Language codes to compare (space-separated)")
    ap.add_argument(
        "--names",
        type=str,
        default="Hindi Bengali Tamil Telugu",
        help="Folder language names (space-separated), aligned with --langs",
    )
    ap.add_argument("--out_dir", type=str, default="./mask_change_reports")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    lang_codes = args.langs.split()
    lang_names = args.names.split()
    if len(lang_codes) != len(lang_names):
        raise ValueError(f"--langs has {len(lang_codes)} items but --names has {len(lang_names)} items")

    base_path = os.path.join(args.base_dir, "src.pt")
    base_obj = load_mask(base_path)
    if base_obj.get("span", "src") != "src":
        raise ValueError(f"Base file span is {base_obj.get('span')}, expected 'src'")
    print(f"Loaded base: {base_path}")
    base_langs = base_obj.get("languages", [])
    print(f"Base languages: {base_langs}")

    os.makedirs(args.out_dir, exist_ok=True)

    for code, name in zip(lang_codes, lang_names):
        ft_dir = os.path.join(args.ft_root, args.ft_template.format(name=name))
        ft_path = os.path.join(ft_dir, "src.pt")
        ft_obj = load_mask(ft_path)
        if ft_obj.get("span", "src") != "src":
            raise ValueError(f"FT file span is {ft_obj.get('span')} at {ft_path}, expected 'src'")

        # Ensure both masks contain this language code
        base_layers = per_layer_sets(base_obj, code)
        ft_layers = per_layer_sets(ft_obj, code)

        stats = compare_layerwise(base_layers, ft_layers)
        print_table(code, stats)

        if args.plot:
            maybe_plot(args.out_dir, code, stats, title_prefix="Mask change: ")

    print(f"\nDone. Reports in: {args.out_dir}")


if __name__ == "__main__":
    main()
