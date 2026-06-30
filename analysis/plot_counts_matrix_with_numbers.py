#!/usr/bin/env python3
"""
Plot a layerwise neuron-count MATRIX for a single mask file (your tgt.pt),
and write the integer counts inside each cell.

Matrix definition:
  rows = layers (layer 0 at bottom, max layer at top)
  cols = languages
  value = number of selected neurons (len(final_indices[lang][layer]))

Input format (torch.load):
  {
    "languages": [...],
    "final_indices": [lang_idx][num_layers] tensors,
    "span": "src" or "tgt" (optional)
  }

Usage:
  python plot_counts_matrix_with_numbers.py \
    --pt ./activation_mask/indic_en/Qwen_Qwen2.5-3B-Instruct/src.pt \
    --langs "hi bn ta te" \
    --out_dir ./matrix_plots


  python plot_counts_matrix_with_numbers.py \
    --pt ./activation_mask/indic_en/baban_QwenTranslate_Telugu_English/src.pt \
    --langs "hi bn ta te" \
    --out_dir ./matrix_plots_telugu_english

  python plot_counts_matrix_with_numbers.py \
    --pt ./activation_mask/indic_en/baban_QwenTranslate_Tamil_English/src.pt \
    --langs "hi bn ta te" \
    --out_dir ./matrix_plots_tamil_english

  python plot_counts_matrix_with_numbers.py \
    --pt ./activation_mask/indic_en/baban_QwenTranslate_Hindi_English/src.pt \
    --langs "hi bn ta te" \
    --out_dir ./matrix_plots_hindi_english

  python plot_counts_matrix_with_numbers.py \
    --pt ./activation_mask/indic_en/baban_QwenTranslate_Bengali_English/src.pt \
    --langs "hi bn ta te" \
    --out_dir ./matrix_plots_bengali_english


  python plot_counts_matrix_with_numbers.py \
    --pt ./activation_mask/en_indic/Qwen_Qwen2.5-3B-Instruct/tgt.pt \
    --langs "hi bn ta te" \
    --out_dir ./matrix_plots_en_indic


  python plot_counts_matrix_with_numbers.py \
    --pt ./activation_mask/en_indic/baban_QwenTranslate_English_Telugu/tgt.pt \
    --langs "hi bn ta te" \
    --out_dir ./matrix_plots_en_indic_english_telugu

  python plot_counts_matrix_with_numbers.py \
    --pt ./activation_mask/en_indic/baban_QwenTranslate_English_Tamil/tgt.pt \
    --langs "hi bn ta te" \
    --out_dir ./matrix_plots_en_indic_english_tamil

  python plot_counts_matrix_with_numbers.py \
    --pt ./activation_mask/en_indic/baban_QwenTranslate_English_Hindi/tgt.pt \
    --langs "hi bn ta te" \
    --out_dir ./matrix_plots_en_indic_english_hindi

  python plot_counts_matrix_with_numbers.py \
    --pt ./activation_mask/en_indic/baban_QwenTranslate_English_Bengali/tgt.pt \
    --langs "hi bn ta te" \
    --out_dir ./matrix_plots_en_indic_english_bengali



  # or for all languages in file order:
  python plot_counts_matrix_with_numbers.py \
    --pt ./activation_mask/indic_en/Qwen_Qwen2.5-3B-Instruct/tgt.pt \
    --langs all \
    --out_dir ./matrix_plots

Notes:
- Always saves a PDF with a detailed filename.
- Numbers are always drawn inside cells.
- If you have many languages, reduce font size with --font_size.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import torch


def load_mask(path: str) -> Dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing file: {path}")
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise ValueError(f"Unexpected object at {path}: {type(obj)}")

    for k in ("languages", "final_indices"):
        if k not in obj:
            raise KeyError(f"{path} missing key '{k}'")

    langs = obj["languages"]
    if not isinstance(langs, (list, tuple)) or len(langs) == 0:
        raise ValueError(f"{path} has invalid 'languages'")

    fi = obj["final_indices"]
    if not isinstance(fi, (list, tuple)) or len(fi) == 0:
        raise ValueError(f"{path} has invalid 'final_indices'")

    span = obj.get("span", None)
    if span is not None and span not in ("src", "tgt"):
        raise ValueError(f"{path} span={span}, expected 'src' or 'tgt' (or missing)")

    return obj


def safe_stem(path: str) -> str:
    base = os.path.basename(path)
    for ext in (".pt", ".pth", ".bin"):
        if base.endswith(ext):
            base = base[: -len(ext)]
            break
    return base.replace(" ", "_")


def resolve_langs(all_langs: List[str], langs_arg: str) -> List[str]:
    s = langs_arg.strip()
    if s.lower() == "all":
        return list(all_langs)
    langs = s.split()
    if not langs:
        raise ValueError("--langs must be 'all' or a space-separated list like 'hi bn ta te'")
    missing = [l for l in langs if l not in all_langs]
    if missing:
        raise KeyError(f"Missing languages in file: {missing}. Available: {list(all_langs)}")
    return langs


def build_counts_matrix(obj: Dict, langs: List[str]) -> Tuple[torch.Tensor, int]:
    all_langs = list(obj["languages"])
    fi = obj["final_indices"]

    first_li = all_langs.index(langs[0])
    first_per_layer = fi[first_li]
    if not isinstance(first_per_layer, (list, tuple)) or len(first_per_layer) == 0:
        raise ValueError(f"final_indices[{langs[0]}] is not a non-empty list/tuple")
    num_layers = len(first_per_layer)

    mat = torch.zeros((num_layers, len(langs)), dtype=torch.long)

    for j, lang in enumerate(langs):
        li = all_langs.index(lang)
        per_layer = fi[li]
        if not isinstance(per_layer, (list, tuple)):
            raise TypeError(f"final_indices[{lang}] is {type(per_layer)}, expected list/tuple")
        if len(per_layer) != num_layers:
            raise ValueError(
                f"Layer mismatch for lang={lang}: expected {num_layers} layers, got {len(per_layer)}"
            )
        for i, t in enumerate(per_layer):
            if not isinstance(t, torch.Tensor):
                raise TypeError(f"final_indices[{lang}][{i}] is {type(t)}, expected torch.Tensor")
            mat[i, j] = int(t.numel())

    return mat, num_layers


def plot_matrix_pdf(
    mat: torch.Tensor,
    langs: List[str],
    title_path: str,
    span: str,
    out_pdf: str,
    font_size: int,
    max_fig_w: float,
    max_fig_h: float,
) -> None:
    import matplotlib.pyplot as plt  # local import

    data = mat.detach().cpu().numpy()  # shape: [layers, langs]
    n_layers, n_langs = data.shape

    # Figure size heuristics, clamped by user-provided maxima
    fig_w = min(max_fig_w, max(7.0, 0.65 * n_langs + 2.5))
    fig_h = min(max_fig_h, max(7.0, 0.30 * n_layers + 2.5))

    plt.figure(figsize=(fig_w, fig_h))
    im = plt.imshow(data, aspect="auto", origin="lower")  # layer 0 at bottom
    plt.colorbar(im, label="Neuron count")

    plt.xticks(range(n_langs), langs, rotation=45, ha="right")
    plt.yticks(range(n_layers), [str(i) for i in range(n_layers)])

    plt.xlabel("Language")
    plt.ylabel("Layer")
    # plt.title(f"Layerwise neuron counts matrix | span={span}\n{title_path}")

    # Write counts inside cells (always)
    # To keep readability when values are large, use a compact font size via --font_size.
    for i in range(n_layers):
        for j in range(n_langs):
            plt.text(
                j,
                i,
                str(int(data[i, j])),
                ha="center",
                va="center",
                fontsize=font_size,
            )

    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pt",
        type=str,
        default="./activation_mask/indic_en/Qwen_Qwen2.5-3B-Instruct/tgt.pt",
        help="Path to the mask .pt file (tgt.pt).",
    )
    ap.add_argument(
        "--langs",
        type=str,
        default="all",
        help="Space-separated languages (e.g., 'hi bn ta te') or 'all' for file order.",
    )
    ap.add_argument("--out_dir", type=str, default="./matrix_plots")
    ap.add_argument(
        "--font_size",
        type=int,
        default=8,
        help="Font size for numbers inside cells.",
    )
    ap.add_argument(
        "--max_fig_w",
        type=float,
        default=24.0,
        help="Maximum figure width in inches.",
    )
    ap.add_argument(
        "--max_fig_h",
        type=float,
        default=36.0,
        help="Maximum figure height in inches.",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    obj = load_mask(args.pt)
    all_langs = list(obj["languages"])
    span = obj.get("span", "unknown")

    langs = resolve_langs(all_langs, args.langs)
    mat, num_layers = build_counts_matrix(obj, langs)

    tag = (
        f"MATRIX__counts_with_numbers"
        f"__span={span}"
        f"__layers={num_layers}"
        f"__langs={'-'.join(langs)}"
        f"__file={safe_stem(args.pt)}"
    )
    out_pdf = os.path.join(args.out_dir, f"{tag}.pdf")

    print("Loaded:", os.path.abspath(args.pt))
    print("Span  :", span)
    print("Langs :", langs)
    print("Layers:", num_layers)
    print("Matrix shape (layers x langs):", tuple(mat.shape))
    print("Saving PDF:", os.path.abspath(out_pdf))

    plot_matrix_pdf(
        mat=mat,
        langs=langs,
        title_path=os.path.abspath(args.pt),
        span=span,
        out_pdf=out_pdf,
        font_size=args.font_size,
        max_fig_w=args.max_fig_w,
        max_fig_h=args.max_fig_h,
    )

    # Optional: also print a compact text view
    print("\nPer-layer counts (layer 0 shown first; heatmap shows layer 0 at bottom):")
    for i in range(num_layers):
        row = " ".join(f"{int(mat[i, j]):5d}" for j in range(mat.shape[1]))
        print(f"Layer {i:02d}: {row}")

    print("\nDone.")


if __name__ == "__main__":
    main()
