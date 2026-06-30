#!/usr/bin/env python3
"""
Compare language-specific neuron masks: Base vs Fine-tuned model, for MULTIPLE languages,
using a selectable span ("src" or "tgt").

This script expects mask files saved as a torch dict with keys:
  - "languages": list[str]
  - "final_indices": [lang_idx][num_layers] tensors
  - "span": optional str ("src" or "tgt") for sanity checks

For each language in --langs, the script:
- extracts per-layer neuron sets for the requested --span
- computes per-layer overlap, gained, lost, jaccard
- prints a per-layer table + totals
- saves plots as PDF with explicit, detailed filenames
- optionally saves a JSON summary with explicit, detailed filename

Usage:
  python compare_base_vs_ft.py \
    --base_pt ./activation_mask/indic_en/Qwen_Qwen2.5-3B-Instruct/src.pt \
    --ft_pt   ./activation_mask/indic_en/anonymous_AnonMT_Hindi_English/src.pt \
    --span src \
    --langs "hi bn ta te" \
    --out_dir ./compare_reports \
    --plot \
    --save_json
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Set

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


def _require_file(path: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing file: {path}")


def load_mask(path: str) -> Dict:
    _require_file(path)
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


def as_set(x: torch.Tensor) -> Set[int]:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(x)}")
    if x.numel() == 0:
        return set()
    x = x.detach().cpu().to(torch.long).flatten()
    return set(int(v) for v in x.tolist())


def lang_index(obj: Dict, lang: str) -> int:
    langs = obj["languages"]
    if lang not in langs:
        raise KeyError(f"Language '{lang}' not found. Available={list(langs)}")
    return int(list(langs).index(lang))


def per_layer_sets(obj: Dict, lang: str) -> List[Set[int]]:
    li = lang_index(obj, lang)
    per_layer = obj["final_indices"][li]  # [num_layers] tensors
    if not isinstance(per_layer, (list, tuple)) or len(per_layer) == 0:
        raise ValueError(f"final_indices[{lang}] is not a non-empty list/tuple")
    return [as_set(t) for t in per_layer]


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


def totals(stats: List[LayerStats]) -> Dict[str, float]:
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
        "num_layers": float(len(stats)),
    }


def print_lang_report(lang: str, span: str, stats: List[LayerStats]) -> None:
    print(f"\n=== Language: {lang} | Span: {span} | Base vs FT ===")
    # header = f"{'L':>3} | {'base':>6} {'ft':>6} | {'gained':>6} {'lost':>6} | {'overlap':>7} {'jaccard':>7}"
    print(header)
    print("-" * len(header))
    for i, s in enumerate(stats):
        print(
            f"{i:>3} | {s.base:>6} {s.ft:>6} | {s.gained:>6} {s.lost:>6} | {s.overlap:>7} {s.jaccard:>7.3f}"
        )
    t = totals(stats)
    print(
        "\nTotals:"
        f" base={int(t['base_total'])}"
        f" ft={int(t['ft_total'])}"
        f" gained={int(t['gained_total'])}"
        f" lost={int(t['lost_total'])}"
        f" global_jaccard={t['jaccard_global']:.3f}"
        f" mean_layer_jaccard={t['jaccard_mean_layer']:.3f}"
        f" layers={int(t['num_layers'])}"
    )


def _safe_stem(path: str) -> str:
    # Keep it stable and filesystem-friendly.
    base = os.path.basename(path)
    for ext in (".pt", ".pth", ".bin"):
        if base.endswith(ext):
            base = base[: -len(ext)]
            break
    base = base.replace(" ", "_").replace("/", "_")
    return base


def build_tag(base_pt: str, ft_pt: str, span: str, langs: List[str]) -> str:
    base_stem = _safe_stem(base_pt)
    ft_stem = _safe_stem(ft_pt)
    langs_tag = "-".join(langs)
    return f"COMPARE__span={span}__langs={langs_tag}__base={base_stem}__ft={ft_stem}"


def maybe_plot(out_dir: str, file_prefix: str, lang: str, span: str, stats: List[LayerStats]) -> Dict[str, str]:
    import matplotlib.pyplot as plt  # local import

    L = list(range(len(stats)))
    base = [s.base for s in stats]
    ft = [s.ft for s in stats]
    gained = [s.gained for s in stats]
    lost = [s.lost for s in stats]
    jacc = [s.jaccard for s in stats]

    counts_pdf = os.path.join(
        out_dir,
        f"{file_prefix}__lang={lang}__span={span}__plot=counts_base_ft_gained_lost.pdf",
    )
    jacc_pdf = os.path.join(
        out_dir,
        f"{file_prefix}__lang={lang}__span={span}__plot=jaccard_per_layer.pdf",
    )

    plt.figure()
    plt.plot(L, base, label="base_count")
    plt.plot(L, ft, label="ft_count")
    plt.plot(L, gained, label="gained")
    plt.plot(L, lost, label="lost")
    plt.xlabel("Layer")
    plt.ylabel("Neuron count")
    plt.title(f"{lang} ({span}): counts (base vs ft)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(counts_pdf)
    plt.close()

    plt.figure()
    plt.plot(L, jacc, label="jaccard")
    plt.xlabel("Layer")
    plt.ylabel("Jaccard")
    plt.title(f"{lang} ({span}): per-layer Jaccard (base vs ft)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(jacc_pdf)
    plt.close()

    return {"counts_pdf": counts_pdf, "jaccard_pdf": jacc_pdf}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base_pt",
        type=str,
        required=True,
        help="Path to base mask .pt for the selected span (src or tgt).",
    )
    ap.add_argument(
        "--ft_pt",
        type=str,
        required=True,
        help="Path to fine-tuned mask .pt for the selected span (src or tgt).",
    )
    ap.add_argument(
        "--span",
        type=str,
        choices=["src", "tgt"],
        required=True,
        help="Which span you are comparing (should match the .pt you pass).",
    )
    ap.add_argument("--langs", type=str, default="hi bn ta te")
    ap.add_argument("--out_dir", type=str, default="./compare_reports")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--save_json", action="store_true")
    args = ap.parse_args()

    langs = args.langs.split()
    if len(langs) == 0:
        raise ValueError("--langs must contain at least one language code")

    os.makedirs(args.out_dir, exist_ok=True)

    base = load_mask(args.base_pt)
    ft = load_mask(args.ft_pt)

    # If span is present in files, sanity-check it.
    base_span = base.get("span", None)
    ft_span = ft.get("span", None)
    if base_span is not None and base_span != args.span:
        raise ValueError(f"Base file span={base_span} but --span={args.span}. Use the matching .pt.")
    if ft_span is not None and ft_span != args.span:
        raise ValueError(f"FT file span={ft_span} but --span={args.span}. Use the matching .pt.")

    file_prefix = build_tag(args.base_pt, args.ft_pt, args.span, langs)

    print("Loaded base:", args.base_pt)
    print("Loaded ft  :", args.ft_pt)
    print("Span       :", args.span)
    print("Base languages:", base["languages"])
    print("FT languages  :", ft["languages"])
    print("Output dir    :", os.path.abspath(args.out_dir))
    print("File prefix   :", file_prefix)

    summary: Dict[str, Dict] = {
        "base_pt": os.path.abspath(args.base_pt),
        "ft_pt": os.path.abspath(args.ft_pt),
        "span": args.span,
        "langs": langs,
        "file_prefix": file_prefix,
        "per_language": {},
    }

    for lang in langs:
        base_layers = per_layer_sets(base, lang)
        ft_layers = per_layer_sets(ft, lang)

        stats = compare_layerwise(base_layers, ft_layers)
        print_lang_report(lang, args.span, stats)

        t = totals(stats)
        entry: Dict[str, object] = {
            "totals": t,
            "per_layer": [
                {
                    "layer": i,
                    "base": s.base,
                    "ft": s.ft,
                    "gained": s.gained,
                    "lost": s.lost,
                    "overlap": s.overlap,
                    "union": s.union,
                    "jaccard": s.jaccard,
                }
                for i, s in enumerate(stats)
            ],
        }

        if args.plot:
            paths = maybe_plot(args.out_dir, file_prefix, lang, args.span, stats)
            entry["plots"] = {k: os.path.abspath(v) for k, v in paths.items()}

        summary["per_language"][lang] = entry

    if args.save_json:
        out_json = os.path.join(args.out_dir, f"{file_prefix}__summary.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        print("Saved JSON:", os.path.abspath(out_json))

    print("Done. Output dir:", os.path.abspath(args.out_dir))


if __name__ == "__main__":
    main()
