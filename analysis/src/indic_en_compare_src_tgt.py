#!/usr/bin/env python3
import os
import argparse
from typing import Any, Dict, List, Tuple

import torch


def load_mask_file(path: str) -> Tuple[List[str], List[List[torch.Tensor]]]:
    """
    Supports two formats:
      A) dict: {"languages": [...], "span": "src/tgt", "final_indices": List[lang][layer]=Tensor}
      B) raw list: List[lang][layer]=Tensor (language order must be provided by --languages)
    Returns: (languages, final_indices)
    """
    obj = torch.load(path, map_location="cpu")

    if isinstance(obj, dict) and "final_indices" in obj:
        langs = obj.get("languages", None)
        final_indices = obj["final_indices"]
        if langs is None:
            raise ValueError(f"{path}: dict has final_indices but missing languages")
        return list(langs), final_indices

    if isinstance(obj, list):
        # raw list format
        return [], obj

    raise ValueError(f"Unsupported file structure in {path}: type={type(obj)} keys={getattr(obj, 'keys', lambda: [])()}")


def to_set(t: torch.Tensor) -> set:
    if t.numel() == 0:
        return set()
    return set(t.tolist())


def per_layer_counts(final_indices: List[List[torch.Tensor]]) -> List[List[int]]:
    """
    final_indices: [lang][layer] tensor(neuron_ids)
    Returns counts: [lang][layer] int
    """
    counts = []
    for per_layer in final_indices:
        counts.append([int(x.numel()) for x in per_layer])
    return counts


def jaccard(a: torch.Tensor, b: torch.Tensor) -> float:
    sa = to_set(a)
    sb = to_set(b)
    if not sa and not sb:
        return 1.0
    inter = len(sa.intersection(sb))
    union = len(sa.union(sb))
    return inter / union if union > 0 else 0.0


def summarize_layer_shifts(layers: int, src_counts: List[int], tgt_counts: List[int]) -> Dict[str, Any]:
    """
    Returns summary stats and top layers by positive/negative shift.
    """
    diffs = [tgt_counts[i] - src_counts[i] for i in range(layers)]
    total_src = sum(src_counts)
    total_tgt = sum(tgt_counts)

    top_inc = sorted(range(layers), key=lambda i: diffs[i], reverse=True)[:5]
    top_dec = sorted(range(layers), key=lambda i: diffs[i])[:5]

    return {
        "total_src": total_src,
        "total_tgt": total_tgt,
        "diff_total": total_tgt - total_src,
        "top_increase_layers": [(i, diffs[i], src_counts[i], tgt_counts[i]) for i in top_inc],
        "top_decrease_layers": [(i, diffs[i], src_counts[i], tgt_counts[i]) for i in top_dec],
        "diffs": diffs,
    }


def print_table(langs: List[str], src_counts_ll: List[List[int]], tgt_counts_ll: List[List[int]]) -> None:
    L = len(src_counts_ll[0])
    header = ["lang", "total_src", "total_tgt"] + [f"L{i}" for i in range(L)]
    print("\nPer-layer counts (src vs tgt shown as src/tgt):")
    print("  " + " | ".join(header))
    print("  " + "-+-".join(["-" * len(h) for h in header]))

    for li, lang in enumerate(langs):
        row = [lang, str(sum(src_counts_ll[li])), str(sum(tgt_counts_ll[li]))]
        row += [f"{src_counts_ll[li][i]}/{tgt_counts_ll[li][i]}" for i in range(L)]
        print("  " + " | ".join(row))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, required=True, help="Path to src mask .pt")
    ap.add_argument("--tgt", type=str, required=True, help="Path to tgt mask .pt")
    ap.add_argument("--languages", type=str, default="hi bn ta te",
                    help="Used only if mask files do not store languages")
    ap.add_argument("--show_layers", type=int, default=10, help="How many layers to print in overlap summary")
    args = ap.parse_args()

    src_langs, src_final = load_mask_file(args.src)
    tgt_langs, tgt_final = load_mask_file(args.tgt)

    if not src_langs:
        src_langs = args.languages.split()
    if not tgt_langs:
        tgt_langs = args.languages.split()

    if src_langs != tgt_langs:
        raise ValueError(f"Language order mismatch:\n  src: {src_langs}\n  tgt: {tgt_langs}")

    langs = src_langs

    if len(src_final) != len(tgt_final):
        raise ValueError(f"Lang dimension mismatch: src={len(src_final)} tgt={len(tgt_final)}")

    num_lang = len(langs)
    num_layers = len(src_final[0])
    if any(len(src_final[i]) != num_layers for i in range(num_lang)) or any(len(tgt_final[i]) != num_layers for i in range(num_lang)):
        raise ValueError("Layer dimension mismatch inside final_indices")

    # Counts
    src_counts_ll = per_layer_counts(src_final)
    tgt_counts_ll = per_layer_counts(tgt_final)

    print(f"Loaded src: {args.src}")
    print(f"Loaded tgt: {args.tgt}")
    print(f"Languages: {langs}")
    print(f"Num layers: {num_layers}")

    # Table
    print_table(langs, src_counts_ll, tgt_counts_ll)

    # Per-language shift summary
    print("\nPer-language layer shift summaries (tgt - src):")
    for li, lang in enumerate(langs):
        summ = summarize_layer_shifts(num_layers, src_counts_ll[li], tgt_counts_ll[li])
        print(f"\n[{lang}] totals: src={summ['total_src']} tgt={summ['total_tgt']} diff={summ['diff_total']}")
        print("  Top increases (layer, diff, src, tgt):")
        for x in summ["top_increase_layers"]:
            print(f"    L{x[0]}: {x[1]}  (src={x[2]}, tgt={x[3]})")
        print("  Top decreases (layer, diff, src, tgt):")
        for x in summ["top_decrease_layers"]:
            print(f"    L{x[0]}: {x[1]}  (src={x[2]}, tgt={x[3]})")

    # Overlap (Jaccard) per layer per language
    print("\nSrc vs tgt overlap (Jaccard) per language, per layer:")
    # Also compute aggregate overlap per language
    for li, lang in enumerate(langs):
        js = []
        for l in range(num_layers):
            js.append(jaccard(src_final[li][l], tgt_final[li][l]))
        avg_j = sum(js) / len(js)
        print(f"\n[{lang}] avg_jaccard={avg_j:.4f}")
        # Print a compact view for first N layers and last N layers
        n = min(args.show_layers, num_layers)
        head = "  head: " + " ".join([f"L{i}:{js[i]:.2f}" for i in range(n)])
        tail = "  tail: " + " ".join([f"L{i}:{js[i]:.2f}" for i in range(num_layers - n, num_layers)])
        print(head)
        if num_layers > n:
            print(tail)

        # Lowest overlap layers (where src and tgt diverge most)
        worst = sorted(range(num_layers), key=lambda i: js[i])[:5]
        best = sorted(range(num_layers), key=lambda i: js[i], reverse=True)[:5]
        print("  Lowest overlap layers:", ", ".join([f"L{i}({js[i]:.2f})" for i in worst]))
        print("  Highest overlap layers:", ", ".join([f"L{i}({js[i]:.2f})" for i in best]))

    # Global overlap across all langs and layers
    all_js = []
    for li in range(num_lang):
        for l in range(num_layers):
            all_js.append(jaccard(src_final[li][l], tgt_final[li][l]))
    print(f"\nGlobal avg_jaccard across all langs+layers: {sum(all_js)/len(all_js):.4f}")


if __name__ == "__main__":
    main()
