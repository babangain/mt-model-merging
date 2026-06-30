#!/usr/bin/env python3
"""
Export neuron-count matrices to JSON for MULTIPLE models and BOTH spans (src.pt, tgt.pt).

What it does
- You give a root directory that contains per-model subfolders (or pt files directly).
- For each model you specify, it loads:
    <model_dir>/src.pt
    <model_dir>/tgt.pt
  and exports JSONs with:
    - languages (file order)
    - span
    - num_layers
    - matrix[layer][lang_idx] = count of selected neurons (numel of final_indices entry)
    - summaries (per-layer totals, per-lang totals, grand total)
- Also writes a single combined JSON that contains all models + both spans.

This replaces plotting and supports your "5 models incl Qwen instruct" use case.

Example (Indic->En):
python plot_counts_json.py \
    --root ./activation_mask/indic_en \
    --models \
      Qwen_Qwen2.5-3B-Instruct \
      baban_QwenTranslate_Hindi_English \
      baban_QwenTranslate_Bengali_English \
      baban_QwenTranslate_Tamil_English \
      baban_QwenTranslate_Telugu_English \
    --out_dir ./matrix_json_indic_en

Example (En->Indic):
  python plot_counts_json.py \
    --root ./activation_mask/en_indic \
    --models \
      Qwen_Qwen2.5-3B-Instruct \
      baban_QwenTranslate_English_Hindi \
      baban_QwenTranslate_English_Bengali \
      baban_QwenTranslate_English_Tamil \
      baban_QwenTranslate_English_Telugu \
    --out_dir ./matrix_json_en_indic

Notes
- If a model folder is missing src.pt or tgt.pt, the script errors by default.
  Use --skip_missing to continue and just warn.
- The JSON matrix is indexed as matrix[layer][lang_idx] with layer 0 first.

Input format (torch.load):
  {
    "languages": [...],
    "final_indices": [lang_idx][num_layers] tensors,
    "span": "src" or "tgt" (optional)
  }
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass
class MatrixResult:
    model: str
    span: str
    source_pt: str
    languages: List[str]
    num_layers: int
    matrix: List[List[int]]  # [layer][lang]
    summaries: Dict[str, Any]


def _require_file(path: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing file: {path}")


def load_mask(path: str) -> Dict[str, Any]:
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

    if len(fi) != len(langs):
        raise ValueError(
            f"{path}: languages has {len(langs)} entries but final_indices has {len(fi)}"
        )

    span = obj.get("span", None)
    if span is not None and span not in ("src", "tgt"):
        raise ValueError(f"{path} span={span}, expected 'src' or 'tgt' (or missing)")

    return obj


def safe_stem(s: str) -> str:
    # safe for filenames
    return s.replace("/", "_").replace(" ", "_")


def build_counts_matrix_all_langs(obj: Dict[str, Any]) -> Tuple[List[List[int]], int, List[str]]:
    langs: List[str] = list(obj["languages"])
    fi = obj["final_indices"]

    first = fi[0]
    if not isinstance(first, (list, tuple)) or len(first) == 0:
        raise ValueError("final_indices[0] is not a non-empty list/tuple")
    num_layers = len(first)

    matrix: List[List[int]] = [[0 for _ in range(len(langs))] for _ in range(num_layers)]

    for lang_idx, lang in enumerate(langs):
        per_layer = fi[lang_idx]
        if not isinstance(per_layer, (list, tuple)):
            raise TypeError(f"final_indices[{lang}] is {type(per_layer)}, expected list/tuple")
        if len(per_layer) != num_layers:
            raise ValueError(
                f"Layer mismatch for lang={lang}: expected {num_layers}, got {len(per_layer)}"
            )
        for layer_idx, t in enumerate(per_layer):
            if not isinstance(t, torch.Tensor):
                raise TypeError(
                    f"final_indices[{lang}][{layer_idx}] is {type(t)}, expected torch.Tensor"
                )
            matrix[layer_idx][lang_idx] = int(t.numel())

    return matrix, num_layers, langs


def compute_summaries(matrix: List[List[int]]) -> Dict[str, Any]:
    num_layers = len(matrix)
    num_langs = len(matrix[0]) if num_layers > 0 else 0

    per_layer_total = [sum(matrix[l]) for l in range(num_layers)]
    per_lang_total = [sum(matrix[l][j] for l in range(num_layers)) for j in range(num_langs)]
    grand_total = sum(per_layer_total)

    return {
        "per_layer_total": per_layer_total,
        "per_lang_total": per_lang_total,
        "grand_total": grand_total,
    }


def export_one_pt(model: str, pt_path: str, expected_span: str) -> MatrixResult:
    obj = load_mask(pt_path)
    span = obj.get("span", expected_span)  # tolerate missing span key
    if span != expected_span:
        # Not fatal, but often indicates file mismatch
        raise ValueError(f"{pt_path}: expected span='{expected_span}' but found '{span}'")

    matrix, num_layers, langs = build_counts_matrix_all_langs(obj)
    summaries = compute_summaries(matrix)

    return MatrixResult(
        model=model,
        span=span,
        source_pt=os.path.abspath(pt_path),
        languages=langs,
        num_layers=num_layers,
        matrix=matrix,
        summaries=summaries,
    )


def write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root directory that contains per-model folders (each with src.pt and tgt.pt).",
    )
    ap.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Model folder names under --root. Example: Qwen_Qwen2.5-3B-Instruct baban_QwenTranslate_Telugu_English ...",
    )
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument(
        "--skip_missing",
        action="store_true",
        help="If set, skip models/spans where src.pt or tgt.pt is missing.",
    )
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    combined: Dict[str, Any] = {
        "root": root,
        "models": {},
    }

    failures: List[str] = []

    for model in args.models:
        model_dir = os.path.join(root, model)
        src_pt = os.path.join(model_dir, "src.pt")
        tgt_pt = os.path.join(model_dir, "tgt.pt")

        model_block: Dict[str, Any] = {}
        combined["models"][model] = model_block

        for span, pt_path in (("src", src_pt), ("tgt", tgt_pt)):
            if not os.path.isfile(pt_path):
                msg = f"[missing] {model} {span}: {pt_path}"
                if args.skip_missing:
                    print("WARN:", msg)
                    model_block[span] = {"missing": True, "path": pt_path}
                    continue
                failures.append(msg)
                continue

            try:
                res = export_one_pt(model=model, pt_path=pt_path, expected_span=span)
            except Exception as e:
                msg = f"[error] {model} {span}: {pt_path} -> {repr(e)}"
                if args.skip_missing:
                    print("WARN:", msg)
                    model_block[span] = {"error": str(e), "path": pt_path}
                    continue
                failures.append(msg)
                continue

            payload = {
                "model": res.model,
                "span": res.span,
                "source_pt": res.source_pt,
                "num_layers": res.num_layers,
                "languages": res.languages,
                "matrix": res.matrix,         # matrix[layer][lang_idx]
                "summaries": res.summaries,
            }

            # Write per-model-per-span JSON
            out_json = os.path.join(
                out_dir,
                f"{safe_stem(model)}__span={span}__layers={res.num_layers}.json",
            )
            write_json(out_json, payload)

            # Add to combined JSON
            model_block[span] = payload

            print(f"OK: {model} {span} -> {out_json}")

    # If strict mode and any failures occurred, raise at end (after reporting)
    if failures and not args.skip_missing:
        print("\nErrors encountered:")
        for m in failures:
            print(" -", m)
        raise SystemExit(1)

    # Write combined JSON
    combined_path = os.path.join(out_dir, "ALL_MODELS__src_tgt__counts_matrices.json")
    write_json(combined_path, combined)
    print("\nSaved combined JSON:", combined_path)


if __name__ == "__main__":
    main()