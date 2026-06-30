#!/usr/bin/env python3
"""
Plot selected principal-angle curves across layers from:
  combined.tgt_principal_angles_and_maps.json

Curves shown:
  - Base vs Hi
  - Base vs Bn
  - Base vs Ta
  - Base vs Te
  - Hi vs Bn
  - Hi vs Te
  - Ta vs Te

Short labels only: Base, Hi, Bn, Ta, Te.

Outputs:
  principal_angles_selected.png
  principal_angles_selected.pdf
"""

import json
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================
JSON_PATH = Path(
    "/home/baban/scripts/Language-Neurons-Manipulation/"
    "geom_outputs_en_indic_all/combined.tgt_principal_angles_and_maps.json"
)

OUT_PNG = Path("principal_angles_selected.png")
OUT_PDF = Path("principal_angles_selected.pdf")


# ============================================================
# Models
# ============================================================
BASE = "Qwen/Qwen2.5-3B-Instruct"

MODELS = {
    "Hi": "baban/QwenTranslate_English_Hindi",
    "Bn": "baban/QwenTranslate_English_Bengali",
    "Ta": "baban/QwenTranslate_English_Tamil",
    "Te": "baban/QwenTranslate_English_Telugu",
}

CURVES = [
    ("Base", "Hi"),
    ("Base", "Bn"),
    ("Base", "Ta"),
    ("Base", "Te"),
    ("Hi", "Bn"),
    ("Hi", "Te"),
    ("Ta", "Te"),
]


# ============================================================
# Plot style: bigger and bold fonts
# ============================================================
plt.rcParams.update({
    "font.size": 18,
    "font.weight": "bold",
    "axes.labelsize": 21,
    "axes.labelweight": "bold",
    "axes.titlesize": 22,
    "axes.titleweight": "bold",
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 15,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# Helpers
# ============================================================
def layer_idx(layer_key: str) -> int:
    return int(layer_key.split("_")[1])


def pair_key(model_a: str, model_b: str) -> str:
    return f"{model_a}|||{model_b}"


def resolve_model(label: str) -> str:
    if label == "Base":
        return BASE

    if label in MODELS:
        return MODELS[label]

    raise ValueError(f"Unknown label: {label}")


def get_angle_for_layer(median_angle: Dict[str, float], label_a: str, label_b: str) -> float:
    """
    Pair keys in JSON may be either A|||B or B|||A.
    """
    model_a = resolve_model(label_a)
    model_b = resolve_model(label_b)

    key_ab = pair_key(model_a, model_b)
    if key_ab in median_angle:
        return float(median_angle[key_ab])

    key_ba = pair_key(model_b, model_a)
    if key_ba in median_angle:
        return float(median_angle[key_ba])

    raise KeyError(f"Pair not found in JSON: {label_a} vs {label_b}")


def load_series(data: Dict) -> Tuple[np.ndarray, Dict[Tuple[str, str], np.ndarray]]:
    per_layer = data["per_layer"]

    layer_keys = sorted(per_layer.keys(), key=layer_idx)
    layers = np.array([layer_idx(layer_key) for layer_key in layer_keys], dtype=int)

    series: Dict[Tuple[str, str], List[float]] = {
        curve: [] for curve in CURVES
    }

    for layer_key in layer_keys:
        median_angle = per_layer[layer_key]["median_angle"]

        for curve in CURVES:
            label_a, label_b = curve
            angle = get_angle_for_layer(median_angle, label_a, label_b)
            series[curve].append(angle)

    series_np = {
        curve: np.array(values, dtype=float)
        for curve, values in series.items()
    }

    return layers, series_np


# ============================================================
# Main
# ============================================================
def main() -> None:
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    layers, series = load_series(data)

    plt.figure(figsize=(11, 6.2))

    for (label_a, label_b), values in series.items():
        plt.plot(
            layers,
            values,
            linewidth=3.0,
            label=f"{label_a}-{label_b}",
        )

    plt.xlabel("Layer", fontweight="bold")
    plt.ylabel("Median Principal Angle (degrees)", fontweight="bold")
    plt.title(
        "Target-Span Principal Angles Across Layers",
        fontweight="bold",
        pad=14,
    )

    plt.grid(True, alpha=0.30)

    legend = plt.legend(
        ncol=2,
        frameon=False,
        fontsize=15,
        loc="best",
    )

    for text in legend.get_texts():
        text.set_fontweight("bold")

    ax = plt.gca()

    for tick in ax.get_xticklabels():
        tick.set_fontweight("bold")

    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    plt.tight_layout()

    plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.savefig(OUT_PDF, bbox_inches="tight")
    plt.show()

    print(f"Saved:\n  {OUT_PNG}\n  {OUT_PDF}")


if __name__ == "__main__":
    main()