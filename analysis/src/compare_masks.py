import os
import torch
import matplotlib.pyplot as plt

LANGS = ["hi", "bn", "ta", "te"]

BEFORE_DIR = "activation_mask"
AFTER_DIR = "activation_mask_QwenTranslate_Hindi_En"
GRAPH_DIR = "graphs_comparison"

os.makedirs(GRAPH_DIR, exist_ok=True)

def load_layer_counts(mask_path):
    masks = torch.load(mask_path)  # [lang][layer][neurons]
    num_langs = len(masks)
    num_layers = len(masks[0])

    counts = torch.zeros(num_langs, num_layers)
    for l in range(num_langs):
        for layer_id in range(num_layers):
            counts[l, layer_id] = masks[l][layer_id].numel()

    return counts  # [lang, layer]


def compare_masks():
    files = sorted(os.listdir(BEFORE_DIR))

    for fname in files:
        before_path = os.path.join(BEFORE_DIR, fname)
        after_path = os.path.join(AFTER_DIR, fname)

        if not os.path.exists(after_path):
            print(f"Skipping {fname}, not found in after-ft directory")
            continue

        before_counts = load_layer_counts(before_path)
        after_counts = load_layer_counts(after_path)

        delta = after_counts - before_counts  # positive = increase

        num_langs, num_layers = delta.shape

        # -------- Layerwise delta plots --------
        plt.figure(figsize=(8, 5))
        for l in range(num_langs):
            plt.plot(
                range(num_layers),
                delta[l],
                label=LANGS[l] if l < len(LANGS) else f"lang-{l}"
            )

        plt.axhline(0, linestyle="--", linewidth=1)
        plt.xlabel("Layer")
        plt.ylabel("Δ selected neurons (after - before)")
        plt.title(f"Layerwise neuron change: {fname}")
        plt.legend()
        plt.tight_layout()

        save_path = os.path.join(GRAPH_DIR, f"{fname}_delta_layerwise.pdf")
        plt.savefig(save_path)
        plt.close()

        # -------- Numeric summary --------
        print(f"\n=== {fname} ===")
        for l in range(num_langs):
            total_before = before_counts[l].sum().item()
            total_after = after_counts[l].sum().item()
            total_delta = delta[l].sum().item()

            sign = "increase" if total_delta > 0 else "decrease"
            print(
                f"{LANGS[l]} | "
                f"before: {int(total_before)} | "
                f"after: {int(total_after)} | "
                f"net {sign}: {int(total_delta)}"
            )

        print(f"Saved delta plot: {save_path}")


if __name__ == "__main__":
    compare_masks()
