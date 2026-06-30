import os
import torch
import matplotlib.pyplot as plt

LANGS = ["hi", "bn", "ta", "te"]

def plot_layerwise(mask_dir="activation_mask", graph_dir="graphs"):
    os.makedirs(graph_dir, exist_ok=True)

    files = sorted(os.listdir(mask_dir))

    for fname in files:
        path = os.path.join(mask_dir, fname)
        masks = torch.load(path)  # [lang][layer][neurons]

        num_langs = len(masks)
        num_layers = len(masks[0])

        layer_counts = torch.zeros(num_langs, num_layers)

        for l in range(num_langs):
            for layer_id in range(num_layers):
                layer_counts[l, layer_id] = masks[l][layer_id].numel()

        plt.figure(figsize=(8, 5))
        for l in range(num_langs):
            plt.plot(
                range(num_layers),
                layer_counts[l],
                label=LANGS[l] if l < len(LANGS) else f"lang-{l}"
            )

        plt.xlabel("Layer")
        plt.ylabel("Number of selected neurons")
        plt.title(f"Layerwise neuron distribution: {fname}")
        plt.legend()
        plt.tight_layout()

        save_path = os.path.join(graph_dir, f"{fname}_layerwise.pdf")
        plt.savefig(save_path)
        plt.close()

        print(f"Saved: {save_path}")

if __name__ == "__main__":
    plot_layerwise()
