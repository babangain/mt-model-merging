import os
import torch
import matplotlib.pyplot as plt

LANGS = ["hi", "bn", "ta", "te"]

BASE_DIR = "activation_mask_Qwen_Instruct"
FT_DIRS = {
    "AnonMT_Hindi_English": "activation_mask_AnonMT_Hindi_English",
    "AnonMT_Bengali_English": "activation_mask_AnonMT_Bengali_English",
    "AnonMT_Tamil_English": "activation_mask_AnonMT_Tamil_English",
    "AnonMT_Telugu_English": "activation_mask_AnonMT_Telugu_English",
}

MODEL_ID = "qwen-5"
GRAPH_DIR = "graphs_layerwise_combined"
os.makedirs(GRAPH_DIR, exist_ok=True)


def load_layer_counts(mask_path):
    """
    Returns tensor of shape [lang, layer]
    """
    masks = torch.load(mask_path)  # [lang][layer][neurons]
    num_langs = len(masks)
    num_layers = len(masks[0])

    counts = torch.zeros(num_langs, num_layers)
    for l in range(num_langs):
        for layer_id in range(num_layers):
            counts[l, layer_id] = masks[l][layer_id].numel()

    return counts


# def main():
#     base_path = os.path.join(BASE_DIR, MODEL_ID)
#     base_counts = load_layer_counts(base_path)

#     for model_name, ft_dir in FT_DIRS.items():
#         ft_path = os.path.join(ft_dir, MODEL_ID)

#         if not os.path.exists(ft_path):
#             print(f"Skipping {model_name}, mask not found")
#             continue

#         ft_counts = load_layer_counts(ft_path)
#         delta = ft_counts - base_counts  # [lang, layer]

#         num_layers = delta.shape[1]

#         plt.figure(figsize=(8, 5))
#         for l, lang in enumerate(LANGS):
#             plt.plot(
#                 range(num_layers),
#                 delta[l],
#                 linewidth=2,
#                 label=lang
#             )

#         plt.axhline(0, linestyle="--", linewidth=1)

#         plt.xlabel("Transformer layer")
#         plt.ylabel("Δ number of language-specific neurons\n(Fine-tuned − pretrained)")
#         plt.title(
#             f"{model_name}\n"
#             f"Baseline: Qwen_Instruct"
#         )
#         plt.legend(title="Probed language")
#         plt.tight_layout()

#         save_path = os.path.join(
#             GRAPH_DIR,
#             f"{MODEL_ID}_{model_name}_layerwise_delta_combined.pdf"
#         )
#         plt.savefig(save_path)
#         plt.close()

#         print(f"Saved: {save_path}")

def main():
    base_path = os.path.join(BASE_DIR, MODEL_ID)
    base_counts = load_layer_counts(base_path)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharey=True)
    axes = axes.flatten()

    plot_idx = 0

    for model_name, ft_dir in FT_DIRS.items():
        ft_path = os.path.join(ft_dir, MODEL_ID)

        if not os.path.exists(ft_path):
            print(f"Skipping {model_name}, mask not found")
            continue

        ft_counts = load_layer_counts(ft_path)
        delta = ft_counts - base_counts  # [lang, layer]

        num_layers = delta.shape[1]
        ax = axes[plot_idx]

        for l, lang in enumerate(LANGS):
            ax.plot(
                range(num_layers),
                delta[l],
                linewidth=2,
                label=lang
            )

        ax.axhline(0, linestyle="--", linewidth=1)
        ax.set_title(model_name)
        ax.set_xlabel("Transformer layer")

        if plot_idx % 2 == 0:
            ax.set_ylabel("Δ # language-specific neurons\n(Fine-tuned − pretrained)")

        plot_idx += 1

    # single legend for all subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Probed language",
        loc="lower center",
        ncol=len(LANGS),
        bbox_to_anchor=(0.5, 0.02)
    )

    fig.suptitle(
        f"Layer-wise Change in Language-Specific Neurons",
        fontsize=14
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, top=0.9)

    save_path = os.path.join(
        GRAPH_DIR,
        f"{MODEL_ID}_layerwise_delta_combined_2x2.pdf"
    )
    plt.savefig(save_path)
    plt.close()

    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
