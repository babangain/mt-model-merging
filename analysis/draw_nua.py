import json
import os
import matplotlib.pyplot as plt

# ============================================================
# Paths
# ============================================================
paths = {
    "XX->En": {
        "src": "./nua_outputs/nua.src.rate.json",
        "tgt": "./nua_outputs/nua.tgt.rate.json",
    },
    "En->XX": {
        "src": "./nua_outputs_en_indic/nua.src.rate.json",
        "tgt": "./nua_outputs_en_indic/nua.tgt.rate.json",
    },
}

OUT_DIR = "./nua_plots"
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# Plot style: bigger and bold fonts
# ============================================================
plt.rcParams.update({
    "font.size": 18,
    "font.weight": "bold",
    "axes.labelsize": 20,
    "axes.labelweight": "bold",
    "axes.titlesize": 21,
    "axes.titleweight": "bold",
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# Helpers
# ============================================================
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def identify_base(models):
    for model_name in models:
        if model_name.startswith("Qwen_") or model_name.startswith("Qwen/"):
            return model_name
    return models[0]


def layerwise_avg(data, pair_keys):
    pairs_layerwise = data["pairs_layerwise"]
    num_layers = data["meta"].get("num_layers", 36)

    sums = [0.0] * num_layers
    counts = [0] * num_layers

    for pair_key in pair_keys:
        if pair_key not in pairs_layerwise:
            continue

        pair_data = pairs_layerwise[pair_key]

        for layer_idx in range(num_layers):
            layer_key = f"layer_{layer_idx:02d}"
            if layer_key in pair_data:
                sums[layer_idx] += float(pair_data[layer_key])
                counts[layer_idx] += 1

    averages = [
        sums[layer_idx] / counts[layer_idx] if counts[layer_idx] else float("nan")
        for layer_idx in range(num_layers)
    ]

    return averages


def compute_curves(path):
    data = load_json(path)

    models = data["meta"]["models"]
    base_model = identify_base(models)
    ft_models = [model_name for model_name in models if model_name != base_model]

    existing_pairs = set(data["pairs_layerwise"].keys())

    def get_pair_key(model_a, model_b):
        key_ab = f"{model_a}|||{model_b}"
        key_ba = f"{model_b}|||{model_a}"

        if key_ab in existing_pairs:
            return key_ab
        if key_ba in existing_pairs:
            return key_ba
        return None

    base_ft_keys = []
    for ft_model in ft_models:
        pair_key = get_pair_key(base_model, ft_model)
        if pair_key is not None:
            base_ft_keys.append(pair_key)

    ft_ft_keys = []
    for i in range(len(ft_models)):
        for j in range(i + 1, len(ft_models)):
            pair_key = get_pair_key(ft_models[i], ft_models[j])
            if pair_key is not None:
                ft_ft_keys.append(pair_key)

    base_ft_avg = layerwise_avg(data, base_ft_keys)
    ft_ft_avg = layerwise_avg(data, ft_ft_keys)

    num_layers = data["meta"].get("num_layers", 36)

    return base_ft_avg, ft_ft_avg, base_model, ft_models, num_layers


def make_safe_filename(direction):
    return (
        direction
        .replace(">", "to")
        .replace("–", "-")
        .replace(" ", "_")
    )


# ============================================================
# Main plotting
# ============================================================
out_files = []

for direction, spans in paths.items():
    curves = {}
    num_layers = 36

    for span, path in spans.items():
        base_ft_avg, ft_ft_avg, base_model, ft_models, num_layers = compute_curves(path)

        curves[(span, "Base-FT")] = base_ft_avg
        curves[(span, "FT-FT")] = ft_ft_avg

    xs = list(range(num_layers))

    plt.figure(figsize=(11, 6))

    for (span, kind), values in curves.items():
        label = f"{span.upper()} {kind}"
        plt.plot(
            xs,
            values,
            linewidth=3.0,
            label=label,
        )

    plt.ylim(0.7, 1.01)
    plt.xlim(0, num_layers - 1)

    plt.xlabel("Layer", fontweight="bold")
    plt.ylabel("Average NUA", fontweight="bold")
    plt.title(
        f"{direction}: Average Base-FT vs FT-FT Neuron Usage Alignment",
        fontweight="bold",
        pad=14,
    )

    plt.grid(True, alpha=0.30)

    legend = plt.legend(
        frameon=False,
        fontsize=15,
        ncol=2,
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

    out_path = os.path.join(
        OUT_DIR,
        f"nua_avg_{make_safe_filename(direction)}.pdf",
    )

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    out_files.append(out_path)

print(out_files)