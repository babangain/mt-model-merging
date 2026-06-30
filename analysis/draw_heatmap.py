import torch
import seaborn as sns
import matplotlib.pyplot as plt
import os

# -------- config --------
mask_path = "activation_mask_Qwen_Instruct/qwen-5"
languages = ["hi", "bn", "ta", "te"]
out_dir = "graphs"
os.makedirs(out_dir, exist_ok=True)
# ------------------------

final_indice = torch.load(mask_path)

num_langs = len(final_indice)
num_layers = len(final_indice[0])

# build [layer, lang] matrix
heatmap = torch.zeros(num_layers, num_langs, dtype=torch.int32)
for lang_id in range(num_langs):
    for layer_id in range(num_layers):
        heatmap[layer_id, lang_id] = final_indice[lang_id][layer_id].numel()

def plot_heatmap(data, layer_ids, title, filename):
    plt.figure(figsize=(7, 12))
    ax = sns.heatmap(
        data.numpy(),
        annot=True,
        fmt="d",
        cmap="viridis",
        linewidths=0.6,
        linecolor="black",
        xticklabels=languages,
        yticklabels=layer_ids,
        cbar_kws={"label": "Number of language-specific neurons"}
    )
    ax.set_xlabel("Language")
    ax.set_ylabel("Layer")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

# -----------------------------
# Version 1: with 0th layer
# -----------------------------
heatmap_with_0 = heatmap.flip(0)
layers_with_0 = list(reversed(range(num_layers)))

plot_heatmap(
    heatmap_with_0,
    layers_with_0,
    "Language-specific neuron distribution (before fine-tuning)",
    f"{out_dir}/heatmap_qwen_with_layer0.pdf"
)

# -----------------------------
# Version 2: without 0th layer
# -----------------------------
heatmap_no_0 = heatmap[1:].flip(0)
layers_no_0 = list(reversed(range(1, num_layers)))

plot_heatmap(
    heatmap_no_0,
    layers_no_0,
    "Language-specific neuron distribution (before fine-tuning)",
    f"{out_dir}/heatmap_qwen_without_layer0.pdf"
)

# import torch
# import seaborn as sns
# import matplotlib.pyplot as plt
# import os
# from collections import defaultdict

# # ---------------- config ----------------
# mask_path = "activation_mask_Qwen_Instruct/qwen-5"
# activation_path_prefix = "data_qwen_flores_indic_en"
# activation_suffix = "nemo"
# languages = ["hi", "bn", "ta", "te"]
# out_dir = "graphs"
# os.makedirs(out_dir, exist_ok=True)
# # ---------------------------------------

# # -------- load activation mask ----------
# final_indice = torch.load(mask_path)

# num_langs = len(languages)
# num_layers = len(final_indice[0])

# # -------- load over_zero + n -------------
# over_zero = []
# n = []

# for lang in languages:
#     data = torch.load(
#         f"{activation_path_prefix}/activation.{lang}.valid.{activation_suffix}"
#     )
#     over_zero.append(data["over_zero"])
#     n.append(data["n"])

# over_zero = torch.stack(over_zero, dim=-1)  # [layer, inter, lang]
# n = torch.tensor(n, dtype=torch.float32)    # [lang]

# # -------- activation probabilities ------
# activation_probs = over_zero / n
# activation_probs[torch.isnan(activation_probs)] = 0

# num_layers, inter_size, _ = activation_probs.shape

# # -------- activation threshold ----------
# activation_bar_ratio = 0.95
# flattened = activation_probs.flatten()
# activation_bar = flattened.kthvalue(
#     int(len(flattened) * activation_bar_ratio)
# ).values.item()

# # -------- active language mask -----------
# active_mask = activation_probs > activation_bar
# active_count = active_mask.sum(dim=-1)   # [layer, inter]

# # -------- neuron categories --------------
# single_lang = active_count == 1
# multi_lang = (active_count >= 2) & (active_count < num_langs)
# shared_lang = active_count == num_langs

# print("Single language neurons:", int(single_lang.sum().item()))
# print("Multi language neurons :", int(multi_lang.sum().item()))
# print("Shared neurons         :", int(shared_lang.sum().item()))

# # -------- language combination stats -----
# combo_counter = defaultdict(int)

# for l in range(num_layers):
#     for h in range(inter_size):
#         if multi_lang[l, h]:
#             langs = tuple(torch.where(active_mask[l, h])[0].tolist())
#             combo_counter[langs] += 1

# print("\nMulti-language combinations:")
# for combo, count in sorted(combo_counter.items(), key=lambda x: -x[1]):
#     combo_names = tuple(languages[i] for i in combo)
#     print(combo_names, count)

# # -------- layer x language-count heatmap -
# # columns: neurons active in {2, 3} languages
# heatmap_multi = torch.zeros(
#     num_layers, num_langs - 1, dtype=torch.int32
# )

# for l in range(num_layers):
#     for k in range(2, num_langs + 1):
#         if k < num_langs:
#             heatmap_multi[l, k - 2] = (active_count[l] == k).sum().int()

# # reverse layers so layer 0 is bottom
# heatmap_multi = heatmap_multi.flip(0)

# # -------- plot multi-language heatmap ----
# plt.figure(figsize=(7, 10))
# ax = sns.heatmap(
#     heatmap_multi.numpy(),
#     annot=True,
#     fmt="d",
#     cmap="magma",
#     linewidths=0.6,
#     linecolor="black",
#     xticklabels=[f"{i} languages" for i in range(2, num_langs)],
#     yticklabels=list(reversed(range(num_layers))),
#     cbar_kws={"label": "Number of neurons"}
# )

# ax.set_xlabel("Number of languages neuron is active in")
# ax.set_ylabel("Layer")
# ax.set_title("Multi-language specific neuron distribution")

# plt.tight_layout()
# plt.savefig(f"{out_dir}/multi_language_neuron_heatmap.pdf")
# plt.show()
