#!/usr/bin/env python3
import os
import argparse
from typing import List

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--act_dir", type=str, required=True)
    ap.add_argument("--languages", type=str, default="hi bn ta te")
    ap.add_argument("--span", choices=["src", "tgt"], required=True)
    ap.add_argument("--top_rate", type=float, default=0.1)
    ap.add_argument("--filter_rate", type=float, default=0.0)
    ap.add_argument("--activation_bar_ratio", type=float, default=0.8)
    ap.add_argument("--save_path", type=str, required=True)
    args = ap.parse_args()

    langs = args.languages.split()

    n_list = []
    over_list = []
    for lang in langs:
        p = os.path.join(args.act_dir, f"activation.{lang}.{args.span}.pt")
        obj = torch.load(p, map_location="cpu")
        n_list.append(obj["n"])
        over_list.append(obj["over_zero"])

    n = torch.tensor(n_list, dtype=torch.float32)  # [lang]
    over_zero = torch.stack(over_list, dim=-1).float()  # [L, I, lang]

    num_layers, intermediate_size, lang_num = over_zero.shape
    print("num_layers", num_layers, "intermediate_size", intermediate_size, "lang_num", lang_num)

    activation_probs = over_zero / n.view(1, 1, -1).clamp_min(1.0)

    denom = activation_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    normed = activation_probs / denom

    log_probs = torch.where(normed > 0, normed.log(), torch.zeros_like(normed))
    entropy = -torch.sum(normed * log_probs, dim=-1)  # [L, I]

    if torch.isnan(entropy).any():
        raise ValueError("NaN entropy encountered")

    total_neurons = num_layers * intermediate_size
    print("Total neurons:", total_neurons)

    flat_probs = activation_probs.flatten()
    k1 = max(1, round(len(flat_probs) * args.filter_rate))
    prob_bar = flat_probs.kthvalue(k1).values.item()
    print("filter bar value =", prob_bar)

    top_position = (activation_probs > prob_bar).sum(dim=-1)  # [L, I]
    kept = int((top_position != 0).sum().item())
    print("Neurons after first bar:", kept)

    entropy2 = entropy.clone()
    entropy2[top_position == 0] = float("inf")

    flat_entropy = entropy2.flatten()
    k2 = max(1, round(len(flat_entropy) * args.top_rate))
    _, idx = flat_entropy.topk(k2, largest=False)

    row_index = idx // intermediate_size
    col_index = idx % intermediate_size

    selected_probs = activation_probs[row_index, col_index]  # [k2, lang]
    selected_probs_t = selected_probs.transpose(0, 1)        # [lang, k2]

    k3 = max(1, round(len(flat_probs) * args.activation_bar_ratio))
    act_bar = flat_probs.kthvalue(k3).values.item()
    print("second bar value =", act_bar)

    per_lang_counts = (selected_probs_t > act_bar).sum(dim=1).tolist()
    print("per-lang counts above second bar:", per_lang_counts)

    lang_ids, neuron_ids = torch.where(selected_probs_t > act_bar)
    merged_index = torch.stack((row_index, col_index), dim=-1)  # [k2,2]

    final_indices: List[List[torch.Tensor]] = []
    splits = torch.bincount(lang_ids, minlength=lang_num).tolist()

    pos = 0
    for li, cnt in enumerate(splits):
        sub = merged_index[neuron_ids[pos:pos + cnt]]
        pos += cnt

        pairs = [tuple(x.tolist()) for x in sub]
        pairs.sort()

        layer_bins: List[List[int]] = [[] for _ in range(num_layers)]
        for l, h in pairs:
            layer_bins[l].append(h)

        per_layer = []
        for l in range(num_layers):
            per_layer.append(torch.tensor(layer_bins[l], dtype=torch.long))
        final_indices.append(per_layer)

    total_selected = sum(t.numel() for per_lang in final_indices for t in per_lang)
    print("Total selected neurons:", total_selected)

    torch.save({"languages": langs, "span": args.span, "final_indices": final_indices}, args.save_path)
    print("Saved:", args.save_path)


if __name__ == "__main__":
    main()
