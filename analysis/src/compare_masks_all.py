import os
import torch

LANGS = ["hi", "bn", "ta", "te"]

BASE_DIR = "activation_mask_Qwen_Instruct"
FT_DIRS = {
    "hi-en": "activation_mask_QwenTranslate_Hindi_English",
    "bn-en": "activation_mask_QwenTranslate_Bengali_English",
    "ta-en": "activation_mask_QwenTranslate_Tamil_English",
    "te-en": "activation_mask_QwenTranslate_Telugu_English",
}

MODEL_NAME = "qwen-5"


def load_total_counts(mask_path):
    """
    Returns total number of selected neurons per language.
    """
    masks = torch.load(mask_path)  # [lang][layer][neurons]
    num_langs = len(masks)

    totals = []
    for l in range(num_langs):
        count = sum(layer.numel() for layer in masks[l])
        totals.append(count)

    return totals  # [lang]


def main():
    base_path = os.path.join(BASE_DIR, MODEL_NAME)
    base_counts = load_total_counts(base_path)

    print(f"\n=== Baseline: pretrained (qwen-5) ===")
    for l, lang in enumerate(LANGS):
        print(f"{lang}: {base_counts[l]}")

    for ft_name, ft_dir in FT_DIRS.items():
        ft_path = os.path.join(ft_dir, MODEL_NAME)

        if not os.path.exists(ft_path):
            print(f"\nSkipping {ft_name}, mask not found")
            continue

        ft_counts = load_total_counts(ft_path)

        print(f"\n=== Fine tuned on {ft_name} ===")
        for l, lang in enumerate(LANGS):
            delta = ft_counts[l] - base_counts[l]
            sign = "+" if delta >= 0 else ""
            print(
                f"{lang}: "
                f"{ft_counts[l]} "
                f"({sign}{delta})"
            )


if __name__ == "__main__":
    main()
