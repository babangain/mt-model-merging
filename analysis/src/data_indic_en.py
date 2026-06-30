from datasets import load_dataset
from transformers import AutoTokenizer
import torch
import os

# Configuration
lang_map = {
    "hi": "hindi",
    "bn": "bengali",
    "ta": "tamil",
    "te": "telugu",
}
languages = ["hi", "bn", "ta", "te"]

target_size_mb = 100
save_dir = "data_qwen_flores_indic_en"
os.makedirs(save_dir, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    use_fast=True
)

target_total_bytes = target_size_mb * 1024 * 1024

for lang in languages:
    flores_lang = lang_map[lang]
    dataset_name = f"baban/flores_{flores_lang}_en_valid"

    print(f"Processing language: {lang} ({dataset_name})")

    ds = load_dataset(dataset_name, split="validation")

    valid_ids = []
    total_bytes = 0

    for item in ds:
        input_text = item["input"]

        # Apply chat template (prompt only)
        messages = [{"role": "user", "content": input_text}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        ids = tokenizer.encode(prompt, add_special_tokens=False)

        byte_tensor = torch.LongTensor(ids).numpy().tobytes()
        size_bytes = len(byte_tensor)

        if total_bytes + size_bytes > target_total_bytes:
            break

        valid_ids.extend(ids)
        total_bytes += size_bytes

    tensor = torch.LongTensor(valid_ids)
    save_path = os.path.join(save_dir, f"id.{lang}.valid.nemo")
    torch.save(tensor, save_path)

    print(
        f"Saved valid: {len(tensor)} tokens, "
        f"{tensor.numpy().nbytes / 1024 / 1024:.2f} MB "
        f"to {save_path}"
    )
