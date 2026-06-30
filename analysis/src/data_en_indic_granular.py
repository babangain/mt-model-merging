#!/usr/bin/env python3
import os
import argparse
from typing import List, Tuple
import sys
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

LANG_MAP = {"hi": "hindi", "bn": "bengali", "ta": "tamil", "te": "telugu"}


def spans_to_token_mask(offsets: List[Tuple[int, int]], span: Tuple[int, int]) -> torch.Tensor:
    s, e = span
    m = torch.zeros(len(offsets), dtype=torch.bool)
    for i, (a, b) in enumerate(offsets):
        # Special tokens often show (0, 0) in offset_mapping
        if a == b == 0:
            continue
        # Overlap check
        if not (b <= s or a >= e):
            m[i] = True
    return m


def build_qwen_chat(tokenizer: AutoTokenizer, prefix: str, user_text: str, assistant_text: str) -> str:
    """
    Qwen template-style chat: user + assistant.
    This matches your current training format (no explicit system message).
    """
    messages = [
        {"role": "user", "content":  user_text},
        {"role": "assistant", "content": assistant_text},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def extract_indic_sentence(src_text: str, lang_code: str) -> str:
    """
    Extract Indic sentence from:
    "Translate the following Bengali sentence to English <Indic text>"
    """

    prefix = f"Translate the following English sentence to {LANG_MAP[lang_code].title()}"

    if src_text.startswith(prefix):
        remaining = src_text[len(prefix):].strip()
        #print(remaining)
        return prefix, remaining

    else:
        print("Error!")
        sys.exit(-1)



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer_name", type=str, required=True, help="e.g. Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--save_dir", type=str, required=True, help="e.g. data_masked_qwen/en_indic")
    ap.add_argument("--languages", type=str, default="hi bn ta te")
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--target_mb", type=int, default=100)
    ap.add_argument("--dataset_pattern", type=str, default="anonymous/flores_en_{lang}_valid")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    target_bytes = args.target_mb * 1024 * 1024
    langs = args.languages.split()

    for lang_code in langs:
        if lang_code not in LANG_MAP:
            raise ValueError(f"Unsupported lang_code: {lang_code}")

        flores_lang = LANG_MAP[lang_code]
        dataset_name = args.dataset_pattern.format(lang=flores_lang)
        print(f"[indic_en] lang={lang_code} dataset={dataset_name}")

        ds = load_dataset(dataset_name, split="validation")
        cols = ds.column_names
        if "input" not in cols or "output" not in cols:
            raise RuntimeError(f"Expected columns input/output, got: {cols}")

        input_ids_list = []
        src_mask_list = []
        tgt_mask_list = []
        lengths_list = []

        total_bytes = 0

        for item in ds:
            src_text = item["input"]   # instruction + Indic sentence
            tgt_text = item["output"]  # English translation

            # Keep original user text as-is (matches training distribution)
            user_text = src_text

            # Extract only Indic sentence for masking
            prefix, indic_sent = extract_indic_sentence(src_text, lang_code)
            if not indic_sent:
                continue

            chat_text = build_qwen_chat(tokenizer, prefix, user_text, tgt_text)
            # print(chat_text)
            # Locate spans inside the final rendered chat string
            src_start = chat_text.rfind(indic_sent)
            tgt_start = chat_text.rfind(tgt_text)
            if src_start < 0 or tgt_start < 0:
                continue

            src_span = (src_start, src_start + len(indic_sent))  # mask only Indic sentence
            tgt_span = (tgt_start, tgt_start + len(tgt_text))    # mask English target

            enc = tokenizer(
                chat_text,
                return_offsets_mapping=True,
                add_special_tokens=True,
                truncation=True,
                max_length=args.seq_len,
                padding="max_length",
            )

            input_ids = torch.tensor(enc["input_ids"], dtype=torch.long)
            offsets = enc["offset_mapping"]

            src_mask = spans_to_token_mask(offsets, src_span)
            tgt_mask = spans_to_token_mask(offsets, tgt_span)

            # Skip if truncation removed spans
            if src_mask.sum().item() == 0 or tgt_mask.sum().item() == 0:
                continue

            example_bytes = input_ids.numel() * input_ids.element_size()
            if total_bytes + example_bytes > target_bytes:
                break

            input_ids_list.append(input_ids)
            src_mask_list.append(src_mask)
            tgt_mask_list.append(tgt_mask)
            lengths_list.append(int((input_ids != tokenizer.pad_token_id).sum().item()))
            total_bytes += example_bytes

        if not input_ids_list:
            raise RuntimeError(f"No samples saved for {lang_code}. Reduce seq_len or check dataset.")

        input_ids_t = torch.stack(input_ids_list, dim=0)
        src_mask_t = torch.stack(src_mask_list, dim=0)
        tgt_mask_t = torch.stack(tgt_mask_list, dim=0)
        lengths_t = torch.tensor(lengths_list, dtype=torch.int32)

        torch.save(input_ids_t, os.path.join(args.save_dir, f"input_ids.{lang_code}.pt"))
        torch.save(src_mask_t, os.path.join(args.save_dir, f"src_mask.{lang_code}.pt"))
        torch.save(tgt_mask_t, os.path.join(args.save_dir, f"tgt_mask.{lang_code}.pt"))
        torch.save(lengths_t, os.path.join(args.save_dir, f"lengths.{lang_code}.pt"))

        print(
            f"  saved {lang_code}: N={input_ids_t.size(0)} T={input_ids_t.size(1)} "
            f"src_tokens={int(src_mask_t.sum().item())} tgt_tokens={int(tgt_mask_t.sum().item())}"
        )


if __name__ == "__main__":
    main()
