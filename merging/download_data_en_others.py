#!/usr/bin/env python3

import os
import json
from itertools import islice
from datasets import load_dataset


# ============================================================
# Settings
# ============================================================

BASE_DIR = "/workspace"
OUTPUT_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_TRAIN_EXAMPLES = 1_000_000

LANGUAGES = {
    "fr": {
        "name": "French",
        "train_source": "wmt14",
        "dataset_name": "wmt/wmt14",
        "dataset_config": "fr-en",
        "src_key": "en",
        "target_key": "fr",
        "flores_code": "fra_Latn",
    },
    "de": {
        "name": "German",
        "train_source": "wmt14",
        "dataset_name": "wmt/wmt14",
        "dataset_config": "de-en",
        "src_key": "en",
        "target_key": "de",
        "flores_code": "deu_Latn",
    },
    "ar": {
        "name": "Arabic",
        "train_source": "opus100",
        "dataset_name": "Helsinki-NLP/opus-100",
        "dataset_config": "ar-en",
        "src_key": "en",
        "target_key": "ar",
        "flores_code": "arb_Arab",
    },
    "uk": {
        "name": "Ukrainian",
        "train_source": "opus100",
        "dataset_name": "Helsinki-NLP/opus-100",
        "dataset_config": "en-uk",
        "src_key": "en",
        "target_key": "uk",
        "flores_code": "ukr_Cyrl",
    },
}

FLORES_DATASET = "facebook/flores"
FLORES_CONFIG = "all"
FLORES_EN_CODE = "eng_Latn"


# ============================================================
# Helpers
# ============================================================

def make_prompt(lang_name: str) -> str:
    return f"Translate the following English sentence to {lang_name}"


def save_json(records, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def flores_column(code: str) -> str:
    if code.startswith("sentence_"):
        return code

    return f"sentence_{code}"


def extract_translation(item, src_key: str, tgt_key: str):
    if "translation" not in item:
        raise KeyError(
            f"Expected key 'translation'. Available item keys: {list(item.keys())}"
        )

    trans = item["translation"]

    if src_key not in trans:
        raise KeyError(
            f"Source key '{src_key}' not found in translation. "
            f"Available translation keys: {list(trans.keys())}"
        )

    if tgt_key not in trans:
        raise KeyError(
            f"Target key '{tgt_key}' not found in translation. "
            f"Available translation keys: {list(trans.keys())}"
        )

    return trans[src_key], trans[tgt_key]


def make_record(src: str, tgt: str, lang_name: str) -> dict:
    return {
        "instruction": make_prompt(lang_name),
        "input": src,
        "output": tgt,
    }


def preview_dataset_keys(dataset_name: str, dataset_config: str) -> None:
    print("\n[CHECK] Previewing one example", flush=True)
    print(f"[CHECK] Dataset: {dataset_name}", flush=True)
    print(f"[CHECK] Config:  {dataset_config}", flush=True)

    ds = load_dataset(
        dataset_name,
        dataset_config,
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    first_item = next(iter(ds))

    print(f"[CHECK] Top-level keys: {list(first_item.keys())}", flush=True)

    if "translation" in first_item:
        print(
            f"[CHECK] Translation keys: {list(first_item['translation'].keys())}",
            flush=True,
        )
    else:
        print(f"[CHECK] First item: {first_item}", flush=True)


# ============================================================
# Train builder
# ============================================================

def build_parallel_train(lang_code: str, lang_info: dict) -> int:
    lang_name = lang_info["name"]
    dataset_name = lang_info["dataset_name"]
    dataset_config = lang_info["dataset_config"]
    src_key = lang_info["src_key"]
    target_key = lang_info["target_key"]

    output_path = os.path.join(OUTPUT_DIR, f"MT_En_{lang_name}.json")

    print("\n" + "=" * 80, flush=True)
    print(f"Building train: English to {lang_name} ({lang_code})", flush=True)
    print("=" * 80, flush=True)
    print(f"Dataset:      {dataset_name}", flush=True)
    print(f"Config:       {dataset_config}", flush=True)
    print(f"Source key:   {src_key}", flush=True)
    print(f"Target key:   {target_key}", flush=True)
    print(f"Output:       {output_path}", flush=True)
    print(f"Limit:        {MAX_TRAIN_EXAMPLES:,}", flush=True)
    print("Filtering:    none", flush=True)

    preview_dataset_keys(dataset_name, dataset_config)

    ds = load_dataset(
        dataset_name,
        dataset_config,
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    records = []

    for idx, item in enumerate(islice(ds, MAX_TRAIN_EXAMPLES), start=1):
        src, tgt = extract_translation(
            item=item,
            src_key=src_key,
            tgt_key=target_key,
        )

        records.append(make_record(src, tgt, lang_name))

        if idx % 50_000 == 0:
            print(
                f"Processed {idx:,} train examples for English to {lang_name}",
                flush=True,
            )

    save_json(records, output_path)

    print(f"Saved train split to {output_path}", flush=True)
    print(f"Total train examples saved: {len(records):,}", flush=True)

    if len(records) < MAX_TRAIN_EXAMPLES:
        print(
            f"[WARN] Only {len(records):,} examples were available for {lang_name}.",
            flush=True,
        )

    return len(records)


# ============================================================
# FLORES valid/test builder
# ============================================================

def load_flores_all_split(flores_split: str):
    print("\n" + "=" * 80, flush=True)
    print(f"Loading FLORES split: {flores_split}", flush=True)
    print("=" * 80, flush=True)
    print(f"Dataset: {FLORES_DATASET}", flush=True)
    print(f"Config:  {FLORES_CONFIG}", flush=True)

    ds = load_dataset(
        FLORES_DATASET,
        FLORES_CONFIG,
        split=flores_split,
        trust_remote_code=True,
    )

    print(f"Loaded FLORES rows: {len(ds):,}", flush=True)

    return ds


def validate_flores_columns(ds, lang_info: dict) -> None:
    source_col = flores_column(FLORES_EN_CODE)
    target_col = flores_column(lang_info["flores_code"])

    columns = set(ds.column_names)

    missing = []

    if source_col not in columns:
        missing.append(source_col)

    if target_col not in columns:
        missing.append(target_col)

    if missing:
        available_sentence_columns = sorted(
            col for col in ds.column_names if col.startswith("sentence_")
        )

        raise ValueError(
            "Missing FLORES columns:\n"
            + "\n".join([f"  - {col}" for col in missing])
            + "\n\nAvailable sentence columns include:\n"
            + "\n".join(available_sentence_columns[:150])
        )


def build_flores_split_from_loaded(
    ds,
    lang_info: dict,
    flores_split: str,
    split_name: str,
) -> int:
    lang_name = lang_info["name"]
    target_flores_code = lang_info["flores_code"]

    source_col = flores_column(FLORES_EN_CODE)
    target_col = flores_column(target_flores_code)

    output_path = os.path.join(OUTPUT_DIR, f"MT_En_{lang_name}_{split_name}.json")

    print("\n" + "=" * 80, flush=True)
    print(
        f"Building FLORES {split_name}: English to {lang_name} [{flores_split}]",
        flush=True,
    )
    print("=" * 80, flush=True)
    print(f"Source column: {source_col}", flush=True)
    print(f"Target column: {target_col}", flush=True)
    print(f"Output:        {output_path}", flush=True)
    print("Filtering:     none", flush=True)

    validate_flores_columns(ds, lang_info)

    records = []

    for item in ds:
        src = item[source_col]
        tgt = item[target_col]

        records.append(make_record(src, tgt, lang_name))

    save_json(records, output_path)

    print(f"Saved {split_name} split to {output_path}", flush=True)
    print(f"Total {split_name} examples saved: {len(records):,}", flush=True)

    return len(records)


# ============================================================
# dataset_info.json
# ============================================================

def build_dataset_info() -> None:
    dataset_info = {}

    for lang_info in LANGUAGES.values():
        lang_name = lang_info["name"]

        dataset_info[f"MT_En_{lang_name}"] = {
            "file_name": f"MT_En_{lang_name}.json"
        }

        dataset_info[f"MT_En_{lang_name}_valid"] = {
            "file_name": f"MT_En_{lang_name}_valid.json"
        }

        dataset_info[f"MT_En_{lang_name}_test"] = {
            "file_name": f"MT_En_{lang_name}_test.json"
        }

    info_path = os.path.join(OUTPUT_DIR, "dataset_info.json")
    save_json(dataset_info, info_path)

    print(f"\ndataset_info.json saved to {info_path}", flush=True)


# ============================================================
# Main
# ============================================================

def main() -> None:
    summary = {}

    flores_dev = load_flores_all_split("dev")
    flores_devtest = load_flores_all_split("devtest")

    for lang_code, lang_info in LANGUAGES.items():
        lang_name = lang_info["name"]

        print("\n" + "#" * 80, flush=True)
        print(f"Processing language: {lang_name} ({lang_code})", flush=True)
        print("#" * 80, flush=True)

        train_count = build_parallel_train(
            lang_code=lang_code,
            lang_info=lang_info,
        )

        valid_count = build_flores_split_from_loaded(
            ds=flores_dev,
            lang_info=lang_info,
            flores_split="dev",
            split_name="valid",
        )

        test_count = build_flores_split_from_loaded(
            ds=flores_devtest,
            lang_info=lang_info,
            flores_split="devtest",
            split_name="test",
        )

        summary_key = f"MT_En_{lang_name}"

        summary[summary_key] = {
            "train": train_count,
            "valid": valid_count,
            "test": test_count,
            "train_source": lang_info["train_source"],
            "train_dataset": lang_info["dataset_name"],
            "train_config": lang_info["dataset_config"],
            "src_key": lang_info["src_key"],
            "target_key": lang_info["target_key"],
            "flores_code": lang_info["flores_code"],
            "filtering": "none",
        }

    build_dataset_info()

    summary_path = os.path.join(OUTPUT_DIR, "build_summary.json")
    save_json(summary, summary_path)

    print(f"\nSummary saved to {summary_path}", flush=True)
    print("\nAll data files and dataset_info.json created.", flush=True)


if __name__ == "__main__":
    main()