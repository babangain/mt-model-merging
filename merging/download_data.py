#!/usr/bin/env python3

from datasets import load_dataset
import argparse
import os
import json


# -------------------------------
# Settings
# -------------------------------
BASE_DIR = "/workspace"

languages = {
    "hi": "Hindi",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
}

flores_codes = {
    "hi": "hin_Deva",
    "bn": "ben_Beng",
    "ta": "tam_Taml",
    "te": "tel_Telu",
}


# -------------------------------
# Helpers
# -------------------------------
def make_prompt(text, lang):
    return f"Translate the following English sentence to {lang}"


def save_json(records, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def update_dataset_info(output_dir, new_entries):
    info_path = os.path.join(output_dir, "dataset_info.json")

    if os.path.exists(info_path):
        with open(info_path, "r", encoding="utf-8") as f:
            try:
                dataset_info = json.load(f)
            except json.JSONDecodeError:
                print(f"⚠️ Existing {info_path} is invalid JSON. Starting fresh.")
                dataset_info = {}
    else:
        dataset_info = {}

    dataset_info.update(new_entries)

    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)

    print(f"📘 dataset_info.json updated at {info_path}")


def create_train_data(output_dir):
    for lang_code, lang_name in languages.items():
        print(f"\n==== Processing train: English -> {lang_name} ({lang_code}) ====")

        print("📥 Loading Samanantar train split...")
        raw = load_dataset("ai4bharat/samanantar", lang_code, split="train")

        print(f"🔢 Total examples: {len(raw)}")

        records = []
        for item in raw:
            src = item["src"]
            tgt = item["tgt"]

            records.append({
                "instruction": make_prompt(src, lang_name),
                "input": src,
                "output": tgt,
            })

        output_path = os.path.join(output_dir, f"MT_En_{lang_name}.json")
        save_json(records, output_path)

        print(f"✅ Saved train split to {output_path}")


def create_valid_test_data(output_dir):
    for lang_code, lang_name in languages.items():
        flores_lang = flores_codes[lang_code]

        for flores_split, split_name in [("dev", "valid"), ("devtest", "test")]:
            print(f"\n==== Processing {split_name}: English -> {lang_name} [{flores_split}] ====")

            ds_src = load_dataset(
                "facebook/flores",
                "eng_Latn",
                split=flores_split,
                trust_remote_code=True,
            )

            ds_tgt = load_dataset(
                "facebook/flores",
                flores_lang,
                split=flores_split,
                trust_remote_code=True,
            )

            print(f"🔢 Total examples in {flores_split}: {len(ds_src)}")

            records = []
            for item_src, item_tgt in zip(ds_src, ds_tgt):
                src = item_src["sentence"]
                tgt = item_tgt["sentence"]

                records.append({
                    "instruction": make_prompt(src, lang_name),
                    "input": src,
                    "output": tgt,
                })

            filename = f"MT_En_{lang_name}_{split_name}.json"
            output_path = os.path.join(output_dir, filename)
            save_json(records, output_path)

            print(f"✅ Saved {split_name} split to {output_path}")


def build_dataset_info(skip_train):
    new_dataset_info = {}

    if not skip_train:
        for lang_code, lang_name in languages.items():
            new_dataset_info[f"MT_En_{lang_name}"] = {
                "file_name": f"MT_En_{lang_name}.json"
            }

    for lang_code, lang_name in languages.items():
        for _, split_name in [("dev", "valid"), ("devtest", "test")]:
            new_dataset_info[f"MT_En_{lang_name}_{split_name}"] = {
                "file_name": f"MT_En_{lang_name}_{split_name}.json"
            }

    return new_dataset_info


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create English to Indic MT datasets from Samanantar and FLORES."
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(BASE_DIR, "data"),
        help="Directory where JSON files and dataset_info.json will be saved.",
    )

    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip Samanantar train dataset creation. Only FLORES valid/test files are created.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.skip_train:
        print("⏭️ Skipping train dataset creation.")
    else:
        create_train_data(args.output_dir)

    create_valid_test_data(args.output_dir)

    new_dataset_info = build_dataset_info(skip_train=args.skip_train)
    update_dataset_info(args.output_dir, new_dataset_info)

    print("\n🎉 English -> Indic datasets processed and dataset_info.json updated.")


if __name__ == "__main__":
    main()