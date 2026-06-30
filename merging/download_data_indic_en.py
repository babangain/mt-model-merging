from datasets import load_dataset
import os
import json


# -------------------------------
# Settings
# -------------------------------
BASE_DIR = "/workspace"
output_dir = os.path.join(BASE_DIR, "data")
os.makedirs(output_dir, exist_ok=True)

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
    return f"Translate the following {lang} sentence to English"


def save_json(records, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def update_dataset_info(output_dir, new_entries):
    """
    Reads existing dataset_info.json if present,
    appends/updates the new entries,
    and writes it back.
    """
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


# -------------------------------
# Create training data from Samanantar
# -------------------------------
for lang_code, lang_name in languages.items():
    print(f"\n==== Processing train: {lang_name} -> English ({lang_code}) ====")

    print("📥 Loading Samanantar train split...")
    raw = load_dataset("ai4bharat/samanantar", lang_code, split="train")

    print(f"🔢 Total examples: {len(raw)}")

    records = []
    for item in raw:
        src = item["src"]
        tgt = item["tgt"]

        records.append({
            "instruction": make_prompt(tgt, lang_name),
            "input": tgt,
            "output": src,
        })

    output_path = os.path.join(output_dir, f"MT_{lang_name}_En.json")
    save_json(records, output_path)

    print(f"✅ Saved train split to {output_path}")


# -------------------------------
# Create validation and test data from FLORES
# -------------------------------
for lang_code, lang_name in languages.items():
    flores_lang = flores_codes[lang_code]

    for split, split_name in [("dev", "valid"), ("devtest", "test")]:
        print(f"\n==== Processing {split_name}: {lang_name} -> English [{split}] ====")

        ds_src = load_dataset(
            "facebook/flores",
            "eng_Latn",
            split=split,
            trust_remote_code=True,
        )

        ds_tgt = load_dataset(
            "facebook/flores",
            flores_lang,
            split=split,
            trust_remote_code=True,
        )

        print(f"🔢 Total examples in {split}: {len(ds_tgt)}")

        records = []
        for item_src, item_tgt in zip(ds_src, ds_tgt):
            src = item_src["sentence"]
            tgt = item_tgt["sentence"]

            records.append({
                "instruction": make_prompt(tgt, lang_name),
                "input": tgt,
                "output": src,
            })

        filename = f"MT_{lang_name}_En_{split_name}.json"
        output_path = os.path.join(output_dir, filename)
        save_json(records, output_path)

        print(f"✅ Saved {split_name} split to {output_path}")


# -------------------------------
# Append/update dataset_info.json
# -------------------------------
new_dataset_info = {}

for lang_code, lang_name in languages.items():
    new_dataset_info[f"MT_{lang_name}_En"] = {
        "file_name": f"MT_{lang_name}_En.json"
    }

for lang_code, lang_name in languages.items():
    for split, split_name in [("dev", "valid"), ("devtest", "test")]:
        new_dataset_info[f"MT_{lang_name}_En_{split_name}"] = {
            "file_name": f"MT_{lang_name}_En_{split_name}.json"
        }

update_dataset_info(output_dir, new_dataset_info)

print("\n🎉 Indic -> English datasets processed and dataset_info.json updated.")