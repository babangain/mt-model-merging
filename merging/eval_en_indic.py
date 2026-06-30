#!/usr/bin/env python3
import csv
import gc
import json
import os
import re
import shutil
import subprocess
from pathlib import Path


# ============================================================
# Hardcoded config
# ============================================================

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"

FINETUNED_MODELS = {
    "Hindi": "anonymous/AnonMT_English_Hindi",
    "Bengali": "anonymous/AnonMT_English_Bengali",
    "Tamil": "anonymous/AnonMT_English_Tamil",
    "Telugu": "anonymous/AnonMT_English_Telugu",
}

# Evaluate directly. Do not merge this model.
MULTILINGUAL_MODEL = "anonymous/AnonMT_English_Indic"

LANGUAGES = ["Hindi", "Bengali", "Tamil", "Telugu"]

DATA_DIR = Path("./data")
WORK_DIR = Path("./english_to_indic_merge_eval")

VALID_SPLIT = "valid"
TEST_SPLIT = "test"

TENSOR_PARALLEL_SIZE = 1
GPU_MEMORY_UTILIZATION = 0.60
EVAL_DTYPE = "bfloat16"
MERGE_DTYPE = "bfloat16"
MAX_TOKENS = 512
EVAL_BATCH_SIZE = 256
TRUST_REMOTE_CODE = False

FORCE_RERUN_EVALS = False
FORCE_RERUN_MERGES = False

DELETE_NON_SELECTED_MERGES = True
ENSURE_SELECTED_MERGE_BEFORE_TEST = True

# Original validation grid
TIES_K_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
TIES_SCALINGS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5]
DARE_SPARSITIES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
DARE_SCALINGS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

TASK_ARITH_SCALINGS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6,0.7, 0.8, 0.9, 1.0]

SCE_TOPK_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# ============================================================
# Helpers
# ============================================================

def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def slug(text):
    text = str(text).replace("/", "__")
    text = re.sub(r"[^A-Za-z0-9_.=+\-]+", "_", text)
    return text[:180]


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_text(path, text):
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def run_cmd(cmd):
    print("\n[RUN]", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run(cmd, check=True)


def avg(values):
    return sum(values) / max(1, len(values))


def load_mt_json(lang, split):
    path = DATA_DIR / f"MT_En_{lang}_{split}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def clean_translation(text):
    if "\ufffd" in text:
        text = text.split("\ufffd", 1)[0]

    text = text.split("\n")[0]
    text = text.strip()
    text = text.strip("'\"")
    return text


def cleanup_cuda():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception as exc:
        print(f"[WARN] CUDA cleanup skipped: {exc}", flush=True)


def delete_path(path, reason):
    path = Path(path)
    if path.exists():
        print(f"[DELETE] {reason}: {path}", flush=True)
        shutil.rmtree(path)



def is_complete_scores(path):
    path = Path(path)

    if not path.exists():
        return False

    try:
        metrics = read_json(path)
    except Exception:
        return False

    if "average_bleu" not in metrics or "average_chrf" not in metrics:
        return False

    if "languages" not in metrics:
        return False

    for lang in LANGUAGES:
        if lang not in metrics["languages"]:
            return False
        if "bleu" not in metrics["languages"][lang]:
            return False
        if "chrf" not in metrics["languages"][lang]:
            return False

    return True


def has_weight_files(path):
    path = Path(path)

    if list(path.glob("*.safetensors")):
        return True
    if list(path.glob("pytorch_model*.bin")):
        return True
    if (path / "model.safetensors.index.json").exists():
        return True
    if (path / "pytorch_model.bin.index.json").exists():
        return True

    return False


def find_stacked_merge_tensors(path):
    path = Path(path)
    safetensor_files = sorted(path.glob("*.safetensors"))

    if not safetensor_files:
        return []

    try:
        from safetensors import safe_open
    except Exception:
        return []

    bad_first_dims = {len(FINETUNED_MODELS), len(FINETUNED_MODELS) + 1}
    suspicious = []

    for sf in safetensor_files:
        try:
            with safe_open(sf, framework="pt", device="cpu") as f:
                for key in f.keys():
                    shape = tuple(f.get_tensor(key).shape)
                    if len(shape) >= 2 and shape[0] in bad_first_dims:
                        suspicious.append((str(sf), key, shape))
        except Exception:
            suspicious.append((str(sf), "__READ_ERROR__", ()))
            break

    return suspicious


def has_stacked_merge_tensors(path):
    suspicious = find_stacked_merge_tensors(path)

    if suspicious:
        print("[BROKEN CHECKPOINT] Found stacked merge tensors:", flush=True)
        for sf, key, shape in suspicious[:8]:
            print(f"  {Path(sf).name}: {key}: {shape}", flush=True)
        return True

    return False


def is_complete_hf_checkpoint(path):
    path = Path(path)

    if not path.exists() or not path.is_dir():
        return False

    if not (path / "config.json").exists():
        return False

    if not has_weight_files(path):
        return False

    if has_stacked_merge_tensors(path):
        return False

    return True


def copy_local_tokenizer(src_dir, dst_dir):
    for name in [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "chat_template.jinja",
        "added_tokens.json",
    ]:
        src = src_dir / name
        if src.exists():
            shutil.copy2(src, dst_dir / name)


def download_tokenizer_files(repo_id, dst_dir):
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo_id,
        allow_patterns=[
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
            "chat_template.jinja",
            "added_tokens.json",
        ],
        local_dir=str(dst_dir),
        local_dir_use_symlinks=False,
    )


def patch_tokenizer_if_needed(model_name):
    """
    This preserves each model's tokenizer.

    For anonymous/AnonMT_English_Indic, HF loading works with:
        extra_special_tokens={}

    But vLLM reloads the tokenizer internally without that argument. So this
    function creates a local patched tokenizer folder and gives that path to vLLM.
    """
    from transformers import AutoTokenizer

    patched_dir = WORK_DIR / "patched_tokenizers" / slug(model_name)
    cfg_path = patched_dir / "tokenizer_config.json"

    if cfg_path.exists():
        tokenizer = AutoTokenizer.from_pretrained(
            str(patched_dir),
            trust_remote_code=TRUST_REMOTE_CODE,
            extra_special_tokens={},
        )
        return tokenizer, str(patched_dir)

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=TRUST_REMOTE_CODE,
            extra_special_tokens={},
        )
    except AttributeError as exc:
        if "list" not in str(exc) or "keys" not in str(exc):
            raise
        tokenizer = None

    needs_local_patch = (model_name == MULTILINGUAL_MODEL) or (tokenizer is None)

    if not needs_local_patch:
        return tokenizer, model_name

    if patched_dir.exists():
        shutil.rmtree(patched_dir)

    ensure_dir(patched_dir)

    print(f"[TOKENIZER PATCH] Creating local patched tokenizer: {patched_dir}", flush=True)

    if Path(str(model_name)).is_dir():
        copy_local_tokenizer(Path(str(model_name)), patched_dir)
    else:
        download_tokenizer_files(model_name, patched_dir)

    cfg_path = patched_dir / "tokenizer_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing tokenizer_config.json in patched tokenizer: {cfg_path}")

    cfg = read_json(cfg_path)

    if isinstance(cfg.get("extra_special_tokens"), list):
        cfg["extra_special_tokens"] = {}

    write_json(cfg_path, cfg)

    tokenizer = AutoTokenizer.from_pretrained(
        str(patched_dir),
        trust_remote_code=TRUST_REMOTE_CODE,
        extra_special_tokens={},
    )

    return tokenizer, str(patched_dir)


# ============================================================
# Evaluation
# ============================================================

def evaluate_model(model_name, model_label, split):
    from vllm import LLM, SamplingParams
    from sacrebleu.metrics import BLEU, CHRF

    safe_label = slug(model_label)
    out_dir = WORK_DIR / "model_eval" / safe_label / split
    metrics_path = out_dir / "scores.json"

    if (not FORCE_RERUN_EVALS) and is_complete_scores(metrics_path):
        print(f"[SKIP EVAL] {model_label} {split}: {metrics_path}", flush=True)
        return read_json(metrics_path)

    if FORCE_RERUN_EVALS and out_dir.exists():
        delete_path(out_dir, "force rerun eval")

    ensure_dir(out_dir)

    print("\n" + "=" * 80, flush=True)
    print(f"[EVAL] label: {model_label}", flush=True)
    print(f"[EVAL] model: {model_name}", flush=True)
    print(f"[EVAL] split: {split}", flush=True)
    print("=" * 80, flush=True)

    tokenizer, tokenizer_source = patch_tokenizer_if_needed(model_name)

    prompts = []
    items_by_lang = {}
    ranges = {}

    cursor = 0
    for lang in LANGUAGES:
        items = load_mt_json(lang, split)
        items_by_lang[lang] = items

        start = cursor
        for item in items:
            messages = [
                {
                    "role": "user",
                    "content": item["instruction"] + "\n" + item["input"],
                }
            ]

            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(prompt)
            cursor += 1

        ranges[lang] = (start, cursor)
        print(f"[LOAD] {lang}: {len(items)} examples", flush=True)

    llm = LLM(
        model=model_name,
        tokenizer=tokenizer_source,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        dtype=EVAL_DTYPE,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        trust_remote_code=TRUST_REMOTE_CODE,
        disable_log_stats=True,
    )

    stop = []
    if tokenizer.eos_token:
        stop.append(tokenizer.eos_token)
    stop.extend(["</s>", "\n\n"])

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=MAX_TOKENS,
        stop=stop,
    )

    flat_predictions = []

    for start in range(0, len(prompts), EVAL_BATCH_SIZE):
        end = min(start + EVAL_BATCH_SIZE, len(prompts))
        print(f"[GEN] {start}:{end} / {len(prompts)}", flush=True)

        outputs = llm.generate(prompts[start:end], sampling_params)
        flat_predictions.extend([clean_translation(out.outputs[0].text) for out in outputs])

    bleu_metric = BLEU()
    chrf_metric = CHRF()

    lang_metrics = {}

    for lang in LANGUAGES:
        s, e = ranges[lang]
        preds = flat_predictions[s:e]
        items = items_by_lang[lang]
        refs = [item["output"] for item in items]

        pred_path = out_dir / "predictions" / f"English_to{lang}_{split}.json"
        write_json(pred_path, preds)

        bleu = bleu_metric.corpus_score(preds, [refs]).score
        chrf = chrf_metric.corpus_score(preds, [refs]).score

        lang_metrics[lang] = {
            "bleu": bleu,
            "chrf": chrf,
            "num_examples": len(items),
            "prediction_file": str(pred_path),
        }

        print(f"[SCORE] {lang}: BLEU={bleu:.4f}, CHRF={chrf:.4f}", flush=True)

    metrics = {
        "model_label": model_label,
        "model_name": model_name,
        "tokenizer_source": tokenizer_source,
        "split": split,
        "average_bleu": avg([lang_metrics[lang]["bleu"] for lang in LANGUAGES]),
        "average_chrf": avg([lang_metrics[lang]["chrf"] for lang in LANGUAGES]),
        "languages": lang_metrics,
    }

    write_json(metrics_path, metrics)

    print(
        f"[AVG] BLEU={metrics['average_bleu']:.4f}, "
        f"CHRF={metrics['average_chrf']:.4f}",
        flush=True,
    )
    print(f"[SAVE] {metrics_path}", flush=True)

    del llm
    del tokenizer
    cleanup_cuda()

    return metrics


# ============================================================
# Official MergeKit config generation
# ============================================================

def make_standard_model_entries(weight, density=None):
    lines = []

    for lang in LANGUAGES:
        model = FINETUNED_MODELS[lang]
        lines.append(f"  - model: {model}")
        lines.append("    parameters:")
        lines.append(f"      weight: {weight}")

        if density is not None:
            lines.append(f"      density: {density}")

    return "\n".join(lines)


def make_sce_model_entries():
    lines = []

    # SCE follows the FuseChat-style structure:
    # base/pivot model first, then the finetuned models.
    lines.append(f"  - model: {BASE_MODEL}")

    for lang in LANGUAGES:
        lines.append(f"  - model: {FINETUNED_MODELS[lang]}")

    return "\n".join(lines)


def write_mergekit_config(job):
    config_dir = WORK_DIR / "mergekit_configs"
    ensure_dir(config_dir)

    config_path = config_dir / f"{job['name']}.yaml"
    method = job["merge_method"]

    lines = []

    if method == "sce":
        lines.append("models:")
        lines.append(make_sce_model_entries())
        lines.append("merge_method: sce")
        lines.append(f"base_model: {BASE_MODEL}")
        lines.append("parameters:")
        lines.append(f"  select_topk: {job['select_topk']}")
        lines.append(f"dtype: {MERGE_DTYPE}")

    else:
        weight = job["scaling"]
        density = job.get("density")
        normalize = job.get("normalize")
        int8_mask = job.get("int8_mask")
        rescale = job.get("rescale")

        lines.append(f"merge_method: {method}")
        lines.append(f"base_model: {BASE_MODEL}")
        lines.append("models:")
        lines.append(make_standard_model_entries(weight=weight, density=density))
        lines.append("parameters:")

        if normalize is not None:
            lines.append(f"  normalize: {str(normalize).lower()}")

        if int8_mask is not None:
            lines.append(f"  int8_mask: {str(int8_mask).lower()}")

        if rescale is not None:
            lines.append(f"  rescale: {str(rescale).lower()}")

        lines.append(f"dtype: {MERGE_DTYPE}")
        lines.append("tokenizer:")
        lines.append("  source: base")
        lines.append("chat_template: auto")

    write_text(config_path, "\n".join(lines) + "\n")
    return config_path


def make_merge_jobs():
    jobs = []

    for scaling in TASK_ARITH_SCALINGS:
        jobs.append(
            {
                "method_label": "Task Arithmetic",
                "name": f"task_arithmetic_scale_{scaling}",
                "merge_method": "task_arithmetic",
                "scaling": scaling,
                "density": None,
                "normalize": None,
                "int8_mask": None,
                "rescale": None,
            }
        )

    for k in TIES_K_VALUES:
        for scaling in TIES_SCALINGS:
            jobs.append(
                {
                    "method_label": "TIES",
                    "name": f"ties_keep_{k}_scale_{scaling}",
                    "merge_method": "ties",
                    "scaling": scaling,
                    "density": k,
                    "normalize": True,
                    "int8_mask": False,
                    "rescale": None,
                }
            )

    for sparsity in DARE_SPARSITIES:
        for scaling in DARE_SCALINGS:
            density = round(1.0 - sparsity, 10)

            jobs.append(
                {
                    "method_label": "DARE",
                    "name": f"dare_sparsity_{sparsity}_scale_{scaling}",
                    "merge_method": "dare_linear",
                    "scaling": scaling,
                    "density": density,
                    "paper_sparsity": sparsity,
                    "normalize": True,
                    "int8_mask": False,
                    "rescale": True,
                }
            )

    for topk in SCE_TOPK_VALUES:
        jobs.append(
            {
                "method_label": "SCE-Merging",
                "name": f"sce_topk_{topk}",
                "merge_method": "sce",
                "select_topk": topk,
            }
        )

    return jobs


def merged_checkpoint_path(job):
    return WORK_DIR / "merged_checkpoints" / job["name"]


def ensure_merged_checkpoint(job):
    config_path = write_mergekit_config(job)
    out_dir = merged_checkpoint_path(job)

    if FORCE_RERUN_MERGES and out_dir.exists():
        delete_path(out_dir, "force rerun merge")

    if is_complete_hf_checkpoint(out_dir):
        print(f"[USE MERGE] {job['name']}: {out_dir}", flush=True)
        return out_dir

    if out_dir.exists():
        delete_path(out_dir, "missing or broken merge checkpoint")

    ensure_dir(out_dir.parent)

    cmd = [
        "mergekit-yaml",
        str(config_path),
        str(out_dir),
        "--cuda",
        "--lazy-unpickle",
    ]

    if TRUST_REMOTE_CODE:
        cmd.append("--trust-remote-code")

    run_cmd(cmd)

    if not is_complete_hf_checkpoint(out_dir):
        raise RuntimeError(
            f"Merge finished but checkpoint looks broken: {out_dir}\n"
            f"Config used: {config_path}"
        )

    return out_dir


# ============================================================
# Result writing
# ============================================================

def row_from_metrics(label, model_name, metrics):
    row = {
        "Model": label,
        "model_name": model_name,
        "average_bleu": metrics["average_bleu"],
        "average_chrf": metrics["average_chrf"],
    }

    for lang in LANGUAGES:
        row[f"{lang}_BLEU"] = metrics["languages"][lang]["bleu"]

    for lang in LANGUAGES:
        row[f"{lang}_CHRF"] = metrics["languages"][lang]["chrf"]

    return row


def write_csv(path, rows):
    ensure_dir(path.parent)

    all_keys = []
    seen = set()

    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                all_keys.append(key)

    preferred = (
        ["Model"]
        + [f"{lang}_BLEU" for lang in LANGUAGES]
        + ["average_bleu"]
        + [f"{lang}_CHRF" for lang in LANGUAGES]
        + ["average_chrf", "model_name"]
    )

    fields = []
    for key in preferred:
        if key in seen:
            fields.append(key)

    for key in all_keys:
        if key not in fields:
            fields.append(key)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_scores(rows):
    print("\n" + "=" * 80, flush=True)
    print("FINAL ENGLISH TO INDIC TEST SCORES", flush=True)
    print("=" * 80, flush=True)

    header = (
        ["Model"]
        + [f"{lang} BLEU" for lang in LANGUAGES]
        + ["Avg BLEU"]
        + [f"{lang} CHRF" for lang in LANGUAGES]
        + ["Avg CHRF"]
    )

    print("\t".join(header), flush=True)

    for row in rows:
        vals = [row["Model"]]
        vals.extend([f"{row[f'{lang}_BLEU']:.2f}" for lang in LANGUAGES])
        vals.append(f"{row['average_bleu']:.2f}")
        vals.extend([f"{row[f'{lang}_CHRF']:.2f}" for lang in LANGUAGES])
        vals.append(f"{row['average_chrf']:.2f}")
        print("\t".join(vals), flush=True)


def valid_record_from_metrics(job, merged_path, valid_metrics):
    record = dict(job)
    record["merged_path"] = str(merged_path)
    record["valid_average_bleu"] = valid_metrics["average_bleu"]
    record["valid_average_chrf"] = valid_metrics["average_chrf"]

    for lang in LANGUAGES:
        record[f"valid_{lang}_BLEU"] = valid_metrics["languages"][lang]["bleu"]
        record[f"valid_{lang}_CHRF"] = valid_metrics["languages"][lang]["chrf"]

    return record


def delete_unselected_checkpoints(valid_records, selected_records):
    selected_paths = set(str(record["merged_path"]) for record in selected_records)

    for record in valid_records:
        path = str(record["merged_path"])
        if path not in selected_paths:
            delete_path(path, "not selected after valid eval")


# ============================================================
# Main
# ============================================================

def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    ensure_dir(WORK_DIR)

    print("\nOriginal validation grid:", flush=True)
    print(f"TIES K: {TIES_K_VALUES}", flush=True)
    print(f"TIES scaling: {TIES_SCALINGS}", flush=True)
    print(f"DARE sparsity: {DARE_SPARSITIES}", flush=True)
    print(f"DARE scaling: {DARE_SCALINGS}", flush=True)
    print(f"Task Arithmetic scaling: {TASK_ARITH_SCALINGS}", flush=True)
    print(f"SCE topk: {SCE_TOPK_VALUES}", flush=True)
    print(f"Work dir: {WORK_DIR}", flush=True)

    final_rows = []

    # 1. Evaluate base, specialists, and multilingual model on test.
    base_test = evaluate_model(BASE_MODEL, "Base", TEST_SPLIT)
    final_rows.append(row_from_metrics("Base", BASE_MODEL, base_test))

    for lang in LANGUAGES:
        label = f"Finetuned English to {lang}"
        model_name = FINETUNED_MODELS[lang]
        test_metrics = evaluate_model(model_name, label, TEST_SPLIT)
        final_rows.append(row_from_metrics(label, model_name, test_metrics))

    multilingual_test = evaluate_model(
        MULTILINGUAL_MODEL,
        "Finetuned English to Indic",
        TEST_SPLIT,
    )
    final_rows.append(
        row_from_metrics(
            "Finetuned English to Indic",
            MULTILINGUAL_MODEL,
            multilingual_test,
        )
    )

    # 2. Merge grid, evaluate on valid, keep only best per method.
    merge_jobs = make_merge_jobs()
    valid_records = []
    best_by_method = {}

    for job_id, job in enumerate(merge_jobs, start=1):
        print("\n" + "=" * 80, flush=True)
        print(f"[MERGE GRID] {job_id}/{len(merge_jobs)}: {job['name']}", flush=True)
        print("=" * 80, flush=True)

        merged_path = merged_checkpoint_path(job)
        valid_label = f"valid_{job['name']}"
        valid_scores_path = WORK_DIR / "model_eval" / slug(valid_label) / VALID_SPLIT / "scores.json"

        if (not FORCE_RERUN_EVALS) and is_complete_scores(valid_scores_path):
            print(f"[USE VALID SCORE] {job['name']}: {valid_scores_path}", flush=True)
            valid_metrics = read_json(valid_scores_path)
        else:
            merged_path = ensure_merged_checkpoint(job)
            valid_metrics = evaluate_model(
                model_name=str(merged_path),
                model_label=valid_label,
                split=VALID_SPLIT,
            )

        record = valid_record_from_metrics(job, merged_path, valid_metrics)
        valid_records.append(record)

        method = job["method_label"]
        current_best = best_by_method.get(method)

        is_new_best = (
            current_best is None
            or record["valid_average_bleu"] > current_best["valid_average_bleu"]
        )

        if is_new_best:
            if (
                DELETE_NON_SELECTED_MERGES
                and current_best is not None
                and current_best["merged_path"] != record["merged_path"]
            ):
                delete_path(current_best["merged_path"], f"old best replaced for {method}")

            best_by_method[method] = record
            print(
                f"[BEST SO FAR] {method}: {record['name']} "
                f"valid_avg_bleu={record['valid_average_bleu']:.4f}",
                flush=True,
            )
        else:
            if DELETE_NON_SELECTED_MERGES:
                delete_path(record["merged_path"], f"not best for {method}")

        write_json(WORK_DIR / "valid_merge_records.json", valid_records)
        write_json(WORK_DIR / "selected_merge_checkpoints_so_far.json", best_by_method)

    # 3. Final selected checkpoints.
    method_order = ["Task Arithmetic", "TIES", "DARE", "SCE-Merging"]

    selected_records = []
    for method in method_order:
        if method not in best_by_method:
            raise RuntimeError(f"No selected checkpoint found for method: {method}")
        selected_records.append(best_by_method[method])

    if DELETE_NON_SELECTED_MERGES:
        delete_unselected_checkpoints(valid_records, selected_records)

    write_json(WORK_DIR / "selected_merge_checkpoints.json", selected_records)

    # 4. Evaluate selected merged checkpoints on test.
    for selected in selected_records:
        method = selected["method_label"]

        print("\n" + "=" * 80, flush=True)
        print(
            f"[SELECTED] {method}: {selected['name']} "
            f"valid_avg_bleu={selected['valid_average_bleu']:.4f}",
            flush=True,
        )
        print("=" * 80, flush=True)

        if ENSURE_SELECTED_MERGE_BEFORE_TEST:
            selected_path = ensure_merged_checkpoint(selected)
            selected["merged_path"] = str(selected_path)

        test_label = f"test_selected_{slug(method)}_{selected['name']}"

        test_metrics = evaluate_model(
            model_name=selected["merged_path"],
            model_label=test_label,
            split=TEST_SPLIT,
        )

        final_rows.append(row_from_metrics(method, selected["merged_path"], test_metrics))

    # 5. Save final scores.
    write_json(WORK_DIR / "selected_merge_checkpoints.json", selected_records)
    write_json(WORK_DIR / "final_english_to_indic_test_scores.json", final_rows)
    write_csv(WORK_DIR / "final_english_to_indic_test_scores.csv", final_rows)
    write_csv(WORK_DIR / "valid_merge_records.csv", valid_records)

    print_scores(final_rows)

    print("\nSaved:", flush=True)
    print(f"  {WORK_DIR / 'final_english_to_indic_test_scores.csv'}", flush=True)
    print(f"  {WORK_DIR / 'final_english_to_indic_test_scores.json'}", flush=True)
    print(f"  {WORK_DIR / 'valid_merge_records.csv'}", flush=True)
    print(f"  {WORK_DIR / 'valid_merge_records.json'}", flush=True)
    print(f"  {WORK_DIR / 'selected_merge_checkpoints.json'}", flush=True)


if __name__ == "__main__":
    main()
