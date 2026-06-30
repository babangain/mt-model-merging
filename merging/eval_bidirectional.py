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

LANGUAGES = ["Hindi", "Bengali", "Tamil", "Telugu"]

DATA_DIR = Path("./data")
WORK_DIR = Path("./pairwise_bidirectional_merge_eval")

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
TIES_SCALINGS = [
    0.1, 0.2, 0.3, 0.4, 0.5,
    0.6, 0.7, 0.8, 0.9, 1.0,
    1.1, 1.2, 1.3, 1.4, 1.5,
    1.6, 1.7, 1.8, 1.9, 2.0,
    2.1, 2.2, 2.3, 2.4, 2.5,
]
DARE_SPARSITIES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
DARE_SCALINGS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

TASK_ARITH_SCALINGS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

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


def make_pair_entries(lang):
    return [
        {
            "task_key": f"English_to_{lang}",
            "source": "English",
            "target": lang,
            "data_prefix": f"MT_En_{lang}",
            "model": f"baban/QwenTranslate_English_{lang}",
        },
        {
            "task_key": f"{lang}_to_English",
            "source": lang,
            "target": "English",
            "data_prefix": f"MT_{lang}_En",
            "model": f"baban/QwenTranslate_{lang}_English",
        },
    ]


def task_keys(pair_entries):
    return [entry["task_key"] for entry in pair_entries]


def load_mt_json(task_entry, split):
    path = DATA_DIR / f"{task_entry['data_prefix']}_{split}.json"
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


def is_complete_scores(path, expected_task_keys):
    path = Path(path)

    if not path.exists():
        return False

    try:
        metrics = read_json(path)
    except Exception:
        return False

    if "average_bleu" not in metrics or "average_chrf" not in metrics:
        return False

    if "tasks" not in metrics:
        return False

    for task_key in expected_task_keys:
        if task_key not in metrics["tasks"]:
            return False
        if "bleu" not in metrics["tasks"][task_key]:
            return False
        if "chrf" not in metrics["tasks"][task_key]:
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


def find_stacked_merge_tensors(path, num_merged_models):
    path = Path(path)
    safetensor_files = sorted(path.glob("*.safetensors"))

    if not safetensor_files:
        return []

    try:
        from safetensors import safe_open
    except Exception:
        return []

    bad_first_dims = {num_merged_models, num_merged_models + 1}
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


def has_stacked_merge_tensors(path, num_merged_models):
    suspicious = find_stacked_merge_tensors(path, num_merged_models)

    if suspicious:
        print("[BROKEN CHECKPOINT] Found stacked merge tensors:", flush=True)
        for sf, key, shape in suspicious[:8]:
            print(f"  {Path(sf).name}: {key}: {shape}", flush=True)
        return True

    return False


def is_complete_hf_checkpoint(path, num_merged_models):
    path = Path(path)

    if not path.exists() or not path.is_dir():
        return False

    if not (path / "config.json").exists():
        return False

    if not has_weight_files(path):
        return False

    if has_stacked_merge_tensors(path, num_merged_models):
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


def patch_tokenizer_if_needed(model_name, lang_work_dir):
    """
    Robust tokenizer loader for HF repos and local merged checkpoints.

    If tokenizer_config.json has extra_special_tokens as a list, this creates
    a patched local tokenizer folder and passes that path to vLLM.
    """
    from transformers import AutoTokenizer

    patched_dir = lang_work_dir / "patched_tokenizers" / slug(model_name)
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
        return tokenizer, model_name
    except AttributeError as exc:
        if "list" not in str(exc) or "keys" not in str(exc):
            raise

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

def evaluate_model(model_name, model_label, split, pair_entries, lang_work_dir):
    from vllm import LLM, SamplingParams
    from sacrebleu.metrics import BLEU, CHRF

    expected_task_keys = task_keys(pair_entries)

    safe_label = slug(model_label)
    out_dir = lang_work_dir / "model_eval" / safe_label / split
    metrics_path = out_dir / "scores.json"

    if (not FORCE_RERUN_EVALS) and is_complete_scores(metrics_path, expected_task_keys):
        print(f"[SKIP EVAL] {model_label} {split}: {metrics_path}", flush=True)
        return read_json(metrics_path)

    if FORCE_RERUN_EVALS and out_dir.exists():
        delete_path(out_dir, "force rerun eval")

    ensure_dir(out_dir)

    print("\n" + "=" * 80, flush=True)
    print(f"[EVAL] label: {model_label}", flush=True)
    print(f"[EVAL] model: {model_name}", flush=True)
    print(f"[EVAL] split: {split}", flush=True)
    print(f"[EVAL] tasks: {expected_task_keys}", flush=True)
    print("=" * 80, flush=True)

    tokenizer, tokenizer_source = patch_tokenizer_if_needed(model_name, lang_work_dir)

    prompts = []
    items_by_task = {}
    ranges = {}

    cursor = 0
    for task_entry in pair_entries:
        task_key = task_entry["task_key"]
        items = load_mt_json(task_entry, split)
        items_by_task[task_key] = items

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

        ranges[task_key] = (start, cursor)
        print(f"[LOAD] {task_key}: {len(items)} examples", flush=True)

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

    task_metrics = {}

    for task_entry in pair_entries:
        task_key = task_entry["task_key"]
        s, e = ranges[task_key]
        preds = flat_predictions[s:e]
        items = items_by_task[task_key]
        refs = [item["output"] for item in items]

        pred_path = out_dir / "predictions" / f"{task_key}_{split}.json"
        write_json(pred_path, preds)

        bleu = bleu_metric.corpus_score(preds, [refs]).score
        chrf = chrf_metric.corpus_score(preds, [refs]).score

        task_metrics[task_key] = {
            "bleu": bleu,
            "chrf": chrf,
            "num_examples": len(items),
            "prediction_file": str(pred_path),
            "source": task_entry["source"],
            "target": task_entry["target"],
            "data_prefix": task_entry["data_prefix"],
        }

        print(f"[SCORE] {task_key}: BLEU={bleu:.4f}, CHRF={chrf:.4f}", flush=True)

    metrics = {
        "model_label": model_label,
        "model_name": model_name,
        "tokenizer_source": tokenizer_source,
        "split": split,
        "average_bleu": avg([task_metrics[task_key]["bleu"] for task_key in expected_task_keys]),
        "average_chrf": avg([task_metrics[task_key]["chrf"] for task_key in expected_task_keys]),
        "tasks": task_metrics,
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

def make_standard_model_entries(pair_entries, weight, density=None):
    lines = []

    for entry in pair_entries:
        model = entry["model"]
        lines.append(f"  - model: {model}")
        lines.append("    parameters:")
        lines.append(f"      weight: {weight}")

        if density is not None:
            lines.append(f"      density: {density}")

    return "\n".join(lines)


def make_sce_model_entries(pair_entries):
    lines = []

    # SCE follows the FuseChat-style structure:
    # base/pivot model first, then the two direction-specific finetuned models.
    lines.append(f"  - model: {BASE_MODEL}")

    for entry in pair_entries:
        lines.append(f"  - model: {entry['model']}")

    return "\n".join(lines)


def write_mergekit_config(job, pair_entries, lang_work_dir):
    config_dir = lang_work_dir / "mergekit_configs"
    ensure_dir(config_dir)

    config_path = config_dir / f"{job['name']}.yaml"
    method = job["merge_method"]

    lines = []

    if method == "sce":
        lines.append("models:")
        lines.append(make_sce_model_entries(pair_entries))
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
        lines.append(make_standard_model_entries(pair_entries, weight=weight, density=density))
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


def merged_checkpoint_path(job, lang_work_dir):
    return lang_work_dir / "merged_checkpoints" / job["name"]


def ensure_merged_checkpoint(job, pair_entries, lang_work_dir):
    config_path = write_mergekit_config(job, pair_entries, lang_work_dir)
    out_dir = merged_checkpoint_path(job, lang_work_dir)
    num_merged_models = len(pair_entries)

    if FORCE_RERUN_MERGES and out_dir.exists():
        delete_path(out_dir, "force rerun merge")

    if is_complete_hf_checkpoint(out_dir, num_merged_models):
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

    if not is_complete_hf_checkpoint(out_dir, num_merged_models):
        raise RuntimeError(
            f"Merge finished but checkpoint looks broken: {out_dir}\n"
            f"Config used: {config_path}"
        )

    return out_dir


# ============================================================
# Result writing
# ============================================================

def row_from_metrics(lang, label, model_name, metrics, pair_entries):
    row = {
        "Language": lang,
        "Model": label,
        "model_name": model_name,
        "average_bleu": metrics["average_bleu"],
        "average_chrf": metrics["average_chrf"],
    }

    for task_key in task_keys(pair_entries):
        row[f"{task_key}_BLEU"] = metrics["tasks"][task_key]["bleu"]

    for task_key in task_keys(pair_entries):
        row[f"{task_key}_CHRF"] = metrics["tasks"][task_key]["chrf"]

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

    preferred_start = ["Language", "Model"]
    preferred_end = ["average_bleu", "average_chrf", "model_name"]

    fields = []

    for key in preferred_start:
        if key in seen:
            fields.append(key)

    for key in all_keys:
        if key not in fields and key not in preferred_end and key != "model_name":
            fields.append(key)

    for key in preferred_end:
        if key in seen and key not in fields:
            fields.append(key)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_scores(rows):
    print("\n" + "=" * 80, flush=True)
    print("FINAL PAIRWISE BIDIRECTIONAL MERGE TEST SCORES", flush=True)
    print("=" * 80, flush=True)

    for row in rows:
        lang = row["Language"]
        print(
            f"{lang}\t{row['Model']}\t"
            f"Avg BLEU={row['average_bleu']:.4f}\t"
            f"Avg CHRF={row['average_chrf']:.4f}\t"
            f"{row['model_name']}",
            flush=True,
        )


def valid_record_from_metrics(job, merged_path, valid_metrics, pair_entries):
    record = dict(job)
    record["merged_path"] = str(merged_path)
    record["valid_average_bleu"] = valid_metrics["average_bleu"]
    record["valid_average_chrf"] = valid_metrics["average_chrf"]

    for task_key in task_keys(pair_entries):
        record[f"valid_{task_key}_BLEU"] = valid_metrics["tasks"][task_key]["bleu"]
        record[f"valid_{task_key}_CHRF"] = valid_metrics["tasks"][task_key]["chrf"]

    return record


def delete_unselected_checkpoints(valid_records, selected_records):
    selected_paths = set(str(record["merged_path"]) for record in selected_records)

    for record in valid_records:
        path = str(record["merged_path"])
        if path not in selected_paths:
            delete_path(path, "not selected after valid eval")


# ============================================================
# Per-language runner
# ============================================================

def run_language(lang):
    pair_entries = make_pair_entries(lang)
    expected_task_keys = task_keys(pair_entries)
    lang_work_dir = WORK_DIR / slug(lang)

    ensure_dir(lang_work_dir)

    print("\n" + "#" * 80, flush=True)
    print(f"[LANGUAGE] {lang}", flush=True)
    print("#" * 80, flush=True)

    print(f"Base model: {BASE_MODEL}", flush=True)
    print(f"Work dir: {lang_work_dir}", flush=True)

    print("\nPairwise models to merge:", flush=True)
    for entry in pair_entries:
        print(f"  {entry['task_key']}: {entry['model']}", flush=True)

    print("\nValidation and test tasks:", flush=True)
    for entry in pair_entries:
        print(f"  {entry['task_key']}: {entry['data_prefix']}_valid/test.json", flush=True)

    print("\nOriginal validation grid:", flush=True)
    print(f"TIES K: {TIES_K_VALUES}", flush=True)
    print(f"TIES scaling: {TIES_SCALINGS}", flush=True)
    print(f"DARE sparsity: {DARE_SPARSITIES}", flush=True)
    print(f"DARE scaling: {DARE_SCALINGS}", flush=True)
    print(f"Task Arithmetic scaling: {TASK_ARITH_SCALINGS}", flush=True)
    print(f"SCE topk: {SCE_TOPK_VALUES}", flush=True)

    final_rows = []

    # 1. Merge grid, validate only on the two directions for this language,
    #    and keep the best checkpoint per merge method.
    merge_jobs = make_merge_jobs()
    valid_records = []
    best_by_method = {}

    for job_id, job in enumerate(merge_jobs, start=1):
        print("\n" + "=" * 80, flush=True)
        print(f"[{lang}] [MERGE GRID] {job_id}/{len(merge_jobs)}: {job['name']}", flush=True)
        print("=" * 80, flush=True)

        merged_path = merged_checkpoint_path(job, lang_work_dir)
        valid_label = f"{lang}_valid_{job['name']}"
        valid_scores_path = (
            lang_work_dir
            / "model_eval"
            / slug(valid_label)
            / VALID_SPLIT
            / "scores.json"
        )

        if (not FORCE_RERUN_EVALS) and is_complete_scores(valid_scores_path, expected_task_keys):
            print(f"[USE VALID SCORE] {job['name']}: {valid_scores_path}", flush=True)
            valid_metrics = read_json(valid_scores_path)
        else:
            merged_path = ensure_merged_checkpoint(job, pair_entries, lang_work_dir)
            valid_metrics = evaluate_model(
                model_name=str(merged_path),
                model_label=valid_label,
                split=VALID_SPLIT,
                pair_entries=pair_entries,
                lang_work_dir=lang_work_dir,
            )

        record = valid_record_from_metrics(job, merged_path, valid_metrics, pair_entries)
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
                delete_path(current_best["merged_path"], f"old best replaced for {lang} {method}")

            best_by_method[method] = record
            print(
                f"[BEST SO FAR] {lang} {method}: {record['name']} "
                f"valid_avg_bleu={record['valid_average_bleu']:.4f}",
                flush=True,
            )
        else:
            if DELETE_NON_SELECTED_MERGES:
                delete_path(record["merged_path"], f"not best for {lang} {method}")

        write_json(lang_work_dir / "valid_merge_records.json", valid_records)
        write_json(lang_work_dir / "selected_merge_checkpoints_so_far.json", best_by_method)

    # 2. Final selected checkpoints.
    method_order = ["Task Arithmetic", "TIES", "DARE", "SCE-Merging"]

    selected_records = []
    for method in method_order:
        if method not in best_by_method:
            raise RuntimeError(f"No selected checkpoint found for {lang} method: {method}")
        selected_records.append(best_by_method[method])

    if DELETE_NON_SELECTED_MERGES:
        delete_unselected_checkpoints(valid_records, selected_records)

    write_json(lang_work_dir / "selected_merge_checkpoints.json", selected_records)

    # 3. Test only on the same two directions for this language.
    for selected in selected_records:
        method = selected["method_label"]

        print("\n" + "=" * 80, flush=True)
        print(
            f"[{lang}] [SELECTED] {method}: {selected['name']} "
            f"valid_avg_bleu={selected['valid_average_bleu']:.4f}",
            flush=True,
        )
        print("=" * 80, flush=True)

        if ENSURE_SELECTED_MERGE_BEFORE_TEST:
            selected_path = ensure_merged_checkpoint(selected, pair_entries, lang_work_dir)
            selected["merged_path"] = str(selected_path)

        test_label = f"{lang}_test_selected_{slug(method)}_{selected['name']}"

        test_metrics = evaluate_model(
            model_name=selected["merged_path"],
            model_label=test_label,
            split=TEST_SPLIT,
            pair_entries=pair_entries,
            lang_work_dir=lang_work_dir,
        )

        final_rows.append(
            row_from_metrics(
                lang=lang,
                label=method,
                model_name=selected["merged_path"],
                metrics=test_metrics,
                pair_entries=pair_entries,
            )
        )

    # 4. Save per-language results.
    write_json(lang_work_dir / "selected_merge_checkpoints.json", selected_records)
    write_json(lang_work_dir / "final_pairwise_bidirectional_test_scores.json", final_rows)
    write_csv(lang_work_dir / "final_pairwise_bidirectional_test_scores.csv", final_rows)
    write_csv(lang_work_dir / "valid_merge_records.csv", valid_records)

    print("\nSaved per-language files:", flush=True)
    print(f"  {lang_work_dir / 'final_pairwise_bidirectional_test_scores.csv'}", flush=True)
    print(f"  {lang_work_dir / 'final_pairwise_bidirectional_test_scores.json'}", flush=True)
    print(f"  {lang_work_dir / 'valid_merge_records.csv'}", flush=True)
    print(f"  {lang_work_dir / 'valid_merge_records.json'}", flush=True)
    print(f"  {lang_work_dir / 'selected_merge_checkpoints.json'}", flush=True)

    return final_rows


# ============================================================
# Main
# ============================================================

def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    ensure_dir(WORK_DIR)

    all_final_rows = []

    for lang in LANGUAGES:
        lang_rows = run_language(lang)
        all_final_rows.extend(lang_rows)

        write_json(WORK_DIR / "all_final_pairwise_bidirectional_test_scores.json", all_final_rows)
        write_csv(WORK_DIR / "all_final_pairwise_bidirectional_test_scores.csv", all_final_rows)

    print_scores(all_final_rows)

    print("\nSaved combined files:", flush=True)
    print(f"  {WORK_DIR / 'all_final_pairwise_bidirectional_test_scores.csv'}", flush=True)
    print(f"  {WORK_DIR / 'all_final_pairwise_bidirectional_test_scores.json'}", flush=True)


if __name__ == "__main__":
    main()