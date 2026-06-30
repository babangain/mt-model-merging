#!/usr/bin/env python3
"""
English to Indic merge + selected-checkpoint layer patch evaluation.

Behavior:
  1. TIES and DARE are removed.
  2. Task Arithmetic and SCE merge grids are kept as configured below.
  3. No embed_tokens or lm_head replacement experiments are run.
  4. MergeKit is used normally. No layers are excluded from MergeKit merging.
  5. Validation evaluates only the default MergeKit output for every merge job.
  6. One best default merged checkpoint is selected per merge method using
     validation average BLEU. If the selection file already exists, it is reused.
  7. On each selected checkpoint, test-time routed layer patches are evaluated
     as two separate experiment families:
       a) replace the target last 5 full transformer layers
       b) replace randomly selected target 1, 2, 3, 4, 5 layers, excluding the last 5 layers
  8. Last-5 and random-layer experiments are kept separate and are not mixed.
  9. Layer patching edits tensors in a copied checkpoint after MergeKit finishes.
 10. No tokenizer files are copied or patched. Evaluation always uses the base
     tokenizer directly.
 11. The English-to-Indic multilingual model is skipped.
"""

import csv
import gc
import json
import os
import re
import shutil
import random
import subprocess
from collections import defaultdict
from functools import lru_cache
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
FORCE_RERUN_PATCHES = False
FORCE_RERUN_SELECTION = False

# Use the base tokenizer for all evaluation. No tokenizer files are copied or patched.
EVAL_TOKENIZER = BASE_MODEL

# Disk control.
DELETE_VALID_MERGE_CHECKPOINTS_AFTER_EVAL = True
ENSURE_SELECTED_MERGE_BEFORE_TEST = True
RUN_TEST_AFTER_SELECTION = True

# Full validation grid used in the current study.
TASK_ARITH_SCALINGS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SCE_TOPK_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Deterministic seed for random layer controls.
# Random patches exclude the final RANDOM_EXCLUDE_LAST_N_LAYERS layers.
RANDOM_LAYER_SEED = 42
RANDOM_EXCLUDE_LAST_N_LAYERS = 5
RANDOM_LAYER_COUNTS = [1, 2, 3, 4, 5]
LAST_LAYER_COUNT = 5

EMBED_LM_KEY_HINTS = [
    "model.embed_tokens.weight",
    "lm_head.weight",
]

SELECTED_DEFAULT_MERGES_JSON = WORK_DIR / "selected_default_merge_checkpoints.json"
VALID_DEFAULT_RECORDS_JSON = WORK_DIR / "valid_default_merge_records.json"
VALID_DEFAULT_RECORDS_CSV = WORK_DIR / "valid_default_merge_records.csv"
DERIVED_LAYER_TEST_RECORDS_JSON = WORK_DIR / "derived_layer_patch_test_records.json"
FINAL_TEST_JSON = WORK_DIR / "final_english_to_indic_layer_patch_test_scores.json"
FINAL_TEST_CSV = WORK_DIR / "final_english_to_indic_layer_patch_test_scores.csv"
GUIDE_PATH = WORK_DIR / "layer_patch_variant_guide.md"


# ============================================================
# Helpers
# ============================================================

def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)


def slug(text):
    text = str(text).replace("/", "__")
    text = re.sub(r"[^A-Za-z0-9_.=+\-]+", "_", text)
    return text[:180]


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_text(path, text):
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def run_cmd(cmd):
    print("\n[RUN]", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run(cmd, check=True)


def avg(values):
    values = list(values)
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


def is_complete_scores(path, expected_languages=None):
    path = Path(path)
    expected_languages = list(expected_languages or LANGUAGES)

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

    for lang in expected_languages:
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

    if not checkpoint_embed_lm_shapes_match_base(path):
        return False

    return True


def load_eval_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        EVAL_TOKENIZER,
        trust_remote_code=TRUST_REMOTE_CODE,
    )


# ============================================================
# Evaluation
# ============================================================

def evaluate_model_for_languages(model_name, model_label, split, eval_languages):
    from vllm import LLM, SamplingParams
    from sacrebleu.metrics import BLEU, CHRF

    eval_languages = list(eval_languages)
    safe_label = slug(model_label)
    out_dir = WORK_DIR / "model_eval" / safe_label / split
    metrics_path = out_dir / "scores.json"

    if (not FORCE_RERUN_EVALS) and is_complete_scores(metrics_path, eval_languages):
        print(f"[SKIP EVAL] {model_label} {split}: {metrics_path}", flush=True)
        return read_json(metrics_path)

    if FORCE_RERUN_EVALS and out_dir.exists():
        delete_path(out_dir, "force rerun eval")

    ensure_dir(out_dir)

    print("\n" + "=" * 80, flush=True)
    print(f"[EVAL] label: {model_label}", flush=True)
    print(f"[EVAL] model: {model_name}", flush=True)
    print(f"[EVAL] split: {split}", flush=True)
    print(f"[EVAL] languages: {eval_languages}", flush=True)
    print("=" * 80, flush=True)

    tokenizer = load_eval_tokenizer()
    tokenizer_source = EVAL_TOKENIZER

    prompts = []
    items_by_lang = {}
    ranges = {}

    cursor = 0
    for lang in eval_languages:
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

    for lang in eval_languages:
        s, e = ranges[lang]
        preds = flat_predictions[s:e]
        items = items_by_lang[lang]
        refs = [item["output"] for item in items]

        pred_path = out_dir / "predictions" / f"English_to_{lang}_{split}.json"
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
        "evaluated_languages": eval_languages,
        "average_bleu": avg([lang_metrics[lang]["bleu"] for lang in eval_languages]),
        "average_chrf": avg([lang_metrics[lang]["chrf"] for lang in eval_languages]),
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


def evaluate_model(model_name, model_label, split):
    return evaluate_model_for_languages(model_name, model_label, split, LANGUAGES)


# ============================================================
# MergeKit config generation
# ============================================================

def make_standard_model_entries(weight):
    lines = []

    for lang in LANGUAGES:
        model = FINETUNED_MODELS[lang]
        lines.append(f"  - model: {model}")
        lines.append("    parameters:")
        lines.append(f"      weight: {weight}")

    return "\n".join(lines)


def make_sce_model_entries():
    lines = []

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

    elif method == "task_arithmetic":
        weight = job["scaling"]
        lines.append("merge_method: task_arithmetic")
        lines.append(f"base_model: {BASE_MODEL}")
        lines.append("models:")
        lines.append(make_standard_model_entries(weight=weight))
        lines.append("parameters:")
        lines.append(f"dtype: {MERGE_DTYPE}")

    else:
        raise ValueError(f"Unsupported merge method after removing TIES/DARE: {method}")

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
# Checkpoint tensor IO for post-merge layer patching
# ============================================================

@lru_cache(maxsize=None)
def resolve_checkpoint_dir(model_name_or_path):
    model_name_or_path = str(model_name_or_path)
    path = Path(model_name_or_path)

    if path.exists() and path.is_dir():
        return str(path)

    from huggingface_hub import snapshot_download

    local_path = snapshot_download(
        repo_id=model_name_or_path,
        allow_patterns=[
            "*.safetensors",
            "*.safetensors.index.json",
            "model.safetensors.index.json",
            "pytorch_model*.bin",
            "pytorch_model.bin.index.json",
            "config.json",
            "generation_config.json",
        ],
    )
    return str(local_path)


def get_safetensor_files(checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    return sorted(checkpoint_dir.glob("*.safetensors"))


def get_safetensor_weight_map(checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    index_path = checkpoint_dir / "model.safetensors.index.json"

    if index_path.exists():
        index = read_json(index_path)
        return {key: checkpoint_dir / rel for key, rel in index.get("weight_map", {}).items()}

    weight_map = {}
    safetensor_files = get_safetensor_files(checkpoint_dir)

    if not safetensor_files:
        return weight_map

    from safetensors import safe_open

    for sf in safetensor_files:
        with safe_open(sf, framework="pt", device="cpu") as f:
            for key in f.keys():
                weight_map[key] = sf

    return weight_map


@lru_cache(maxsize=None)
def list_checkpoint_keys(model_name_or_path):
    checkpoint_dir = Path(resolve_checkpoint_dir(model_name_or_path))
    weight_map = get_safetensor_weight_map(checkpoint_dir)

    if weight_map:
        return tuple(sorted(weight_map.keys()))

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        str(checkpoint_dir),
        trust_remote_code=TRUST_REMOTE_CODE,
        torch_dtype="auto",
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    keys = tuple(sorted(model.state_dict().keys()))
    del model
    cleanup_cuda()
    return keys


def load_tensors_from_checkpoint(model_name_or_path, keys):
    keys = list(dict.fromkeys(keys))
    if not keys:
        return {}

    checkpoint_dir = Path(resolve_checkpoint_dir(model_name_or_path))
    weight_map = get_safetensor_weight_map(checkpoint_dir)

    if weight_map:
        from safetensors import safe_open

        missing = [key for key in keys if key not in weight_map]
        if missing:
            raise KeyError(
                f"Missing tensor keys in checkpoint {model_name_or_path}: {missing[:10]}"
            )

        grouped = defaultdict(list)
        for key in keys:
            grouped[Path(weight_map[key])].append(key)

        tensors = {}
        for sf, sf_keys in grouped.items():
            with safe_open(sf, framework="pt", device="cpu") as f:
                for key in sf_keys:
                    tensors[key] = f.get_tensor(key).cpu()

        return tensors

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        str(checkpoint_dir),
        trust_remote_code=TRUST_REMOTE_CODE,
        torch_dtype="auto",
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    state = model.state_dict()

    tensors = {}
    for key in keys:
        if key not in state:
            raise KeyError(f"Missing tensor key in checkpoint {model_name_or_path}: {key}")
        tensors[key] = state[key].detach().cpu().clone()

    del model
    cleanup_cuda()
    return tensors


def get_checkpoint_tensor_shape(model_name_or_path, key):
    checkpoint_dir = Path(resolve_checkpoint_dir(model_name_or_path))
    weight_map = get_safetensor_weight_map(checkpoint_dir)

    if weight_map:
        from safetensors import safe_open

        if key not in weight_map:
            return None

        with safe_open(weight_map[key], framework="pt", device="cpu") as f:
            return tuple(f.get_tensor(key).shape)

    tensors = load_tensors_from_checkpoint(model_name_or_path, [key])
    return tuple(tensors[key].shape)


def checkpoint_embed_lm_shapes_match_base(checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)

    try:
        checkpoint_keys = set(list_checkpoint_keys(str(checkpoint_dir)))
        base_keys = set(list_checkpoint_keys(BASE_MODEL))
    except Exception as exc:
        print(f"[SHAPE CHECK WARN] Could not list keys for {checkpoint_dir}: {exc}", flush=True)
        return False

    keys = []
    for key in EMBED_LM_KEY_HINTS:
        if key in checkpoint_keys and key in base_keys:
            keys.append(key)

    for key in keys:
        ckpt_shape = get_checkpoint_tensor_shape(str(checkpoint_dir), key)
        base_shape = get_checkpoint_tensor_shape(BASE_MODEL, key)
        if ckpt_shape != base_shape:
            print(
                f"[BAD MERGE SHAPE] {checkpoint_dir}: {key}: "
                f"checkpoint={ckpt_shape}, base={base_shape}. "
                "Deleting/recreating the merge is required.",
                flush=True,
            )
            return False

    return True


def align_replacement_tensor_to_existing_shape(key, old_tensor, new_tensor):
    old_shape = tuple(old_tensor.shape)
    new_shape = tuple(new_tensor.shape)

    if old_shape == new_shape:
        return new_tensor.to(dtype=old_tensor.dtype).cpu()

    raise ValueError(
        f"Shape mismatch for {key}: old={old_shape}, new={new_shape}. "
        "Layer replacement must be shape-exact."
    )


def replace_tensors_in_checkpoint(checkpoint_dir, replacement_tensors):
    checkpoint_dir = Path(checkpoint_dir)
    replacement_tensors = dict(replacement_tensors)

    if not replacement_tensors:
        return

    weight_map = get_safetensor_weight_map(checkpoint_dir)
    if not weight_map:
        raise RuntimeError(
            f"Checkpoint patching expects safetensors output from MergeKit: {checkpoint_dir}"
        )

    missing = [key for key in replacement_tensors if key not in weight_map]
    if missing:
        raise KeyError(f"Cannot patch missing tensors in {checkpoint_dir}: {missing[:10]}")

    from safetensors import safe_open
    from safetensors.torch import save_file

    grouped = defaultdict(list)
    for key in replacement_tensors:
        grouped[Path(weight_map[key])].append(key)

    for sf, keys in grouped.items():
        print(f"[PATCH FILE] {sf.name}: {len(keys)} tensors", flush=True)
        with safe_open(sf, framework="pt", device="cpu") as f:
            metadata = f.metadata()
            tensors = {key: f.get_tensor(key).cpu() for key in f.keys()}

        for key in keys:
            old_tensor = tensors[key]
            new_tensor = replacement_tensors[key].cpu()
            tensors[key] = align_replacement_tensor_to_existing_shape(key, old_tensor, new_tensor)

        tmp_path = sf.with_suffix(sf.suffix + ".tmp")
        save_file(tensors, str(tmp_path), metadata=metadata)
        os.replace(tmp_path, sf)


def get_model_num_layers(model_name_or_path):
    checkpoint_dir = Path(resolve_checkpoint_dir(model_name_or_path))
    cfg = read_json(checkpoint_dir / "config.json")

    if "num_hidden_layers" in cfg:
        return int(cfg["num_hidden_layers"])

    keys = list_checkpoint_keys(model_name_or_path)
    layer_ids = []
    for key in keys:
        match = re.match(r"model\.layers\.(\d+)\.", key)
        if match:
            layer_ids.append(int(match.group(1)))

    if not layer_ids:
        raise RuntimeError(f"Could not infer number of layers for {model_name_or_path}")

    return max(layer_ids) + 1


def get_last_layer_ids(model_name_or_path, count):
    num_layers = get_model_num_layers(model_name_or_path)
    count = int(count)
    if count <= 0:
        return []
    start = max(0, num_layers - count)
    return list(range(start, num_layers))


def get_random_layer_ids_excluding_last(model_name_or_path, count, exclude_last_n=RANDOM_EXCLUDE_LAST_N_LAYERS):
    """
    Deterministically sample count layer ids from all layers except the final exclude_last_n layers.

    The sampled ids are stable across runs for the same model depth, seed, count, and exclude_last_n.
    This is intentionally separate from the last-5 experiment family.
    """
    num_layers = get_model_num_layers(model_name_or_path)
    count = int(count)
    exclude_last_n = int(exclude_last_n)

    upper_exclusive = max(0, num_layers - exclude_last_n)
    candidates = list(range(0, upper_exclusive))

    if count <= 0:
        return []

    if count > len(candidates):
        raise ValueError(
            f"Cannot sample {count} random layers while excluding last {exclude_last_n}: "
            f"only {len(candidates)} eligible layers in model with {num_layers} layers."
        )

    rng_seed = RANDOM_LAYER_SEED + 1009 * num_layers + 101 * exclude_last_n + count
    rng = random.Random(rng_seed)
    return sorted(rng.sample(candidates, count))


def resolve_patch_layer_ids(model_name_or_path, patch_config):
    layer_mode = patch_config.get("layer_mode")

    if layer_mode == "last":
        return get_last_layer_ids(
            model_name_or_path,
            patch_config.get("layer_count", LAST_LAYER_COUNT),
        )

    if layer_mode == "random_excluding_last":
        return get_random_layer_ids_excluding_last(
            model_name_or_path,
            count=patch_config.get("layer_count", 0),
            exclude_last_n=patch_config.get("exclude_last_n", RANDOM_EXCLUDE_LAST_N_LAYERS),
        )

    raise ValueError(f"Unknown layer_mode in patch config: {layer_mode}")


def get_layer_patch_keys(model_name_or_path, layer_ids, include_final_norm=False):
    keys = list(list_checkpoint_keys(model_name_or_path))
    layer_ids = sorted(set(int(idx) for idx in layer_ids))

    prefixes = [f"model.layers.{idx}." for idx in layer_ids]

    selected = []
    for key in keys:
        if any(key.startswith(prefix) for prefix in prefixes):
            selected.append(key)

    if include_final_norm:
        for key in keys:
            if key.startswith("model.norm.") and key not in selected:
                selected.append(key)

    return selected

def copy_checkpoint(src_dir, dst_dir):
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)

    if dst_dir.exists():
        shutil.rmtree(dst_dir)

    ignore = shutil.ignore_patterns(
        "*.tmp",
        "*.lock",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "chat_template.jinja",
        "added_tokens.json",
    )
    shutil.copytree(src_dir, dst_dir, ignore=ignore)


# ============================================================
# Layer patch configuration and checkpoint creation
# ============================================================

def make_layer_test_patch_configs():
    configs = []

    # Experiment family A: final generation tail only.
    # This is a single last-5 experiment, not a sweep over last 1..5.
    configs.append(
        {
            "patch_name": f"target_last_{LAST_LAYER_COUNT}_layers",
            "patch_label": f"Target last {LAST_LAYER_COUNT} layer(s)",
            "patch_family": "last5",
            "routed": True,
            "layer_mode": "last",
            "layer_count": LAST_LAYER_COUNT,
            "exclude_last_n": 0,
            "include_final_norm": True,
        }
    )

    # Experiment family B: random internal layers only.
    # These exclude the final 5 layers and never include model.norm.*.
    for k in RANDOM_LAYER_COUNTS:
        configs.append(
            {
                "patch_name": f"target_random_{k}_layers_excluding_last_{RANDOM_EXCLUDE_LAST_N_LAYERS}_seed_{RANDOM_LAYER_SEED}",
                "patch_label": f"Target random {k} layer(s), excluding last {RANDOM_EXCLUDE_LAST_N_LAYERS}",
                "patch_family": "random_excluding_last5",
                "routed": True,
                "layer_mode": "random_excluding_last",
                "layer_count": k,
                "exclude_last_n": RANDOM_EXCLUDE_LAST_N_LAYERS,
                "include_final_norm": False,
                "random_seed": RANDOM_LAYER_SEED,
            }
        )

    return configs


def variant_guide_text():
    return f"""# Layer patch variant guide

This script removes all embed_tokens and lm_head replacement experiments.

Default merge:
- MergeKit is used normally for all tensors and layers.
- No layers are excluded from MergeKit merging.
- Validation is run only on the default MergeKit output.
- The best checkpoint is selected per method using validation average BLEU.
- If selected_default_merge_checkpoints.json already exists, the selection step is skipped.

Layer patching after selection:
- Patching is applied only after selecting the best default merged checkpoint.
- Patching copies full transformer block tensors from each target-language specialist into a copied selected checkpoint.
- A full layer includes self-attention, MLP, and layer norms under model.layers.<idx>.*.

Experiment family A: last-5 layers:
- patch_name: target_last_{LAST_LAYER_COUNT}_layers
- Replaces exactly the final {LAST_LAYER_COUNT} full transformer blocks from the target specialist.
- Also replaces model.norm.* from the target specialist.
- This family is separate from the random-layer experiments.

Experiment family B: random layers excluding the last 5:
- patch_names: target_random_k_layers_excluding_last_{RANDOM_EXCLUDE_LAST_N_LAYERS}_seed_{RANDOM_LAYER_SEED}, where k in {RANDOM_LAYER_COUNTS}.
- For each k, deterministically samples k layers from the eligible range excluding the final {RANDOM_EXCLUDE_LAST_N_LAYERS} layers.
- Does not replace model.norm.*.
- Does not include any of the last {RANDOM_EXCLUDE_LAST_N_LAYERS} layers.
- This family is separate from the last-5 experiment.

No first-layer special experiment:
- There is no dedicated first-layer patch variant.
- Random sampling may include layer 0 only if the deterministic sample selects it, but there is no first-layer-specific condition.

No embedding/head edits:
- model.embed_tokens.weight is never patched.
- lm_head.weight is never patched.
- Vocabulary size is never changed.
- No padding, truncation, or resizing is performed.

Tokenizer rule:
- No tokenizer files are copied or patched.
- Evaluation always uses the base tokenizer path directly.
"""


def write_variant_guide():
    guide = variant_guide_text()
    write_text(GUIDE_PATH, guide)
    print("\n" + guide, flush=True)


def patch_checkpoint_path(job, patch_config, lang):
    return WORK_DIR / "patched_checkpoints" / job["name"] / patch_config["patch_name"] / lang


def build_layer_replacement_tensors(merged_path, patch_config, lang):
    if lang is None:
        raise ValueError("Layer patch requires lang")

    merged_keys = set(list_checkpoint_keys(str(merged_path)))
    source_model = FINETUNED_MODELS[lang]
    source_keys = set(list_checkpoint_keys(source_model))

    layer_ids = resolve_patch_layer_ids(str(merged_path), patch_config)
    include_final_norm = bool(patch_config.get("include_final_norm", False))
    patch_keys = get_layer_patch_keys(
        str(merged_path),
        layer_ids=layer_ids,
        include_final_norm=include_final_norm,
    )
    patch_keys = [key for key in patch_keys if key in source_keys and key in merged_keys]

    print(
        f"[PATCH] {lang}: family={patch_config.get('patch_family')} "
        f"mode={patch_config.get('layer_mode')} layer_ids={layer_ids} "
        f"include_final_norm={include_final_norm} tensor_count={len(patch_keys)}",
        flush=True,
    )

    return load_tensors_from_checkpoint(source_model, patch_keys)


def patch_done_path(out_dir):
    return Path(out_dir) / "patch_done.json"


def ensure_patched_checkpoint(merged_path, job, patch_config, lang):
    out_dir = patch_checkpoint_path(job, patch_config, lang)
    marker_path = patch_done_path(out_dir)

    if FORCE_RERUN_PATCHES and out_dir.exists():
        delete_path(out_dir, "force rerun patch")

    if is_complete_hf_checkpoint(out_dir) and marker_path.exists():
        print(f"[USE PATCH] {patch_config['patch_name']} {lang}: {out_dir}", flush=True)
        return out_dir

    if out_dir.exists():
        delete_path(out_dir, "missing, broken, or unfinished patched checkpoint")

    print("\n" + "=" * 80, flush=True)
    print(f"[PATCH CHECKPOINT] job={job['name']}", flush=True)
    print(f"[PATCH CHECKPOINT] patch={patch_config['patch_name']}", flush=True)
    print(f"[PATCH CHECKPOINT] lang={lang}", flush=True)
    print(f"[PATCH CHECKPOINT] src merged={merged_path}", flush=True)
    print(f"[PATCH CHECKPOINT] out={out_dir}", flush=True)
    print("=" * 80, flush=True)

    copy_checkpoint(merged_path, out_dir)
    replacement_tensors = build_layer_replacement_tensors(out_dir, patch_config, lang)
    replace_tensors_in_checkpoint(out_dir, replacement_tensors)

    if not is_complete_hf_checkpoint(out_dir):
        raise RuntimeError(f"Patched checkpoint looks broken: {out_dir}")

    write_json(
        marker_path,
        {
            "job_name": job["name"],
            "patch_name": patch_config["patch_name"],
            "lang": lang,
            "merged_path": str(merged_path),
            "patch_family": patch_config.get("patch_family"),
            "layer_mode": patch_config.get("layer_mode"),
            "layer_count": int(patch_config.get("layer_count", 0)),
            "exclude_last_n": int(patch_config.get("exclude_last_n", 0)),
            "include_final_norm": bool(patch_config.get("include_final_norm", False)),
            "resolved_layer_ids": resolve_patch_layer_ids(str(out_dir), patch_config),
        },
    )

    return out_dir


# ============================================================
# Routed layer patch evaluation
# ============================================================

def combined_layer_patch_scores_path(job, patch_config, split):
    label = f"{split}_{job['name']}__{patch_config['patch_name']}"
    return WORK_DIR / "model_eval" / slug(label) / split / "scores.json"


def make_combined_metrics(job, patch_config, split, lang_metrics, model_name):
    return {
        "model_label": f"{job['method_label']} | {job['name']} | {patch_config['patch_label']}",
        "model_name": model_name,
        "split": split,
        "merge_job": job,
        "patch_config": patch_config,
        "evaluated_languages": LANGUAGES,
        "average_bleu": avg([lang_metrics[lang]["bleu"] for lang in LANGUAGES]),
        "average_chrf": avg([lang_metrics[lang]["chrf"] for lang in LANGUAGES]),
        "languages": lang_metrics,
    }


def evaluate_layer_patch_config(merged_path, job, patch_config, split):
    scores_path = combined_layer_patch_scores_path(job, patch_config, split)

    if (not FORCE_RERUN_EVALS) and is_complete_scores(scores_path, LANGUAGES):
        print(f"[USE LAYER PATCH SCORE] {job['name']} {patch_config['patch_name']} {split}: {scores_path}", flush=True)
        return read_json(scores_path)

    if FORCE_RERUN_EVALS and scores_path.parent.exists():
        delete_path(scores_path.parent, "force rerun combined layer patch score")

    ensure_dir(scores_path.parent)

    lang_metrics = {}
    routed_paths = {}

    for lang in LANGUAGES:
        patched_path = ensure_patched_checkpoint(merged_path, job, patch_config, lang=lang)
        routed_paths[lang] = str(patched_path)
        label = f"{split}_{job['name']}__{patch_config['patch_name']}__{lang}"
        metrics = evaluate_model_for_languages(str(patched_path), label, split, [lang])
        lang_metrics[lang] = metrics["languages"][lang]

    combined = make_combined_metrics(
        job=job,
        patch_config=patch_config,
        split=split,
        lang_metrics=lang_metrics,
        model_name="routed:" + json.dumps(routed_paths, ensure_ascii=False),
    )
    write_json(scores_path, combined)
    return combined


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
    path = Path(path)
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


def print_scores(rows, title="FINAL ENGLISH TO INDIC LAYER PATCH TEST SCORES"):
    print("\n" + "=" * 80, flush=True)
    print(title, flush=True)
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


def valid_record_from_metrics(job, valid_metrics):
    record = dict(job)
    record["merged_path"] = str(merged_checkpoint_path(job))
    record["valid_average_bleu"] = valid_metrics["average_bleu"]
    record["valid_average_chrf"] = valid_metrics["average_chrf"]
    record["valid_score_path"] = str(
        WORK_DIR / "model_eval" / slug(f"valid_default_merge__{job['name']}") / VALID_SPLIT / "scores.json"
    )

    for lang in LANGUAGES:
        record[f"valid_{lang}_BLEU"] = valid_metrics["languages"][lang]["bleu"]
        record[f"valid_{lang}_CHRF"] = valid_metrics["languages"][lang]["chrf"]

    return record


# ============================================================
# Selection
# ============================================================

def selection_is_complete(records):
    if not isinstance(records, list):
        return False
    methods = {record.get("method_label") for record in records}
    return "Task Arithmetic" in methods and "SCE-Merging" in methods


def load_existing_selection_if_available():
    if FORCE_RERUN_SELECTION:
        return None

    if not SELECTED_DEFAULT_MERGES_JSON.exists():
        return None

    try:
        records = read_json(SELECTED_DEFAULT_MERGES_JSON)
    except Exception as exc:
        print(f"[SELECTION WARN] Could not read {SELECTED_DEFAULT_MERGES_JSON}: {exc}", flush=True)
        return None

    if not selection_is_complete(records):
        print(f"[SELECTION WARN] Existing selection is incomplete: {SELECTED_DEFAULT_MERGES_JSON}", flush=True)
        return None

    print(f"[USE EXISTING SELECTION] {SELECTED_DEFAULT_MERGES_JSON}", flush=True)
    return records


def select_best_default_merges():
    existing = load_existing_selection_if_available()
    if existing is not None:
        return existing

    merge_jobs = make_merge_jobs()
    valid_records = []
    best_by_method = {}

    for job_id, job in enumerate(merge_jobs, start=1):
        print("\n" + "=" * 80, flush=True)
        print(f"[MERGE GRID] {job_id}/{len(merge_jobs)}: {job['name']}", flush=True)
        print("=" * 80, flush=True)

        valid_label = f"valid_default_merge__{job['name']}"
        valid_scores_path = WORK_DIR / "model_eval" / slug(valid_label) / VALID_SPLIT / "scores.json"

        if (not FORCE_RERUN_EVALS) and is_complete_scores(valid_scores_path, LANGUAGES):
            print(f"[USE VALID SCORE] {job['name']}: {valid_scores_path}", flush=True)
            valid_metrics = read_json(valid_scores_path)
        else:
            merged_path = ensure_merged_checkpoint(job)
            valid_metrics = evaluate_model(str(merged_path), valid_label, VALID_SPLIT)

        record = valid_record_from_metrics(job, valid_metrics)
        valid_records.append(record)

        method = job["method_label"]
        current_best = best_by_method.get(method)
        is_new_best = (
            current_best is None
            or record["valid_average_bleu"] > current_best["valid_average_bleu"]
        )

        if is_new_best:
            best_by_method[method] = record
            print(
                f"[BEST SO FAR] {method}: {record['name']} "
                f"valid_avg_bleu={record['valid_average_bleu']:.4f} "
                f"valid_avg_chrf={record['valid_average_chrf']:.4f}",
                flush=True,
            )

        write_json(VALID_DEFAULT_RECORDS_JSON, valid_records)
        write_json(SELECTED_DEFAULT_MERGES_JSON.with_name("selected_default_merge_checkpoints_so_far.json"), list(best_by_method.values()))

        if DELETE_VALID_MERGE_CHECKPOINTS_AFTER_EVAL:
            delete_path(merged_checkpoint_path(job), "validation merge job evaluated")

    method_order = ["Task Arithmetic", "SCE-Merging"]
    selected_records = []
    for method in method_order:
        if method not in best_by_method:
            raise RuntimeError(f"No selected default checkpoint found for method: {method}")
        selected_records.append(best_by_method[method])

    write_json(SELECTED_DEFAULT_MERGES_JSON, selected_records)
    write_csv(VALID_DEFAULT_RECORDS_CSV, valid_records)

    print("\nSelected default merge records:", flush=True)
    for selected in selected_records:
        print(
            f"  {selected['method_label']}: {selected['name']} "
            f"valid_avg_bleu={selected['valid_average_bleu']:.4f} "
            f"valid_avg_chrf={selected['valid_average_chrf']:.4f}",
            flush=True,
        )

    return selected_records


# ============================================================
# Main
# ============================================================

def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    ensure_dir(WORK_DIR)
    write_variant_guide()

    layer_patch_configs = make_layer_test_patch_configs()

    print("\nValidation grid:", flush=True)
    print(f"Task Arithmetic scaling: {TASK_ARITH_SCALINGS}", flush=True)
    print(f"SCE topk: {SCE_TOPK_VALUES}", flush=True)
    print("Removed methods: TIES, DARE", flush=True)
    print("Skipped model: anonymous/AnonMT_English_Indic", flush=True)
    print(f"Evaluation tokenizer: {EVAL_TOKENIZER}", flush=True)
    print("Validation variant: default MergeKit output only", flush=True)
    print("Test-only layer patch variants:", flush=True)
    for patch_config in layer_patch_configs:
        print(f"  - {patch_config['patch_name']}: {patch_config['patch_label']}", flush=True)
    print(f"Work dir: {WORK_DIR}", flush=True)

    final_rows = []

    if RUN_TEST_AFTER_SELECTION:
        base_test = evaluate_model(BASE_MODEL, "Base", TEST_SPLIT)
        final_rows.append(row_from_metrics("Base", BASE_MODEL, base_test))

        for lang in LANGUAGES:
            label = f"Finetuned English to {lang}"
            model_name = FINETUNED_MODELS[lang]
            test_metrics = evaluate_model(model_name, label, TEST_SPLIT)
            final_rows.append(row_from_metrics(label, model_name, test_metrics))

    selected_records = select_best_default_merges()

    if RUN_TEST_AFTER_SELECTION:
        layer_test_records = []

        for selected in selected_records:
            method = selected["method_label"]

            print("\n" + "=" * 80, flush=True)
            print(
                f"[SELECTED DEFAULT TEST] {method}: {selected['name']} "
                f"valid_avg_bleu={selected['valid_average_bleu']:.4f}",
                flush=True,
            )
            print("=" * 80, flush=True)

            if ENSURE_SELECTED_MERGE_BEFORE_TEST:
                merged_path = ensure_merged_checkpoint(selected)
            else:
                merged_path = merged_checkpoint_path(selected)

            test_label = f"test_selected_default_merge__{selected['name']}"
            test_metrics = evaluate_model(str(merged_path), test_label, TEST_SPLIT)
            final_rows.append(row_from_metrics(f"{method} | Default MergeKit merge", str(merged_path), test_metrics))

            for patch_config in layer_patch_configs:
                print("\n" + "=" * 80, flush=True)
                print(
                    f"[LAYER PATCH TEST] {method}: selected={selected['name']} "
                    f"patch={patch_config['patch_name']} "
                    f"valid_avg_bleu={selected['valid_average_bleu']:.4f}",
                    flush=True,
                )
                print("=" * 80, flush=True)

                patch_metrics = evaluate_layer_patch_config(
                    merged_path=merged_path,
                    job=selected,
                    patch_config=patch_config,
                    split=TEST_SPLIT,
                )

                label = f"{method} | {patch_config['patch_label']} | selected by default merge"
                final_rows.append(row_from_metrics(label, patch_metrics["model_name"], patch_metrics))

                layer_test_records.append(
                    {
                        "method_label": method,
                        "selected_merge_job": selected["name"],
                        "selected_valid_average_bleu": selected["valid_average_bleu"],
                        "selected_valid_average_chrf": selected["valid_average_chrf"],
                        "patch_config": patch_config,
                        "test_average_bleu": patch_metrics["average_bleu"],
                        "test_average_chrf": patch_metrics["average_chrf"],
                        "test_metrics_model_name": patch_metrics["model_name"],
                    }
                )

        write_json(DERIVED_LAYER_TEST_RECORDS_JSON, layer_test_records)
        write_json(FINAL_TEST_JSON, final_rows)
        write_csv(FINAL_TEST_CSV, final_rows)
        print_scores(final_rows)

    print("\nSaved:", flush=True)
    print(f"  {GUIDE_PATH}", flush=True)
    print(f"  {VALID_DEFAULT_RECORDS_CSV}", flush=True)
    print(f"  {VALID_DEFAULT_RECORDS_JSON}", flush=True)
    print(f"  {SELECTED_DEFAULT_MERGES_JSON}", flush=True)

    if RUN_TEST_AFTER_SELECTION:
        print(f"  {DERIVED_LAYER_TEST_RECORDS_JSON}", flush=True)
        print(f"  {FINAL_TEST_CSV}", flush=True)
        print(f"  {FINAL_TEST_JSON}", flush=True)


if __name__ == "__main__":
    main()