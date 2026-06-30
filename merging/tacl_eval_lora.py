#!/usr/bin/env python3
"""
Method-wise LoRA merge selection and evaluation for En -> Indic MT.

FIXED VERSION: handles PEFT nested save layout for non-default adapter names.

Protocol:
  1. Evaluate individual LoRA adapters on valid and test.
  2. For each merge method, one method at a time:
       a. Build/save all merged LoRA adapters for that method.
       b. Immediately evaluate all saved adapters on the valid split.
       c. Select the best config for that method using macro-average valid BLEU.
       d. Immediately evaluate that selected adapter on the test split.
       e. Move to the next merge method.

Methods:
  - linear
  - ties_svd
  - dare_linear_svd
  - cat

Languages:
  - Hindi
  - Bengali
  - Tamil
  - Telugu

Expected data files:
  DATA_DIR/MT_En_Hindi_valid.json
  DATA_DIR/MT_En_Hindi_test.json
  DATA_DIR/MT_En_Bengali_valid.json
  ...

Each file should contain objects with:
  - instruction
  - input
  - output

The script is designed to skip completed work:
  - merged adapter folders with adapter_config.json and adapter_model.* are skipped
  - complete prediction files are skipped
  - complete score files are skipped unless FORCE_RESCORE=1
"""

import csv
import gc
import hashlib
import itertools
import json
import os
import re
import shutil
import sys
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ======================================================================================
# Configuration
# ======================================================================================

BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")

DATA_DIR = Path(os.environ.get("DATA_DIR", "/workspace/data"))
WORK_DIR = Path(os.environ.get("WORK_DIR", "/workspace/eval_lora_merging_en_indic"))

VALID_SPLIT = os.environ.get("VALID_SPLIT", "valid")
TEST_SPLIT = os.environ.get("TEST_SPLIT", "test")

TENSOR_PARALLEL_SIZE = int(os.environ.get("TENSOR_PARALLEL_SIZE", "1"))
GPU_MEMORY_UTILIZATION = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.90"))
EVAL_DTYPE = os.environ.get("EVAL_DTYPE", "bfloat16")
MERGE_DTYPE = os.environ.get("MERGE_DTYPE", "bfloat16")

MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "512"))
EVAL_BATCH_SIZE = int(os.environ.get("EVAL_BATCH_SIZE", "256"))
TRUST_REMOTE_CODE = os.environ.get("TRUST_REMOTE_CODE", "0") == "1"

# This is a floor. The script auto-increases it if an adapter has a higher rank.
MAX_LORA_RANK = int(os.environ.get("MAX_LORA_RANK", "128"))

FORCE_REMERGE = os.environ.get("FORCE_REMERGE", "0") == "1"
FORCE_RERUN_GENERATION = os.environ.get("FORCE_RERUN_GENERATION", "0") == "1"
FORCE_RESCORE = os.environ.get("FORCE_RESCORE", "0") == "1"

ONLY_INDIVIDUALS = os.environ.get("ONLY_INDIVIDUALS", "0") == "1"
SKIP_INDIVIDUALS = os.environ.get("SKIP_INDIVIDUALS", "0") == "1"

ONLY_METHOD = os.environ.get("ONLY_METHOD", "").strip()
SKIP_MERGE_BUILD = os.environ.get("SKIP_MERGE_BUILD", "0") == "1"
ONLY_BUILD_MERGES = os.environ.get("ONLY_BUILD_MERGES", "0") == "1"

# If 0, individual adapters are evaluated only on their own language.
# If 1, each individual adapter is evaluated on all four languages.
EVAL_INDIVIDUAL_ALL_LANGS = os.environ.get("EVAL_INDIVIDUAL_ALL_LANGS", "0") == "1"

LANGUAGES = ["Hindi", "Bengali", "Tamil", "Telugu"]
LANG_CODES = {
    "Hindi": "hi",
    "Bengali": "bn",
    "Tamil": "ta",
    "Telugu": "te",
}

# Override these with env vars if repo names differ.
ADAPTERS = {
    "Hindi": os.environ.get("ADAPTER_HINDI", "baban/QwenTranslate_English_Hindi_LORA"),
    "Bengali": os.environ.get("ADAPTER_BENGALI", "baban/QwenTranslate_English_Bengali_LORA"),
    "Tamil": os.environ.get("ADAPTER_TAMIL", "baban/QwenTranslate_English_Tamil_LORA"),
    "Telugu": os.environ.get("ADAPTER_TELUGU", "baban/QwenTranslate_English_Telugu_LORA"),
}

MERGE_METHOD_ORDER = ["linear", "ties_svd", "dare_linear_svd", "cat"]
if ONLY_METHOD:
    requested = [x.strip() for x in ONLY_METHOD.split(",") if x.strip()]
    unknown = [x for x in requested if x not in MERGE_METHOD_ORDER]
    if unknown:
        raise ValueError(f"Unknown ONLY_METHOD entries: {unknown}. Valid: {MERGE_METHOD_ORDER}")
    MERGE_METHOD_ORDER = requested


# LoRA-specific compact grid.
# Total configs: linear 7 + cat 7 + ties_svd 60 + dare_linear_svd 24 = 98.
MERGE_GRID = {
    "linear": {
        "scale": [0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 1.5],
    },
    "cat": {
        "scale": [0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 1.5],
    },
    "ties_svd": {
        "scale": [0.7, 0.9, 1.0, 1.2, 1.5],
        "density": [0.3, 0.5, 0.7],
        "svd_rank": [64, 128],
        "majority_sign_method": ["frequency", "total"],
    },
    "dare_linear_svd": {
        "scale": [0.7, 0.9, 1.0, 1.2],
        "density": [0.5, 0.7, 0.8],
        "svd_rank": [64, 128],
    },
}


# ======================================================================================
# Data structures
# ======================================================================================

@dataclass
class ModelSpec:
    label: str
    adapter_path: str
    kind: str
    method: str
    params: Dict[str, Any] = field(default_factory=dict)
    language_scope: Optional[str] = None

    @property
    def safe_id(self) -> str:
        return slug(self.label)


# ======================================================================================
# Paths
# ======================================================================================

MERGED_ROOT = WORK_DIR / "merged_adapters"
EVAL_ROOT = WORK_DIR / "eval"
SUMMARY_DIR = WORK_DIR / "summary"
PATCHED_TOKENIZER_ROOT = WORK_DIR / "patched_tokenizers"
MANIFEST_DIR = WORK_DIR / "manifests"


# ======================================================================================
# Basic helpers
# ======================================================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def slug(text: str, max_len: int = 180) -> str:
    text = str(text).replace("/", "__")
    text = re.sub(r"[^A-Za-z0-9_.=+\-]+", "_", text)
    return text[:max_len]


def safe_peft_adapter_name(text: str) -> str:
    """
    PyTorch module names cannot contain dots. PEFT adapter names become ModuleDict keys.
    This function creates a dot-free internal adapter name.
    """
    text = str(text)
    text = text.replace(".", "p")
    text = text.replace("=", "_")
    text = text.replace("-", "m")
    text = re.sub(r"[^A-Za-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        text = "adapter"
    if not text[0].isalpha():
        text = "adapter_" + text
    return text[:120]


def short_hash(text: str, n: int = 10) -> str:
    return hashlib.sha1(str(text).encode("utf-8")).hexdigest()[:n]


def cleanup_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception as exc:
        print(f"[WARN] CUDA cleanup skipped: {exc}", flush=True)


def get_torch_dtype(dtype_name: str):
    import torch

    normalized = str(dtype_name).lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {dtype_name}")


def clean_translation(text: str) -> str:
    text = str(text)

    if "\ufffd" in text:
        text = text.split("\ufffd", 1)[0]

    text = text.split("\n")[0]
    text = text.strip()
    text = text.strip("'\"")
    return text


def print_header(title: str) -> None:
    print("\n" + "=" * 110, flush=True)
    print(title, flush=True)
    print("=" * 110, flush=True)


def print_subheader(title: str) -> None:
    print("\n" + "-" * 110, flush=True)
    print(title, flush=True)
    print("-" * 110, flush=True)


# ======================================================================================
# Data loading
# ======================================================================================

def data_path(lang: str, split: str) -> Path:
    return DATA_DIR / f"MT_En_{lang}_{split}.json"


def load_mt_json(lang: str, split: str) -> List[dict]:
    path = data_path(lang, split)
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}, got {type(data)}")

    return data


# ======================================================================================
# Tokenizer patching, inherited from original single-adapter script
# ======================================================================================

def copy_local_tokenizer(src_dir: Path, dst_dir: Path) -> None:
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


def download_tokenizer_files(repo_id: str, dst_dir: Path) -> None:
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


def patch_tokenizer_if_needed(model_name: str):
    from transformers import AutoTokenizer

    patched_dir = PATCHED_TOKENIZER_ROOT / slug(model_name)
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

    needs_patch = tokenizer is None

    if tokenizer is not None:
        try:
            if isinstance(tokenizer.init_kwargs.get("extra_special_tokens"), list):
                needs_patch = True
        except Exception:
            pass

    if not needs_patch:
        return tokenizer, model_name

    if patched_dir.exists():
        shutil.rmtree(patched_dir)

    ensure_dir(patched_dir)

    print(f"[TOKENIZER PATCH] Creating patched tokenizer: {patched_dir}", flush=True)

    if Path(str(model_name)).is_dir():
        copy_local_tokenizer(Path(str(model_name)), patched_dir)
    else:
        download_tokenizer_files(model_name, patched_dir)

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


# ======================================================================================
# Prompt construction
# ======================================================================================

def build_prompts_for_lang(tokenizer, lang: str, split: str) -> Tuple[List[str], List[dict]]:
    items = load_mt_json(lang, split)
    prompts = []

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

    return prompts, items


# ======================================================================================
# Adapter metadata and completeness checks
# ======================================================================================

ADAPTER_WEIGHT_FILENAMES = (
    "adapter_model.safetensors",
    "adapter_model.bin",
    "pytorch_model.bin",
)


def adapter_dir_has_config_and_weights(adapter_dir: Path) -> bool:
    if not adapter_dir.exists() or not adapter_dir.is_dir():
        return False
    if not (adapter_dir / "adapter_config.json").exists():
        return False
    return any((adapter_dir / name).exists() for name in ADAPTER_WEIGHT_FILENAMES)


def find_complete_adapter_dir(adapter_dir: Path) -> Optional[Path]:
    """
    PEFT may save a non-default selected adapter under:
        save_root/<adapter_name>/adapter_config.json
    instead of directly under save_root.

    The human-readable save_root is kept for organization, but vLLM must receive
    the directory that actually contains adapter_config.json and adapter_model.*.
    """
    adapter_dir = Path(adapter_dir)

    if adapter_dir_has_config_and_weights(adapter_dir):
        return adapter_dir

    if not adapter_dir.exists() or not adapter_dir.is_dir():
        return None

    # Prefer immediate children. This is the normal PEFT layout for non-default adapters.
    for child in sorted(adapter_dir.iterdir()):
        if child.is_dir() and adapter_dir_has_config_and_weights(child):
            return child

    # Fallback: search a little deeper, but avoid expensive unlimited traversal.
    for cfg_path in sorted(adapter_dir.glob("*/*/adapter_config.json")):
        candidate = cfg_path.parent
        if adapter_dir_has_config_and_weights(candidate):
            return candidate

    return None


def adapter_complete(adapter_dir: Path) -> bool:
    return find_complete_adapter_dir(adapter_dir) is not None


def resolve_adapter_path_for_loading(adapter_path: str) -> str:
    """Return the actual local PEFT adapter directory for vLLM/metadata reads.

    For HF repo ids, the path usually does not exist locally, so return it unchanged.
    For saved merged adapters, return the nested PEFT directory if PEFT saved one.
    """
    path = Path(str(adapter_path))
    if path.exists():
        resolved = find_complete_adapter_dir(path)
        if resolved is None:
            raise FileNotFoundError(
                f"No complete PEFT adapter found under {path}. Expected adapter_config.json "
                f"and one of {ADAPTER_WEIGHT_FILENAMES}."
            )
        return str(resolved)
    return str(adapter_path)


def read_adapter_rank(adapter_path: str) -> int:
    """
    Best-effort rank detection from adapter_config.json.
    Works for local saved adapters. For HF repos, returns MAX_LORA_RANK.
    """
    try:
        path = Path(resolve_adapter_path_for_loading(adapter_path))
    except Exception:
        return MAX_LORA_RANK

    cfg_path = path / "adapter_config.json"

    if not cfg_path.exists():
        return MAX_LORA_RANK

    try:
        cfg = read_json(cfg_path)
    except Exception:
        return MAX_LORA_RANK

    r = cfg.get("r", None)
    if isinstance(r, int):
        return int(r)
    if isinstance(r, dict) and r:
        vals = []
        for value in r.values():
            try:
                vals.append(int(value))
            except Exception:
                pass
        if vals:
            return max(vals)

    # PEFT can also store rank_pattern.
    rp = cfg.get("rank_pattern", None)
    if isinstance(rp, dict) and rp:
        vals = []
        for value in rp.values():
            try:
                vals.append(int(value))
            except Exception:
                pass
        if vals:
            return max(vals)

    return MAX_LORA_RANK


def effective_max_lora_rank(adapter_path: str) -> int:
    return max(MAX_LORA_RANK, read_adapter_rank(adapter_path))


def prediction_path(spec: ModelSpec, split: str, lang: str) -> Path:
    return EVAL_ROOT / spec.safe_id / split / "predictions" / f"English_to_{lang}_{split}.json"


def score_path(spec: ModelSpec, split: str, lang: str) -> Path:
    return EVAL_ROOT / spec.safe_id / split / "scores" / f"English_to_{lang}_{split}.json"


def prediction_complete(path: Path, expected_n: int) -> bool:
    if not path.exists():
        return False
    try:
        preds = read_json(path)
    except Exception:
        return False
    return isinstance(preds, list) and len(preds) == expected_n


def score_complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        metrics = read_json(path)
    except Exception:
        return False

    required = [
        "model_label",
        "adapter_path",
        "kind",
        "method",
        "language",
        "split",
        "bleu",
        "chrf",
        "num_examples",
        "prediction_file",
    ]
    return all(key in metrics for key in required)


# ======================================================================================
# Merge config construction
# ======================================================================================

def frange_label(value: Any) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def make_label(method: str, params: Dict[str, Any]) -> str:
    if method in {"linear", "cat"}:
        return f"{method}__scale={params['scale']}"

    if method == "ties_svd":
        return (
            f"ties_svd__scale={params['scale']}"
            f"__density={params['density']}"
            f"__rank={params['svd_rank']}"
            f"__sign={params['majority_sign_method']}"
        )

    if method == "dare_linear_svd":
        return (
            f"dare_linear_svd__scale={params['scale']}"
            f"__density={params['density']}"
            f"__rank={params['svd_rank']}"
        )

    raise ValueError(f"Unknown method: {method}")


def make_internal_adapter_name(method: str, params: Dict[str, Any]) -> str:
    if method in {"linear", "cat"}:
        raw = f"merge_{method}_s{frange_label(params['scale'])}"
    elif method == "ties_svd":
        raw = (
            f"merge_ties_svd_s{frange_label(params['scale'])}"
            f"_d{frange_label(params['density'])}"
            f"_r{params['svd_rank']}"
            f"_sign_{params['majority_sign_method']}"
        )
    elif method == "dare_linear_svd":
        raw = (
            f"merge_dare_linear_svd_s{frange_label(params['scale'])}"
            f"_d{frange_label(params['density'])}"
            f"_r{params['svd_rank']}"
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    return safe_peft_adapter_name(raw)


def expand_grid(method: str) -> List[Dict[str, Any]]:
    grid = MERGE_GRID[method]
    keys = list(grid.keys())
    values = [grid[key] for key in keys]
    out = []
    for combo in itertools.product(*values):
        out.append(dict(zip(keys, combo)))
    return out


def merged_adapter_dir(label: str) -> Path:
    return MERGED_ROOT / slug(label)


def make_spec_for_method_config(method: str, params: Dict[str, Any]) -> ModelSpec:
    label = make_label(method, params)
    return ModelSpec(
        label=label,
        adapter_path=str(merged_adapter_dir(label)),
        kind="merged",
        method=method,
        params=params,
        language_scope=None,
    )


def make_method_specs(method: str) -> List[ModelSpec]:
    return [make_spec_for_method_config(method, params) for params in expand_grid(method)]


def make_individual_specs() -> List[ModelSpec]:
    specs = []
    for lang in LANGUAGES:
        specs.append(
            ModelSpec(
                label=f"individual__{lang}",
                adapter_path=ADAPTERS[lang],
                kind="individual",
                method="individual",
                params={},
                language_scope=None if EVAL_INDIVIDUAL_ALL_LANGS else lang,
            )
        )
    return specs


# ======================================================================================
# Merge building
# ======================================================================================

def load_peft_model_for_merging():
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    import torch

    dtype = get_torch_dtype(MERGE_DTYPE)

    print("[MERGE MODEL LOAD]", flush=True)
    print(f"  base_model={BASE_MODEL}", flush=True)
    print(f"  merge_dtype={MERGE_DTYPE}", flush=True)

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=TRUST_REMOTE_CODE,
    )

    first_lang = LANGUAGES[0]
    first_code = LANG_CODES[first_lang]
    first_adapter = ADAPTERS[first_lang]

    print(f"  load adapter {first_code}: {first_adapter}", flush=True)
    model = PeftModel.from_pretrained(
        base,
        first_adapter,
        adapter_name=first_code,
        is_trainable=False,
    )

    for lang in LANGUAGES[1:]:
        code = LANG_CODES[lang]
        adapter = ADAPTERS[lang]
        print(f"  load adapter {code}: {adapter}", flush=True)
        model.load_adapter(
            adapter,
            adapter_name=code,
            is_trainable=False,
        )

    model.eval()
    return model


def safe_delete_adapter(model, adapter_name: str, fallback_adapter: str = "hi") -> None:
    try:
        model.set_adapter(fallback_adapter)
    except Exception as exc:
        print(
            f"[WARN] Could not switch to fallback adapter {fallback_adapter} "
            f"before deleting temporary adapter {adapter_name}: {exc}",
            flush=True,
        )

    try:
        model.delete_adapter(adapter_name)
        print(f"[DELETE TEMP IN MEMORY ONLY] {adapter_name}", flush=True)
    except Exception as exc:
        print(
            f"[WARN] Could not delete temporary adapter {adapter_name}: {exc}",
            flush=True,
        )


def build_or_skip_method_merges(method: str) -> List[ModelSpec]:
    specs = make_method_specs(method)

    manifest_path = MANIFEST_DIR / f"{method}_manifest.json"
    ensure_dir(MANIFEST_DIR)

    print_subheader(f"MERGE BUILD STAGE FOR METHOD: {method}")
    print(f"[METHOD] {method}", flush=True)
    print(f"[CONFIGS] {len(specs)}", flush=True)
    print("[IMPORTANT] This stage saves adapters to disk. Temporary in-memory adapters may be deleted after saving.", flush=True)
    print("[IMPORTANT] Validation for this method starts immediately after this method's merge build finishes.", flush=True)

    if SKIP_MERGE_BUILD:
        print(f"[SKIP MERGE BUILD] SKIP_MERGE_BUILD=1 for method {method}", flush=True)
        return specs

    missing = []
    for spec in specs:
        save_root = Path(spec.adapter_path)
        if FORCE_REMERGE or not adapter_complete(save_root):
            missing.append(spec)

    if not missing:
        print(f"[SKIP MERGE BUILD] All {len(specs)} adapters already complete for method {method}.", flush=True)
        write_json(manifest_path, [asdict(spec) for spec in specs])
        return specs

    print(f"[MERGE BUILD] Need to build {len(missing)} / {len(specs)} adapters for method {method}.", flush=True)

    model = None
    try:
        model = load_peft_model_for_merging()
        source_codes = [LANG_CODES[lang] for lang in LANGUAGES]

        for idx, spec in enumerate(missing, start=1):
            params = spec.params
            label = spec.label
            save_root = Path(spec.adapter_path)
            internal_adapter_name = make_internal_adapter_name(method, params)

            if FORCE_REMERGE and save_root.exists():
                print(f"[FORCE_REMERGE] Removing old saved adapter: {save_root}", flush=True)
                shutil.rmtree(save_root)

            ensure_dir(save_root.parent)

            weights = [float(params["scale"])] * len(source_codes)

            kwargs = {
                "adapters": source_codes,
                "weights": weights,
                "adapter_name": internal_adapter_name,
                "combination_type": method,
            }

            if "density" in params:
                kwargs["density"] = float(params["density"])

            if "svd_rank" in params:
                kwargs["svd_rank"] = int(params["svd_rank"])
                kwargs["svd_full_matrices"] = False

            if "majority_sign_method" in params:
                kwargs["majority_sign_method"] = params["majority_sign_method"]

            print("\n" + "-" * 110, flush=True)
            print(f"[MERGE {idx}/{len(missing)}] {label}", flush=True)
            print(f"internal_adapter_name={internal_adapter_name}", flush=True)
            print(f"kwargs={kwargs}", flush=True)
            print(f"save_root={save_root}", flush=True)

            try:
                model.add_weighted_adapter(**kwargs)
                model.set_adapter(internal_adapter_name)

                model.save_pretrained(
                    str(save_root),
                    selected_adapters=[internal_adapter_name],
                )

                metadata = {
                    "label": label,
                    "internal_adapter_name": internal_adapter_name,
                    "method": method,
                    "params": params,
                    "base_model": BASE_MODEL,
                    "source_adapters": ADAPTERS,
                    "peft_kwargs": kwargs,
                }
                write_json(save_root / "merge_metadata.json", metadata)

                resolved_saved = find_complete_adapter_dir(save_root)
                print(f"[SAVE MERGED] {save_root}", flush=True)
                print(f"[SAVED PEFT DIR] {resolved_saved if resolved_saved is not None else 'NOT FOUND'}", flush=True)

            except Exception as exc:
                print(f"[MERGE ERROR] {label}: {exc}", flush=True)
                traceback.print_exc()
            finally:
                # Only delete the temporary adapter inside the currently loaded PEFT model.
                # The saved adapter folder remains available for vLLM validation/test loading.
                safe_delete_adapter(model, internal_adapter_name, fallback_adapter=source_codes[0])

    finally:
        if model is not None:
            del model
        cleanup_cuda()

    write_json(manifest_path, [asdict(spec) for spec in specs])

    print_subheader(f"MERGE BUILD STAGE COMPLETE FOR METHOD: {method}")
    print(f"[NEXT] Starting validation sweep for method {method}.", flush=True)

    return specs


# ======================================================================================
# Generation and scoring
# ======================================================================================

def generation_languages_for_spec(spec: ModelSpec) -> List[str]:
    if spec.language_scope is None:
        return list(LANGUAGES)
    return [spec.language_scope]


def get_or_create_tokenizer():
    tokenizer, patched_tokenizer_path = patch_tokenizer_if_needed(BASE_MODEL)
    return tokenizer, patched_tokenizer_path


def predictions_needed(spec: ModelSpec, split: str, langs: List[str]) -> bool:
    for lang in langs:
        items = load_mt_json(lang, split)
        pred_path = prediction_path(spec, split, lang)
        if FORCE_RERUN_GENERATION or not prediction_complete(pred_path, len(items)):
            return True
    return False


def init_vllm_for_spec(spec: ModelSpec, tokenizer_path: str):
    from vllm import LLM

    resolved_adapter_path = resolve_adapter_path_for_loading(spec.adapter_path)
    rank = effective_max_lora_rank(resolved_adapter_path)

    print("[VLLM INIT]", flush=True)
    print(f"  spec={spec.label}", flush=True)
    print(f"  adapter_path={spec.adapter_path}", flush=True)
    print(f"  resolved_adapter_path={resolved_adapter_path}", flush=True)
    print(f"  base_model={BASE_MODEL}", flush=True)
    print(f"  tokenizer={tokenizer_path}", flush=True)
    print(f"  max_lora_rank={rank}", flush=True)
    print(f"  tensor_parallel_size={TENSOR_PARALLEL_SIZE}", flush=True)
    print(f"  gpu_memory_utilization={GPU_MEMORY_UTILIZATION}", flush=True)
    print(f"  dtype={EVAL_DTYPE}", flush=True)

    llm = LLM(
        model=BASE_MODEL,
        tokenizer=tokenizer_path,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        dtype=EVAL_DTYPE,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        trust_remote_code=TRUST_REMOTE_CODE,
        disable_log_stats=True,
        enable_lora=True,
        max_lora_rank=rank,
    )
    return llm


def generate_missing_predictions_for_spec(spec: ModelSpec, split: str, langs: List[str]) -> None:
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest

    tokenizer, patched_tokenizer_path = get_or_create_tokenizer()

    need_generation = predictions_needed(spec, split, langs)

    if not need_generation:
        print(f"[SKIP GENERATION] Complete predictions already exist for {spec.label} split={split}", flush=True)
        del tokenizer
        cleanup_cuda()
        return

    stop = []
    if tokenizer.eos_token:
        stop.append(tokenizer.eos_token)
    stop.extend(["</s>", "\n\n"])

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=MAX_TOKENS,
        stop=stop,
    )

    resolved_adapter_path = resolve_adapter_path_for_loading(spec.adapter_path)
    lora_name = safe_peft_adapter_name(f"vllm_{spec.safe_id}_{short_hash(resolved_adapter_path)}")
    lora_request = LoRARequest(
        lora_name=lora_name,
        lora_int_id=1,
        lora_path=resolved_adapter_path,
    )

    llm = None
    try:
        llm = init_vllm_for_spec(spec, patched_tokenizer_path)

        for lang in langs:
            items = load_mt_json(lang, split)
            pred_path = prediction_path(spec, split, lang)

            if not FORCE_RERUN_GENERATION and prediction_complete(pred_path, len(items)):
                print(f"[SKIP GENERATION] {spec.label} | {split} | {lang}: {pred_path}", flush=True)
                continue

            prompts, items = build_prompts_for_lang(tokenizer, lang, split)
            preds = []

            print(f"[LOAD] {spec.label} | English to {lang} | {split}: {len(items)} examples", flush=True)

            for start in range(0, len(prompts), EVAL_BATCH_SIZE):
                end = min(start + EVAL_BATCH_SIZE, len(prompts))
                print(
                    f"[GEN] {spec.label} | {split} | English to {lang}: {start}:{end} / {len(prompts)}",
                    flush=True,
                )

                outputs = llm.generate(
                    prompts[start:end],
                    sampling_params,
                    lora_request=lora_request,
                )

                preds.extend([clean_translation(out.outputs[0].text) for out in outputs])

            write_json(pred_path, preds)
            print(f"[SAVE PRED] {pred_path}", flush=True)

    finally:
        if llm is not None:
            del llm
        del tokenizer
        cleanup_cuda()


def score_predictions_for_spec_lang(spec: ModelSpec, split: str, lang: str) -> Dict[str, Any]:
    from sacrebleu.metrics import BLEU, CHRF

    items = load_mt_json(lang, split)
    refs = [item["output"] for item in items]

    pred_path = prediction_path(spec, split, lang)
    out_score_path = score_path(spec, split, lang)

    if not prediction_complete(pred_path, len(items)):
        raise RuntimeError(f"Prediction file missing or incomplete: {pred_path}")

    if not FORCE_RESCORE and score_complete(out_score_path):
        metrics = read_json(out_score_path)
        print(
            f"[SKIP SCORE] {spec.label} | {split} | {lang}: "
            f"BLEU={metrics['bleu']:.4f}, CHRF={metrics['chrf']:.4f}",
            flush=True,
        )
        return metrics

    preds = read_json(pred_path)
    preds = [clean_translation(x) for x in preds]

    bleu_metric = BLEU()
    chrf_metric = CHRF()

    bleu = bleu_metric.corpus_score(preds, [refs]).score
    chrf = chrf_metric.corpus_score(preds, [refs]).score

    metrics = {
        "model_label": spec.label,
        "adapter_path": spec.adapter_path,
        "kind": spec.kind,
        "method": spec.method,
        "params": spec.params,
        "language_scope": spec.language_scope,
        "base_model": BASE_MODEL,
        "language": lang,
        "split": split,
        "bleu": float(bleu),
        "chrf": float(chrf),
        "num_examples": len(items),
        "prediction_file": str(pred_path),
        "score_file": str(out_score_path),
    }

    write_json(out_score_path, metrics)

    print(
        f"[SCORE] {spec.label} | {split} | English to {lang}: "
        f"BLEU={bleu:.4f}, CHRF={chrf:.4f}",
        flush=True,
    )
    print(f"[SAVE SCORE] {out_score_path}", flush=True)

    return metrics


def evaluate_spec_on_split(spec: ModelSpec, split: str) -> List[Dict[str, Any]]:
    langs = generation_languages_for_spec(spec)

    print_subheader(f"EVALUATE SPEC ON {split.upper()}: {spec.label}")
    print(f"adapter_path={spec.adapter_path}", flush=True)
    if Path(str(spec.adapter_path)).exists():
        try:
            print(f"resolved_adapter_path={resolve_adapter_path_for_loading(spec.adapter_path)}", flush=True)
        except Exception as exc:
            print(f"resolved_adapter_path=<ERROR: {exc}>", flush=True)
    print(f"kind={spec.kind}", flush=True)
    print(f"method={spec.method}", flush=True)
    print(f"languages={langs}", flush=True)

    if spec.kind == "merged" and not adapter_complete(Path(spec.adapter_path)):
        raise FileNotFoundError(
            f"Merged adapter is missing/incomplete before evaluation: {spec.adapter_path}"
        )

    generate_missing_predictions_for_spec(spec, split, langs)

    rows = []
    for lang in langs:
        rows.append(score_predictions_for_spec_lang(spec, split, lang))
    return rows


def evaluate_specs_on_split(specs: List[ModelSpec], split: str, summary_name: str) -> List[Dict[str, Any]]:
    all_rows = []
    for idx, spec in enumerate(specs, start=1):
        print_header(f"{summary_name}: {idx}/{len(specs)} | split={split} | {spec.label}")
        rows = evaluate_spec_on_split(spec, split)
        all_rows.extend(rows)
        append_global_score_rows(rows)
    return all_rows


# ======================================================================================
# Aggregation and selection
# ======================================================================================

def macro_by_label(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}

    for row in rows:
        label = row["model_label"]
        grouped.setdefault(label, []).append(row)

    out = []
    for label, group in grouped.items():
        langs = sorted({row["language"] for row in group})
        bleu_vals = [float(row["bleu"]) for row in group]
        chrf_vals = [float(row["chrf"]) for row in group]

        first = group[0]
        out.append(
            {
                "model_label": label,
                "adapter_path": first["adapter_path"],
                "kind": first["kind"],
                "method": first["method"],
                "params": json.dumps(first.get("params", {}), ensure_ascii=False, sort_keys=True),
                "split": first["split"],
                "languages": ",".join(langs),
                "num_languages": len(langs),
                "macro_bleu": sum(bleu_vals) / len(bleu_vals),
                "macro_chrf": sum(chrf_vals) / len(chrf_vals),
            }
        )

    out.sort(key=lambda x: (x["method"], -x["macro_bleu"], x["model_label"]))
    return out


def select_best_by_macro_valid_bleu(method: str, valid_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    macros = macro_by_label(valid_rows)
    complete_macros = [row for row in macros if row["num_languages"] == len(LANGUAGES)]

    if not complete_macros:
        raise RuntimeError(
            f"No complete validation macro rows for method {method}. "
            f"Expected all {len(LANGUAGES)} languages."
        )

    complete_macros.sort(key=lambda x: (-float(x["macro_bleu"]), -float(x["macro_chrf"]), x["model_label"]))
    best = complete_macros[0]

    print_subheader(f"SELECT BEST FOR METHOD {method} BY VALID MACRO BLEU")
    for rank, row in enumerate(complete_macros[:10], start=1):
        print(
            f"{rank:02d}. {row['model_label']} | "
            f"valid macro BLEU={row['macro_bleu']:.4f} | "
            f"valid macro CHRF={row['macro_chrf']:.4f}",
            flush=True,
        )

    print(
        f"[SELECTED] method={method} label={best['model_label']} "
        f"valid_macro_bleu={best['macro_bleu']:.4f} "
        f"valid_macro_chrf={best['macro_chrf']:.4f}",
        flush=True,
    )

    return best


def append_global_score_rows(rows: List[Dict[str, Any]]) -> None:
    """
    Append/update global score logs after every spec, so progress is visible even if the script stops.
    """
    ensure_dir(SUMMARY_DIR)

    global_json = SUMMARY_DIR / "all_scores_incremental.json"
    global_csv = SUMMARY_DIR / "all_scores_incremental.csv"

    existing = []
    if global_json.exists():
        try:
            existing = read_json(global_json)
        except Exception:
            existing = []

    # Deduplicate by model_label, split, language.
    merged = {}
    for row in existing + rows:
        key = (row["model_label"], row["split"], row["language"])
        merged[key] = row

    out = list(merged.values())
    out.sort(key=lambda x: (x["split"], x["method"], x["model_label"], x["language"]))

    write_json(global_json, out)
    write_csv(global_csv, out)


def save_method_outputs(method: str, valid_rows: List[Dict[str, Any]], test_rows: List[Dict[str, Any]], best: Dict[str, Any]) -> None:
    method_summary_dir = SUMMARY_DIR / method
    ensure_dir(method_summary_dir)

    valid_macro = macro_by_label(valid_rows)
    test_macro = macro_by_label(test_rows)

    write_json(method_summary_dir / "valid_scores.json", valid_rows)
    write_csv(method_summary_dir / "valid_scores.csv", valid_rows)
    write_json(method_summary_dir / "valid_macro.json", valid_macro)
    write_csv(method_summary_dir / "valid_macro.csv", valid_macro)

    write_json(method_summary_dir / "selected_by_valid_macro_bleu.json", best)

    write_json(method_summary_dir / "test_selected_scores.json", test_rows)
    write_csv(method_summary_dir / "test_selected_scores.csv", test_rows)
    write_json(method_summary_dir / "test_selected_macro.json", test_macro)
    write_csv(method_summary_dir / "test_selected_macro.csv", test_macro)


def load_spec_from_best(method: str, best: Dict[str, Any]) -> ModelSpec:
    label = best["model_label"]
    adapter_path = best["adapter_path"]

    params = {}
    try:
        params = json.loads(best.get("params", "{}"))
    except Exception:
        params = {}

    return ModelSpec(
        label=label,
        adapter_path=adapter_path,
        kind="merged",
        method=method,
        params=params,
        language_scope=None,
    )


# ======================================================================================
# Method-wise protocol
# ======================================================================================

def run_individuals() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    specs = make_individual_specs()

    print_header("INDIVIDUAL ADAPTERS: VALID THEN TEST")
    print(f"[INDIVIDUAL SPECS] {len(specs)}", flush=True)
    print(f"[EVAL_INDIVIDUAL_ALL_LANGS] {EVAL_INDIVIDUAL_ALL_LANGS}", flush=True)

    valid_rows = evaluate_specs_on_split(
        specs=specs,
        split=VALID_SPLIT,
        summary_name="INDIVIDUAL VALID EVALUATION",
    )

    test_rows = evaluate_specs_on_split(
        specs=specs,
        split=TEST_SPLIT,
        summary_name="INDIVIDUAL TEST EVALUATION",
    )

    individual_dir = SUMMARY_DIR / "individual"
    ensure_dir(individual_dir)

    write_json(individual_dir / "valid_scores.json", valid_rows)
    write_csv(individual_dir / "valid_scores.csv", valid_rows)
    write_json(individual_dir / "valid_macro.json", macro_by_label(valid_rows))
    write_csv(individual_dir / "valid_macro.csv", macro_by_label(valid_rows))

    write_json(individual_dir / "test_scores.json", test_rows)
    write_csv(individual_dir / "test_scores.csv", test_rows)
    write_json(individual_dir / "test_macro.json", macro_by_label(test_rows))
    write_csv(individual_dir / "test_macro.csv", macro_by_label(test_rows))

    return valid_rows, test_rows


def run_one_merge_method(method: str) -> Dict[str, Any]:
    print_header(f"START METHOD-WISE PIPELINE: {method}")

    specs = build_or_skip_method_merges(method)

    if ONLY_BUILD_MERGES:
        print(f"[ONLY_BUILD_MERGES] Stopping after build for method {method}.", flush=True)
        return {
            "method": method,
            "status": "built_only",
            "num_specs": len(specs),
        }

    print_header(f"VALIDATION SWEEP STARTS NOW FOR METHOD: {method}")
    print("[SELECTION RULE] Select best config by macro-average valid BLEU over Hindi, Bengali, Tamil, Telugu.", flush=True)

    valid_rows = evaluate_specs_on_split(
        specs=specs,
        split=VALID_SPLIT,
        summary_name=f"{method} VALID SWEEP",
    )

    best = select_best_by_macro_valid_bleu(method, valid_rows)
    selected_spec = load_spec_from_best(method, best)

    print_header(f"TEST EVALUATION STARTS NOW FOR SELECTED {method}")
    print(f"[SELECTED FOR TEST] {selected_spec.label}", flush=True)
    print(f"[SELECTED ADAPTER PATH] {selected_spec.adapter_path}", flush=True)

    test_rows = evaluate_specs_on_split(
        specs=[selected_spec],
        split=TEST_SPLIT,
        summary_name=f"{method} SELECTED TEST EVALUATION",
    )

    save_method_outputs(method, valid_rows, test_rows, best)

    result = {
        "method": method,
        "status": "done",
        "selected": best,
        "test_macro": macro_by_label(test_rows),
        "num_valid_rows": len(valid_rows),
        "num_test_rows": len(test_rows),
    }

    print_header(f"METHOD COMPLETE: {method}")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)

    return result


# ======================================================================================
# Final summary
# ======================================================================================

def write_final_summary(method_results: List[Dict[str, Any]]) -> None:
    ensure_dir(SUMMARY_DIR)

    write_json(SUMMARY_DIR / "methodwise_results.json", method_results)

    selected_rows = []
    test_macro_rows = []

    for result in method_results:
        if result.get("status") != "done":
            continue

        selected = result.get("selected", {})
        selected_rows.append(
            {
                "method": result["method"],
                "selected_label": selected.get("model_label"),
                "selected_adapter_path": selected.get("adapter_path"),
                "valid_macro_bleu": selected.get("macro_bleu"),
                "valid_macro_chrf": selected.get("macro_chrf"),
                "params": selected.get("params"),
            }
        )

        for row in result.get("test_macro", []):
            test_macro_rows.append(row)

    write_json(SUMMARY_DIR / "selected_best_valid_macro_bleu_by_method.json", selected_rows)
    write_csv(SUMMARY_DIR / "selected_best_valid_macro_bleu_by_method.csv", selected_rows)

    write_json(SUMMARY_DIR / "selected_test_macro_by_method.json", test_macro_rows)
    write_csv(SUMMARY_DIR / "selected_test_macro_by_method.csv", test_macro_rows)

    print_header("FINAL METHOD-WISE SUMMARY")
    if selected_rows:
        print("Selected configs by valid macro BLEU:", flush=True)
        for row in selected_rows:
            print(
                f"  {row['method']}: {row['selected_label']} | "
                f"valid BLEU={float(row['valid_macro_bleu']):.4f} | "
                f"valid CHRF={float(row['valid_macro_chrf']):.4f}",
                flush=True,
            )

    if test_macro_rows:
        print("\nSelected test macro results:", flush=True)
        for row in test_macro_rows:
            print(
                f"  {row['method']}: {row['model_label']} | "
                f"test BLEU={float(row['macro_bleu']):.4f} | "
                f"test CHRF={float(row['macro_chrf']):.4f}",
                flush=True,
            )

    print("\nSaved summaries:", flush=True)
    print(f"  {SUMMARY_DIR / 'methodwise_results.json'}", flush=True)
    print(f"  {SUMMARY_DIR / 'selected_best_valid_macro_bleu_by_method.csv'}", flush=True)
    print(f"  {SUMMARY_DIR / 'selected_test_macro_by_method.csv'}", flush=True)


# ======================================================================================
# Startup checks and main
# ======================================================================================

def print_config() -> None:
    print_header("CONFIG")
    print(f"BASE_MODEL={BASE_MODEL}", flush=True)
    print(f"DATA_DIR={DATA_DIR}", flush=True)
    print(f"WORK_DIR={WORK_DIR}", flush=True)
    print(f"VALID_SPLIT={VALID_SPLIT}", flush=True)
    print(f"TEST_SPLIT={TEST_SPLIT}", flush=True)
    print(f"LANGUAGES={LANGUAGES}", flush=True)
    print(f"ADAPTERS={json.dumps(ADAPTERS, indent=2)}", flush=True)
    print(f"MERGE_METHOD_ORDER={MERGE_METHOD_ORDER}", flush=True)
    print(f"TENSOR_PARALLEL_SIZE={TENSOR_PARALLEL_SIZE}", flush=True)
    print(f"GPU_MEMORY_UTILIZATION={GPU_MEMORY_UTILIZATION}", flush=True)
    print(f"EVAL_DTYPE={EVAL_DTYPE}", flush=True)
    print(f"MERGE_DTYPE={MERGE_DTYPE}", flush=True)
    print(f"MAX_TOKENS={MAX_TOKENS}", flush=True)
    print(f"EVAL_BATCH_SIZE={EVAL_BATCH_SIZE}", flush=True)
    print(f"MAX_LORA_RANK floor={MAX_LORA_RANK}", flush=True)
    print(f"TRUST_REMOTE_CODE={TRUST_REMOTE_CODE}", flush=True)
    print(f"FORCE_REMERGE={FORCE_REMERGE}", flush=True)
    print(f"FORCE_RERUN_GENERATION={FORCE_RERUN_GENERATION}", flush=True)
    print(f"FORCE_RESCORE={FORCE_RESCORE}", flush=True)
    print(f"ONLY_INDIVIDUALS={ONLY_INDIVIDUALS}", flush=True)
    print(f"SKIP_INDIVIDUALS={SKIP_INDIVIDUALS}", flush=True)
    print(f"ONLY_METHOD={ONLY_METHOD}", flush=True)
    print(f"SKIP_MERGE_BUILD={SKIP_MERGE_BUILD}", flush=True)
    print(f"ONLY_BUILD_MERGES={ONLY_BUILD_MERGES}", flush=True)
    print(f"EVAL_INDIVIDUAL_ALL_LANGS={EVAL_INDIVIDUAL_ALL_LANGS}", flush=True)


def validate_data_files() -> None:
    print_subheader("DATA FILE CHECK")
    missing = []
    for split in [VALID_SPLIT, TEST_SPLIT]:
        for lang in LANGUAGES:
            path = data_path(lang, split)
            if not path.exists():
                missing.append(str(path))
                print(f"[MISSING] {path}", flush=True)
            else:
                try:
                    n = len(load_mt_json(lang, split))
                    print(f"[OK] {path} | n={n}", flush=True)
                except Exception as exc:
                    missing.append(str(path))
                    print(f"[BAD] {path} | {exc}", flush=True)

    if missing:
        raise FileNotFoundError("Missing or invalid data files:\n" + "\n".join(missing))


def main() -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    ensure_dir(WORK_DIR)
    ensure_dir(MERGED_ROOT)
    ensure_dir(EVAL_ROOT)
    ensure_dir(SUMMARY_DIR)
    ensure_dir(MANIFEST_DIR)

    print_config()
    validate_data_files()

    method_results = []

    if not SKIP_INDIVIDUALS:
        run_individuals()

    if ONLY_INDIVIDUALS:
        print("[ONLY_INDIVIDUALS] Finished individual adapters. Exiting before merge methods.", flush=True)
        return

    for method in MERGE_METHOD_ORDER:
        result = run_one_merge_method(method)
        method_results.append(result)

    write_final_summary(method_results)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED]", flush=True)
        sys.exit(130)
