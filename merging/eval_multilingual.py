#!/usr/bin/env python3
"""
Evaluate baban/QwenTranslate_Multilingual_Bidirectional on all bidirectional
English <-> Indic language pairs, test split only.

Expected data files under DATA_DIR:
  MT_En_Hindi_test.json
  MT_En_Bengali_test.json
  MT_En_Tamil_test.json
  MT_En_Telugu_test.json
  MT_Hindi_En_test.json
  MT_Bengali_En_test.json
  MT_Tamil_En_test.json
  MT_Telugu_En_test.json

Each JSON file is expected to be a list of objects with:
  instruction, input, output

Outputs:
  bidirectional_multilingual_test_eval/scores.json
  bidirectional_multilingual_test_eval/scores.csv
  bidirectional_multilingual_test_eval/predictions/*.jsonl
"""

import argparse
import csv
import gc
import json
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Any


# ============================================================
# Defaults
# ============================================================

DEFAULT_MODEL = "baban/QwenTranslate_Multilingual_Bidirectional"
DEFAULT_DATA_DIR = "./data"
DEFAULT_WORK_DIR = "./bidirectional_multilingual_test_eval"

LANGUAGES = ["Hindi", "Bengali", "Tamil", "Telugu"]
TEST_SPLIT = "test"

DEFAULT_TENSOR_PARALLEL_SIZE = 1
DEFAULT_GPU_MEMORY_UTILIZATION = 0.60
DEFAULT_DTYPE = "bfloat16"
DEFAULT_MAX_TOKENS = 512
DEFAULT_BATCH_SIZE = 256
DEFAULT_TRUST_REMOTE_CODE = False
DEFAULT_PATCH_TOKENIZER = True


# ============================================================
# File helpers
# ============================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def slug(text: str) -> str:
    text = str(text).replace("/", "__")
    text = re.sub(r"[^A-Za-z0-9_.=+\-]+", "_", text)
    return text[:180]


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    fields = [
        "pair",
        "direction",
        "source_language",
        "target_language",
        "bleu",
        "chrf",
        "num_examples",
        "prediction_file",
        "model_name",
    ]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def avg(values: List[float]) -> float:
    return sum(values) / max(1, len(values))


# ============================================================
# Data and prompt helpers
# ============================================================

def make_eval_pairs() -> List[Dict[str, str]]:
    pairs: List[Dict[str, str]] = []

    for lang in LANGUAGES:
        pairs.append(
            {
                "pair": f"English_to_{lang}",
                "direction": "English_to_Indic",
                "source_language": "English",
                "target_language": lang,
                "data_file": f"MT_En_{lang}_{TEST_SPLIT}.json",
            }
        )

    for lang in LANGUAGES:
        pairs.append(
            {
                "pair": f"{lang}_to_English",
                "direction": "Indic_to_English",
                "source_language": lang,
                "target_language": "English",
                "data_file": f"MT_{lang}_En_{TEST_SPLIT}.json",
            }
        )

    return pairs


def load_mt_json(data_dir: Path, data_file: str) -> List[Dict[str, Any]]:
    path = data_dir / data_file
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")

    data = read_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}, found {type(data).__name__}")

    for idx, item in enumerate(data[:5]):
        for key in ["instruction", "input", "output"]:
            if key not in item:
                raise KeyError(f"Missing key '{key}' in {path}, example index {idx}")

    return data


def clean_translation(text: str) -> str:
    if "\ufffd" in text:
        text = text.split("\ufffd", 1)[0]

    for marker in ["<|im_end|>", "</s>"]:
        if marker in text:
            text = text.split(marker, 1)[0]

    text = text.split("\n")[0]
    text = text.strip()
    text = text.strip("'\"")
    return text


def cleanup_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception as exc:
        print(f"[WARN] CUDA cleanup skipped: {exc}", flush=True)


# ============================================================
# Tokenizer patching
# ============================================================

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
    )


def patch_tokenizer_if_needed(
    model_name: str,
    work_dir: Path,
    trust_remote_code: bool,
    patch_tokenizer: bool,
):
    """
    Some uploaded Qwen tokenizer configs contain extra_special_tokens as a list.
    Transformers can be loaded with extra_special_tokens={}, but vLLM reloads the
    tokenizer internally. A local patched tokenizer folder avoids that failure.
    """
    from transformers import AutoTokenizer

    if not patch_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            extra_special_tokens={},
        )
        return tokenizer, model_name

    patched_dir = work_dir / "patched_tokenizers" / slug(model_name)
    cfg_path = patched_dir / "tokenizer_config.json"

    if not cfg_path.exists():
        if patched_dir.exists():
            shutil.rmtree(patched_dir)
        ensure_dir(patched_dir)

        print(f"[TOKENIZER PATCH] Creating local tokenizer folder: {patched_dir}", flush=True)

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
        trust_remote_code=trust_remote_code,
        extra_special_tokens={},
    )

    return tokenizer, str(patched_dir)


# ============================================================
# Evaluation
# ============================================================

def is_complete_scores(path: Path) -> bool:
    if not path.exists():
        return False

    try:
        scores = read_json(path)
    except Exception:
        return False

    pairs = make_eval_pairs()
    expected = {pair["pair"] for pair in pairs}
    found = set(scores.get("pairs", {}).keys())

    if expected != found:
        return False

    for pair_name in expected:
        pair_scores = scores["pairs"].get(pair_name, {})
        if "bleu" not in pair_scores or "chrf" not in pair_scores:
            return False

    return True


def build_prompts(tokenizer, pairs: List[Dict[str, str]], data_dir: Path):
    prompts: List[str] = []
    items_by_pair: Dict[str, List[Dict[str, Any]]] = {}
    ranges: Dict[str, Tuple[int, int]] = {}

    cursor = 0
    for pair in pairs:
        pair_name = pair["pair"]
        items = load_mt_json(data_dir, pair["data_file"])
        items_by_pair[pair_name] = items

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

        ranges[pair_name] = (start, cursor)
        print(f"[LOAD] {pair_name}: {len(items)} examples from {pair['data_file']}", flush=True)

    return prompts, items_by_pair, ranges


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    from sacrebleu.metrics import BLEU, CHRF
    from vllm import LLM, SamplingParams

    model_name = args.model
    data_dir = Path(args.data_dir)
    work_dir = Path(args.work_dir)
    scores_path = work_dir / "scores.json"
    csv_path = work_dir / "scores.csv"

    ensure_dir(work_dir)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if (not args.force) and is_complete_scores(scores_path):
        print(f"[SKIP] Complete scores already exist: {scores_path}", flush=True)
        scores = read_json(scores_path)
        print_scores(scores)
        return scores

    if args.force and work_dir.exists():
        old_predictions = work_dir / "predictions"
        if old_predictions.exists():
            shutil.rmtree(old_predictions)

    pairs = make_eval_pairs()

    print("\n" + "=" * 80, flush=True)
    print("[EVAL] Bidirectional multilingual test evaluation", flush=True)
    print(f"[MODEL] {model_name}", flush=True)
    print(f"[DATA]  {data_dir}", flush=True)
    print(f"[OUT]   {work_dir}", flush=True)
    print("=" * 80, flush=True)

    tokenizer, tokenizer_source = patch_tokenizer_if_needed(
        model_name=model_name,
        work_dir=work_dir,
        trust_remote_code=args.trust_remote_code,
        patch_tokenizer=(not args.no_tokenizer_patch),
    )

    prompts, items_by_pair, ranges = build_prompts(tokenizer, pairs, data_dir)

    llm = LLM(
        model=model_name,
        tokenizer=tokenizer_source,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=args.trust_remote_code,
        disable_log_stats=True,
    )

    stop = []
    for token in [tokenizer.eos_token, "<|im_end|>", "</s>", "\n\n"]:
        if token and token not in stop:
            stop.append(token)

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        stop=stop,
    )

    flat_predictions: List[str] = []
    for start in range(0, len(prompts), args.batch_size):
        end = min(start + args.batch_size, len(prompts))
        print(f"[GEN] {start}:{end} / {len(prompts)}", flush=True)
        outputs = llm.generate(prompts[start:end], sampling_params)
        flat_predictions.extend([clean_translation(out.outputs[0].text) for out in outputs])

    bleu_metric = BLEU()
    chrf_metric = CHRF()

    pair_scores: Dict[str, Dict[str, Any]] = {}
    csv_rows: List[Dict[str, Any]] = []

    for pair in pairs:
        pair_name = pair["pair"]
        s, e = ranges[pair_name]
        preds = flat_predictions[s:e]
        items = items_by_pair[pair_name]
        refs = [item["output"] for item in items]

        prediction_rows = []
        for idx, (item, pred) in enumerate(zip(items, preds)):
            prediction_rows.append(
                {
                    "index": idx,
                    "pair": pair_name,
                    "source_language": pair["source_language"],
                    "target_language": pair["target_language"],
                    "instruction": item.get("instruction", ""),
                    "input": item.get("input", ""),
                    "reference": item.get("output", ""),
                    "prediction": pred,
                }
            )

        pred_path = work_dir / "predictions" / f"{pair_name}_{TEST_SPLIT}.jsonl"
        write_jsonl(pred_path, prediction_rows)

        bleu = bleu_metric.corpus_score(preds, [refs]).score
        chrf = chrf_metric.corpus_score(preds, [refs]).score

        pair_scores[pair_name] = {
            "pair": pair_name,
            "direction": pair["direction"],
            "source_language": pair["source_language"],
            "target_language": pair["target_language"],
            "bleu": bleu,
            "chrf": chrf,
            "num_examples": len(items),
            "prediction_file": str(pred_path),
            "model_name": model_name,
        }

        csv_rows.append(pair_scores[pair_name])
        print(f"[SCORE] {pair_name}: BLEU={bleu:.4f}, CHRF={chrf:.4f}", flush=True)

    en_to_indic = [name for name, row in pair_scores.items() if row["direction"] == "English_to_Indic"]
    indic_to_en = [name for name, row in pair_scores.items() if row["direction"] == "Indic_to_English"]

    scores = {
        "model_name": model_name,
        "tokenizer_source": tokenizer_source,
        "split": TEST_SPLIT,
        "average_bleu": avg([row["bleu"] for row in pair_scores.values()]),
        "average_chrf": avg([row["chrf"] for row in pair_scores.values()]),
        "english_to_indic_average_bleu": avg([pair_scores[name]["bleu"] for name in en_to_indic]),
        "english_to_indic_average_chrf": avg([pair_scores[name]["chrf"] for name in en_to_indic]),
        "indic_to_english_average_bleu": avg([pair_scores[name]["bleu"] for name in indic_to_en]),
        "indic_to_english_average_chrf": avg([pair_scores[name]["chrf"] for name in indic_to_en]),
        "pairs": pair_scores,
    }

    write_json(scores_path, scores)
    write_csv(csv_path, csv_rows)

    print_scores(scores)

    print("\nSaved:", flush=True)
    print(f"  {scores_path}", flush=True)
    print(f"  {csv_path}", flush=True)
    print(f"  {work_dir / 'predictions'}", flush=True)

    del llm
    del tokenizer
    cleanup_cuda()

    return scores


def print_scores(scores: Dict[str, Any]) -> None:
    print("\n" + "=" * 80, flush=True)
    print("FINAL BIDIRECTIONAL TEST SCORES", flush=True)
    print("=" * 80, flush=True)
    print("Pair\tBLEU\tCHRF\tN", flush=True)

    for pair_name, row in scores["pairs"].items():
        print(
            f"{pair_name}\t{row['bleu']:.2f}\t{row['chrf']:.2f}\t{row['num_examples']}",
            flush=True,
        )

    print("-" * 80, flush=True)
    print(
        f"English_to_Indic average: BLEU={scores['english_to_indic_average_bleu']:.2f}, "
        f"CHRF={scores['english_to_indic_average_chrf']:.2f}",
        flush=True,
    )
    print(
        f"Indic_to_English average: BLEU={scores['indic_to_english_average_bleu']:.2f}, "
        f"CHRF={scores['indic_to_english_average_chrf']:.2f}",
        flush=True,
    )
    print(
        f"All-pair average: BLEU={scores['average_bleu']:.2f}, "
        f"CHRF={scores['average_chrf']:.2f}",
        flush=True,
    )


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one bidirectional multilingual MT model on all English <-> Indic test pairs."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--work-dir", default=DEFAULT_WORK_DIR)
    parser.add_argument("--tensor-parallel-size", type=int, default=DEFAULT_TENSOR_PARALLEL_SIZE)
    parser.add_argument("--gpu-memory-utilization", type=float, default=DEFAULT_GPU_MEMORY_UTILIZATION)
    parser.add_argument("--dtype", default=DEFAULT_DTYPE)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--trust-remote-code", action="store_true", default=DEFAULT_TRUST_REMOTE_CODE)
    parser.add_argument("--no-tokenizer-patch", action="store_true", default=(not DEFAULT_PATCH_TOKENIZER))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
