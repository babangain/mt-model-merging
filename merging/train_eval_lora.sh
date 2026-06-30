#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# LoRA-only English -> Indic MT experiment
#
# Behavior:
#   1. Search LoRA configs on 50K examples per language.
#   2. Skip any run if eval metric already exists.
#   3. Select best LoRA config by eval_loss.
#   4. Train final LoRA on full data per language.
#
# Important skip rule:
#   If eval_loss exists, the run is skipped.
#   Adapter checkpoint existence is not required for skipping.
#
# No Full SFT is run anywhere.
# ============================================================

BASE_DIR="${BASE_DIR:-/workspace/tacl_lora_only_search50k}"
DATA_DIR="${DATA_DIR:-/workspace/data}"
RUN_ROOT="${RUN_ROOT:-/workspace/models}"
YAML_ROOT="${YAML_ROOT:-${BASE_DIR}/yamls}"
LOG_ROOT="${LOG_ROOT:-${BASE_DIR}/logs}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-3B-Instruct}"
TEMPLATE="${TEMPLATE:-qwen}"

SEARCH_MAX_SAMPLES="${SEARCH_MAX_SAMPLES:-50}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3.0}"
CUTOFF_LEN="${CUTOFF_LEN:-1024}"

PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-8}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-16}"

PREPROCESSING_NUM_WORKERS="${PREPROCESSING_NUM_WORKERS:-16}"

EVAL_STEPS="${EVAL_STEPS:-5000}"
SAVE_STEPS="${SAVE_STEPS:-5000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-1}"

SELECTION_METRIC="${SELECTION_METRIC:-eval_loss}"

RUN_LORA_SEARCH="${RUN_LORA_SEARCH:-1}"
RUN_FINAL_LORA="${RUN_FINAL_LORA:-1}"

mkdir -p "${BASE_DIR}" "${RUN_ROOT}" "${YAML_ROOT}" "${LOG_ROOT}"

# LANGUAGES=("Hindi" "Bengali" "Tamil" "Telugu")
# LANGUAGES=("Telugu")
# LANGUAGES=("Bengali")
# LANGUAGES=("Hindi")
LANGUAGES=("Tamil")



# LORA_CONFIGS=(
#   "ffn_r64_a128_lr1e-4_d0.05|ffn|64|128|1.0e-4|0.05|gate_proj,up_proj,down_proj"
#   "ffn_r64_a128_lr5e-5_d0.05|ffn|64|128|5.0e-5|0.05|gate_proj,up_proj,down_proj"
#   "ffn_r128_a256_lr1e-4_d0.05|ffn|128|256|1.0e-4|0.05|gate_proj,up_proj,down_proj"
#   "ffn_r128_a256_lr5e-5_d0.05|ffn|128|256|5.0e-5|0.05|gate_proj,up_proj,down_proj"
#   "alllinear_r64_a128_lr1e-4_d0.05|alllinear|64|128|1.0e-4|0.05|q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
#   "alllinear_r64_a128_lr5e-5_d0.05|alllinear|64|128|5.0e-5|0.05|q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
#   "alllinear_r128_a256_lr1e-4_d0.05|alllinear|128|256|1.0e-4|0.05|q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
#   "alllinear_r128_a256_lr5e-5_d0.05|alllinear|128|256|5.0e-5|0.05|q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
# )

LORA_CONFIGS=(
  "alllinear_r128_a256_lr1e-4_d0.05|alllinear|128|256|1.0e-4|0.05|q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
)
require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "[ERROR] Missing required file: ${path}"
    exit 1
  fi
}

check_data() {
  require_file "${DATA_DIR}/dataset_info.json"

  for LANG in "${LANGUAGES[@]}"; do
    require_file "${DATA_DIR}/MT_En_${LANG}.json"
    require_file "${DATA_DIR}/MT_En_${LANG}_valid.json"
  done
}

write_helper_scripts() {
  cat > "${BASE_DIR}/metric_utils.py" <<'PY'
#!/usr/bin/env python3

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def read_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None


def parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(str(value).strip())
    except Exception:
        return None


def metric_direction(metric: str) -> str:
    metric = metric.lower()
    lower_is_better = ["loss", "ppl", "perplexity", "wer", "cer", "error"]
    if any(key in metric for key in lower_is_better):
        return "min"
    return "max"


def checkpoint_dirs(run_dir: Path) -> List[Path]:
    ckpts: List[Tuple[int, Path]] = []
    if not run_dir.exists():
        return []

    for path in run_dir.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        try:
            step = int(path.name.split("-")[-1])
        except Exception:
            step = -1
        ckpts.append((step, path))

    return [path for _, path in sorted(ckpts, key=lambda x: x[0], reverse=True)]


def read_metric_from_trainer_state(run_dir: Path, metric: str) -> Optional[float]:
    candidates = [run_dir] + checkpoint_dirs(run_dir)

    for base in candidates:
        state = read_json(base / "trainer_state.json")
        if not isinstance(state, dict):
            continue

        if metric == "eval_loss":
            best_metric = parse_float(state.get("best_metric"))
            if best_metric is not None:
                return best_metric

        best = None
        hist = state.get("log_history", [])
        if isinstance(hist, list):
            for row in hist:
                if not isinstance(row, dict):
                    continue
                if metric not in row:
                    continue

                val = parse_float(row.get(metric))
                if val is None:
                    continue

                if best is None:
                    best = val
                elif metric_direction(metric) == "min":
                    best = min(best, val)
                else:
                    best = max(best, val)

        if best is not None:
            return best

    return None


def read_metric_from_jsons(run_dir: Path, metric: str) -> Optional[float]:
    candidates = [
        run_dir / "eval_results.json",
        run_dir / "all_results.json",
        run_dir / "train_results.json",
    ]

    for ckpt in checkpoint_dirs(run_dir):
        candidates.extend(
            [
                ckpt / "eval_results.json",
                ckpt / "all_results.json",
                ckpt / "train_results.json",
            ]
        )

    for path in candidates:
        obj = read_json(path)
        if isinstance(obj, dict) and metric in obj:
            val = parse_float(obj.get(metric))
            if val is not None:
                return val

    return None


def read_metric_from_log(log_path: Path, metric: str) -> Optional[float]:
    if not log_path.exists():
        return None

    text = log_path.read_text(encoding="utf-8", errors="replace")

    patterns = [
        rf"{re.escape(metric)}\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)",
        rf"'{re.escape(metric)}'\s*:\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)",
        rf'"{re.escape(metric)}"\s*:\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)',
    ]

    vals = []
    for pattern in patterns:
        for match in re.findall(pattern, text):
            val = parse_float(match)
            if val is not None:
                vals.append(val)

    if not vals:
        return None

    if metric_direction(metric) == "min":
        return min(vals)
    return max(vals)


def get_selection_value(run_dir: Path, log_path: Path, metric: str) -> Optional[float]:
    value = read_metric_from_trainer_state(run_dir, metric)
    if value is not None:
        return value

    value = read_metric_from_jsons(run_dir, metric)
    if value is not None:
        return value

    return read_metric_from_log(log_path, metric)


def find_adapter(run_dir: Path) -> str:
    if (run_dir / "adapter_config.json").exists():
        return str(run_dir)

    for ckpt in checkpoint_dirs(run_dir):
        if (ckpt / "adapter_config.json").exists():
            return str(ckpt)

    return ""


def cmd_check_metric(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    log_path = Path(args.log)
    value = get_selection_value(run_dir, log_path, args.metric)

    if value is None:
        return 1

    print(value)
    return 0


def cmd_find_adapter(args: argparse.Namespace) -> int:
    adapter = find_adapter(Path(args.run_dir))
    if not adapter:
        return 1
    print(adapter)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_check = sub.add_parser("check-metric")
    p_check.add_argument("--run-dir", required=True)
    p_check.add_argument("--log", required=True)
    p_check.add_argument("--metric", required=True)

    p_adapter = sub.add_parser("find-adapter")
    p_adapter.add_argument("--run-dir", required=True)

    args = parser.parse_args()

    if args.cmd == "check-metric":
        raise SystemExit(cmd_check_metric(args))

    if args.cmd == "find-adapter":
        raise SystemExit(cmd_find_adapter(args))

    raise SystemExit(2)


if __name__ == "__main__":
    main()
PY

  chmod +x "${BASE_DIR}/metric_utils.py"

  cat > "${BASE_DIR}/select_best_lora.py" <<'PY'
#!/usr/bin/env python3

import argparse
import json
import re
import shlex
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def read_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None


def parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(str(value).strip())
    except Exception:
        return None


def metric_direction(metric: str) -> str:
    metric = metric.lower()
    lower_is_better = ["loss", "ppl", "perplexity", "wer", "cer", "error"]
    if any(key in metric for key in lower_is_better):
        return "min"
    return "max"


def checkpoint_dirs(run_dir: Path) -> List[Path]:
    ckpts: List[Tuple[int, Path]] = []
    if not run_dir.exists():
        return []

    for path in run_dir.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        try:
            step = int(path.name.split("-")[-1])
        except Exception:
            step = -1
        ckpts.append((step, path))

    return [path for _, path in sorted(ckpts, key=lambda x: x[0], reverse=True)]


def read_metric_from_trainer_state(run_dir: Path, metric: str) -> Optional[float]:
    candidates = [run_dir] + checkpoint_dirs(run_dir)

    for base in candidates:
        state = read_json(base / "trainer_state.json")
        if not isinstance(state, dict):
            continue

        if metric == "eval_loss":
            best_metric = parse_float(state.get("best_metric"))
            if best_metric is not None:
                return best_metric

        best = None
        hist = state.get("log_history", [])
        if isinstance(hist, list):
            for row in hist:
                if not isinstance(row, dict):
                    continue
                if metric not in row:
                    continue

                val = parse_float(row.get(metric))
                if val is None:
                    continue

                if best is None:
                    best = val
                elif metric_direction(metric) == "min":
                    best = min(best, val)
                else:
                    best = max(best, val)

        if best is not None:
            return best

    return None


def read_metric_from_jsons(run_dir: Path, metric: str) -> Optional[float]:
    candidates = [
        run_dir / "eval_results.json",
        run_dir / "all_results.json",
        run_dir / "train_results.json",
    ]

    for ckpt in checkpoint_dirs(run_dir):
        candidates.extend(
            [
                ckpt / "eval_results.json",
                ckpt / "all_results.json",
                ckpt / "train_results.json",
            ]
        )

    for path in candidates:
        obj = read_json(path)
        if isinstance(obj, dict) and metric in obj:
            val = parse_float(obj.get(metric))
            if val is not None:
                return val

    return None


def read_metric_from_log(log_path: Path, metric: str) -> Optional[float]:
    if not log_path.exists():
        return None

    text = log_path.read_text(encoding="utf-8", errors="replace")

    patterns = [
        rf"{re.escape(metric)}\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)",
        rf"'{re.escape(metric)}'\s*:\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)",
        rf'"{re.escape(metric)}"\s*:\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)',
    ]

    vals = []
    for pattern in patterns:
        for match in re.findall(pattern, text):
            val = parse_float(match)
            if val is not None:
                vals.append(val)

    if not vals:
        return None

    if metric_direction(metric) == "min":
        return min(vals)
    return max(vals)


def get_selection_value(run_dir: Path, log_path: Path, metric: str) -> Optional[float]:
    value = read_metric_from_trainer_state(run_dir, metric)
    if value is not None:
        return value

    value = read_metric_from_jsons(run_dir, metric)
    if value is not None:
        return value

    return read_metric_from_log(log_path, metric)


def is_better(a: Dict[str, Any], b: Dict[str, Any], metric: str) -> bool:
    if metric_direction(metric) == "min":
        return a["selection_value"] < b["selection_value"]
    return a["selection_value"] > b["selection_value"]


def shell_quote(value: Any) -> str:
    return shlex.quote(str(value))


def write_env(path: Path, row: Dict[str, Any]) -> None:
    lines = [
        f"BEST_CONFIG_NAME={shell_quote(row['config_name'])}",
        f"BEST_TARGET_GROUP={shell_quote(row['target_group'])}",
        f"BEST_LORA_RANK={shell_quote(row['lora_rank'])}",
        f"BEST_LORA_ALPHA={shell_quote(row['lora_alpha'])}",
        f"BEST_LORA_LR={shell_quote(row['learning_rate'])}",
        f"BEST_LORA_DROPOUT={shell_quote(row['lora_dropout'])}",
        f"BEST_LORA_TARGET={shell_quote(row['lora_target'])}",
        f"BEST_SELECTION_METRIC={shell_quote(row['selection_metric'])}",
        f"BEST_SELECTION_VALUE={shell_quote(row['selection_value'])}",
        f"BEST_SEARCH_RUN_DIR={shell_quote(row['run_dir'])}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", required=True)
    parser.add_argument("--search-root", required=True)
    parser.add_argument("--log-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--metric", default="eval_loss")
    args = parser.parse_args()

    lang_root = Path(args.search_root) / args.language
    log_root = Path(args.log_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not lang_root.exists():
        raise SystemExit(f"Missing search directory: {lang_root}")

    rows: List[Dict[str, Any]] = []

    for run_dir in sorted([p for p in lang_root.iterdir() if p.is_dir()]):
        meta = read_json(run_dir / "config_meta.json")
        if not isinstance(meta, dict):
            continue

        config_name = str(meta.get("config_name", run_dir.name))
        log_path = log_root / f"lora_search_{args.language}_{config_name}.log"

        selection_value = get_selection_value(run_dir, log_path, args.metric)
        if selection_value is None:
            print(f"[WARN] No {args.metric} found for {run_dir}")
            continue

        rows.append(
            {
                "language": args.language,
                "config_name": config_name,
                "target_group": meta.get("target_group", ""),
                "lora_rank": meta.get("lora_rank", ""),
                "lora_alpha": meta.get("lora_alpha", ""),
                "learning_rate": meta.get("learning_rate", ""),
                "lora_dropout": meta.get("lora_dropout", ""),
                "lora_target": meta.get("lora_target", ""),
                "selection_metric": args.metric,
                "selection_value": selection_value,
                "run_dir": str(run_dir),
            }
        )

    if not rows:
        raise SystemExit(f"No selectable LoRA configs found for {args.language} using metric={args.metric}")

    best = rows[0]
    for row in rows[1:]:
        if is_better(row, best, args.metric):
            best = row

    all_json = out_dir / "all_lora_search_results.json"
    best_json = out_dir / "best_lora_config.json"
    best_env = out_dir / "best_lora_config.env"

    all_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    best_json.write_text(json.dumps(best, indent=2, ensure_ascii=False), encoding="utf-8")
    write_env(best_env, best)

    print(f"[SELECT] {args.language}")
    print(f"  best_config={best['config_name']}")
    print(f"  {args.metric}={best['selection_value']}")
    print(f"  env={best_env}")


if __name__ == "__main__":
    main()
PY

  chmod +x "${BASE_DIR}/select_best_lora.py"

  cat > "${BASE_DIR}/collect_lora_results.py" <<'PY'
#!/usr/bin/env python3

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def read_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None


def parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(str(value).strip())
    except Exception:
        return None


def checkpoint_dirs(run_dir: Path) -> List[Path]:
    ckpts: List[Tuple[int, Path]] = []
    if not run_dir.exists():
        return []

    for path in run_dir.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        try:
            step = int(path.name.split("-")[-1])
        except Exception:
            step = -1
        ckpts.append((step, path))

    return [path for _, path in sorted(ckpts, key=lambda x: x[0], reverse=True)]


def find_adapter(run_dir: Path) -> str:
    if (run_dir / "adapter_config.json").exists():
        return str(run_dir)

    for ckpt in checkpoint_dirs(run_dir):
        if (ckpt / "adapter_config.json").exists():
            return str(ckpt)

    return ""


def get_eval_loss_from_state(run_dir: Path) -> Optional[float]:
    for base in [run_dir] + checkpoint_dirs(run_dir):
        state = read_json(base / "trainer_state.json")
        if not isinstance(state, dict):
            continue

        best_metric = parse_float(state.get("best_metric"))
        if best_metric is not None:
            return best_metric

        hist = state.get("log_history", [])
        vals = []
        if isinstance(hist, list):
            for row in hist:
                if isinstance(row, dict) and "eval_loss" in row:
                    val = parse_float(row.get("eval_loss"))
                    if val is not None:
                        vals.append(val)

        if vals:
            return min(vals)

    return None


def get_eval_loss_from_jsons(run_dir: Path) -> Optional[float]:
    candidates = [
        run_dir / "eval_results.json",
        run_dir / "all_results.json",
        run_dir / "train_results.json",
    ]

    for ckpt in checkpoint_dirs(run_dir):
        candidates.extend(
            [
                ckpt / "eval_results.json",
                ckpt / "all_results.json",
                ckpt / "train_results.json",
            ]
        )

    for path in candidates:
        obj = read_json(path)
        if isinstance(obj, dict) and "eval_loss" in obj:
            val = parse_float(obj.get("eval_loss"))
            if val is not None:
                return val

    return None


def get_eval_loss_from_log(log_path: Path) -> Optional[float]:
    if not log_path.exists():
        return None

    text = log_path.read_text(encoding="utf-8", errors="replace")

    patterns = [
        r"eval_loss\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)",
        r"'eval_loss'\s*:\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)",
        r'"eval_loss"\s*:\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)',
    ]

    vals = []
    for pattern in patterns:
        for match in re.findall(pattern, text):
            val = parse_float(match)
            if val is not None:
                vals.append(val)

    if not vals:
        return None

    return min(vals)


def get_eval_loss(run_dir: Path, log_path: Optional[Path] = None) -> Optional[float]:
    value = get_eval_loss_from_state(run_dir)
    if value is not None:
        return value

    value = get_eval_loss_from_jsons(run_dir)
    if value is not None:
        return value

    if log_path is not None:
        return get_eval_loss_from_log(log_path)

    return None


def add_row(rows: List[Dict[str, Any]], family: str, language: str, run_dir: Path, log_path: Optional[Path]) -> None:
    meta = read_json(run_dir / "config_meta.json")
    if not isinstance(meta, dict):
        meta = {}

    rows.append(
        {
            "family": family,
            "language": language,
            "phase": meta.get("phase", ""),
            "config_name": meta.get("config_name", run_dir.name),
            "target_group": meta.get("target_group", ""),
            "lora_rank": meta.get("lora_rank", ""),
            "lora_alpha": meta.get("lora_alpha", ""),
            "learning_rate": meta.get("learning_rate", ""),
            "lora_dropout": meta.get("lora_dropout", ""),
            "lora_target": meta.get("lora_target", ""),
            "max_samples": meta.get("max_samples", ""),
            "cutoff_len": meta.get("cutoff_len", ""),
            "precision": meta.get("precision", ""),
            "eval_loss": get_eval_loss(run_dir, log_path),
            "artifact_path": find_adapter(run_dir),
            "run_dir": str(run_dir),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--log-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    log_root = Path(args.log_root)
    rows: List[Dict[str, Any]] = []

    search_root = run_root / "lora_search_50k"
    if search_root.exists():
        for lang_dir in sorted([p for p in search_root.iterdir() if p.is_dir()]):
            for cfg_dir in sorted([p for p in lang_dir.iterdir() if p.is_dir()]):
                log_path = log_root / f"lora_search_{lang_dir.name}_{cfg_dir.name}.log"
                add_row(rows, "lora_search_50k", lang_dir.name, cfg_dir, log_path)

    final_root = run_root / "final_lora_full"
    if final_root.exists():
        for lang_dir in sorted([p for p in final_root.iterdir() if p.is_dir()]):
            log_path = log_root / f"final_lora_full_{lang_dir.name}.log"
            add_row(rows, "final_lora_full", lang_dir.name, lang_dir, log_path)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "family",
        "language",
        "phase",
        "config_name",
        "target_group",
        "lora_rank",
        "lora_alpha",
        "learning_rate",
        "lora_dropout",
        "lora_target",
        "max_samples",
        "cutoff_len",
        "precision",
        "eval_loss",
        "artifact_path",
        "run_dir",
    ]

    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[DONE] Wrote summary: {out}")


if __name__ == "__main__":
    main()
PY

  chmod +x "${BASE_DIR}/collect_lora_results.py"
}

metric_value_for_run() {
  local run_dir="$1"
  local log="$2"
  local metric="$3"

  python "${BASE_DIR}/metric_utils.py" check-metric \
    --run-dir "${run_dir}" \
    --log "${log}" \
    --metric "${metric}"
}

metric_exists_for_run() {
  local run_dir="$1"
  local log="$2"
  local metric="$3"

  metric_value_for_run "${run_dir}" "${log}" "${metric}" >/dev/null 2>&1
}

find_lora_adapter_path() {
  local run_dir="$1"

  python "${BASE_DIR}/metric_utils.py" find-adapter \
    --run-dir "${run_dir}"
}

run_lora_train() {
  local run_dir="$1"
  local log="$2"
  local yaml="$3"
  local metric="${4:-${SELECTION_METRIC}}"

  mkdir -p "${run_dir}" "$(dirname "${log}")"

  if metric_exists_for_run "${run_dir}" "${log}" "${metric}"; then
    local existing_metric
    existing_metric="$(metric_value_for_run "${run_dir}" "${log}" "${metric}")"
    echo "[SKIP] Existing ${metric} found for ${run_dir}: ${existing_metric}"
    echo "[SKIP] Adapter checkpoint is not required for skipping."
    return 0
  fi

  echo "[RUN] LoRA train: ${run_dir}"
  llamafactory-cli train "${yaml}" 2>&1 | tee "${log}"

  if ! metric_exists_for_run "${run_dir}" "${log}" "${metric}"; then
    echo "[ERROR] ${metric} not found after training: ${run_dir}"
    echo "[ERROR] Check log: ${log}"
    exit 1
  fi

  local final_metric
  final_metric="$(metric_value_for_run "${run_dir}" "${log}" "${metric}")"
  echo "[DONE] ${metric}=${final_metric}"

  if find_lora_adapter_path "${run_dir}" >/dev/null 2>&1; then
    echo "[DONE] LoRA adapter found: $(find_lora_adapter_path "${run_dir}")"
  else
    echo "[WARN] ${metric} exists, but no adapter_config.json was found under: ${run_dir}"
    echo "[WARN] Keeping the run because metric exists."
  fi
}

write_lora_train_yaml() {
  local yaml="$1"
  local out_dir="$2"
  local train_dataset="$3"
  local valid_dataset="$4"
  local max_samples="$5"
  local rank="$6"
  local alpha="$7"
  local lr="$8"
  local dropout="$9"
  local target="${10}"

  mkdir -p "$(dirname "${yaml}")" "${out_dir}"

  cat > "${yaml}" <<EOF
### model
model_name_or_path: ${MODEL_NAME}

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: ${rank}
lora_alpha: ${alpha}
lora_dropout: ${dropout}
lora_target: "${target}"

### dataset
dataset: ${train_dataset}
eval_dataset: ${valid_dataset}
template: ${TEMPLATE}
cutoff_len: ${CUTOFF_LEN}
max_samples: ${max_samples}
overwrite_cache: true
preprocessing_num_workers: ${PREPROCESSING_NUM_WORKERS}
flash_attn: auto
dataset_dir: ${DATA_DIR}

### output
output_dir: ${out_dir}
logging_steps: 10
save_strategy: steps
save_steps: ${SAVE_STEPS}
plot_loss: true

### train
per_device_train_batch_size: ${PER_DEVICE_TRAIN_BATCH_SIZE}
gradient_accumulation_steps: ${GRADIENT_ACCUMULATION_STEPS}
learning_rate: ${lr}
num_train_epochs: ${NUM_TRAIN_EPOCHS}
lr_scheduler_type: inverse_sqrt
pure_bf16: true
ddp_timeout: 180000000
max_grad_norm: 1.0
report_to: tensorboard
trust_remote_code: true
include_num_input_tokens_seen: true
optim: adamw_torch

# Valid
per_device_eval_batch_size: ${PER_DEVICE_EVAL_BATCH_SIZE}
eval_strategy: steps
save_total_limit: ${SAVE_TOTAL_LIMIT}
load_best_model_at_end: true
metric_for_best_model: eval_loss
greater_is_better: false
eval_steps: ${EVAL_STEPS}
EOF
}

write_lora_final_yaml() {
  local yaml="$1"
  local out_dir="$2"
  local train_dataset="$3"
  local valid_dataset="$4"
  local rank="$5"
  local alpha="$6"
  local lr="$7"
  local dropout="$8"
  local target="$9"

  mkdir -p "$(dirname "${yaml}")" "${out_dir}"

  cat > "${yaml}" <<EOF
### model
model_name_or_path: ${MODEL_NAME}

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: ${rank}
lora_alpha: ${alpha}
lora_dropout: ${dropout}
lora_target: "${target}"

### dataset
dataset: ${train_dataset}
eval_dataset: ${valid_dataset}
template: ${TEMPLATE}
cutoff_len: ${CUTOFF_LEN}
overwrite_cache: true
preprocessing_num_workers: ${PREPROCESSING_NUM_WORKERS}
flash_attn: auto
dataset_dir: ${DATA_DIR}

### output
output_dir: ${out_dir}
logging_steps: 10
save_strategy: steps
save_steps: ${SAVE_STEPS}
plot_loss: true

### train
per_device_train_batch_size: ${PER_DEVICE_TRAIN_BATCH_SIZE}
gradient_accumulation_steps: ${GRADIENT_ACCUMULATION_STEPS}
learning_rate: ${lr}
num_train_epochs: ${NUM_TRAIN_EPOCHS}
lr_scheduler_type: inverse_sqrt
pure_bf16: true
ddp_timeout: 180000000
max_grad_norm: 1.0
report_to: tensorboard
trust_remote_code: true
include_num_input_tokens_seen: true
optim: adamw_torch

# Valid
per_device_eval_batch_size: ${PER_DEVICE_EVAL_BATCH_SIZE}
eval_strategy: steps
save_total_limit: ${SAVE_TOTAL_LIMIT}
load_best_model_at_end: true
metric_for_best_model: eval_loss
greater_is_better: false
eval_steps: ${EVAL_STEPS}
EOF
}

write_lora_meta() {
  local file="$1"
  local lang="$2"
  local phase="$3"
  local config_name="$4"
  local target_group="$5"
  local rank="$6"
  local alpha="$7"
  local lr="$8"
  local dropout="$9"
  local target="${10}"
  local max_samples="${11}"

  mkdir -p "$(dirname "${file}")"

  cat > "${file}" <<EOF
{
  "language": "${lang}",
  "phase": "${phase}",
  "config_name": "${config_name}",
  "target_group": "${target_group}",
  "lora_rank": "${rank}",
  "lora_alpha": "${alpha}",
  "learning_rate": "${lr}",
  "lora_dropout": "${dropout}",
  "lora_target": "${target}",
  "max_samples": "${max_samples}",
  "cutoff_len": ${CUTOFF_LEN},
  "per_device_train_batch_size": ${PER_DEVICE_TRAIN_BATCH_SIZE},
  "gradient_accumulation_steps": ${GRADIENT_ACCUMULATION_STEPS},
  "num_train_epochs": "${NUM_TRAIN_EPOCHS}",
  "selection_metric": "${SELECTION_METRIC}",
  "precision": "pure_bf16"
}
EOF
}

echo "============================================================"
echo "LoRA-only MT experiment"
echo "============================================================"
echo "BASE_DIR=${BASE_DIR}"
echo "DATA_DIR=${DATA_DIR}"
echo "RUN_ROOT=${RUN_ROOT}"
echo "YAML_ROOT=${YAML_ROOT}"
echo "LOG_ROOT=${LOG_ROOT}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "SEARCH_MAX_SAMPLES=${SEARCH_MAX_SAMPLES}"
echo "FINAL_MAX_SAMPLES=FULL_DATA_NO_MAX_SAMPLES"
echo "CUTOFF_LEN=${CUTOFF_LEN}"
echo "PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE}"
echo "GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}"
echo "PER_DEVICE_EVAL_BATCH_SIZE=${PER_DEVICE_EVAL_BATCH_SIZE}"
echo "NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS}"
echo "EVAL_STEPS=${EVAL_STEPS}"
echo "SAVE_STEPS=${SAVE_STEPS}"
echo "SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT}"
echo "SELECTION_METRIC=${SELECTION_METRIC}"
echo "RUN_LORA_SEARCH=${RUN_LORA_SEARCH}"
echo "RUN_FINAL_LORA=${RUN_FINAL_LORA}"
echo "LoRA configs available: ${#LORA_CONFIGS[@]}"
echo "Skip rule: skip if ${SELECTION_METRIC} exists"
echo "============================================================"

check_data
write_helper_scripts

# ============================================================
# 1. LoRA search on 50K examples
# ============================================================

if [[ "${RUN_LORA_SEARCH}" == "1" ]]; then
  for LANG in "${LANGUAGES[@]}"; do
    echo "================ LoRA search 50K: ${LANG} ================"

    TRAIN_DATASET="MT_En_${LANG}"
    VALID_DATASET="MT_En_${LANG}_valid"

    for SPEC in "${LORA_CONFIGS[@]}"; do
      IFS='|' read -r CONFIG_NAME TARGET_GROUP RANK ALPHA LR DROPOUT TARGET <<< "${SPEC}"

      RUN_DIR="${RUN_ROOT}/lora_search_50k/${LANG}/${CONFIG_NAME}"
      YAML="${YAML_ROOT}/lora_search_50k/${LANG}/${CONFIG_NAME}.yaml"
      LOG="${LOG_ROOT}/lora_search_${LANG}_${CONFIG_NAME}.log"

      write_lora_train_yaml \
        "${YAML}" \
        "${RUN_DIR}" \
        "${TRAIN_DATASET}" \
        "${VALID_DATASET}" \
        "${SEARCH_MAX_SAMPLES}" \
        "${RANK}" \
        "${ALPHA}" \
        "${LR}" \
        "${DROPOUT}" \
        "${TARGET}"

      write_lora_meta \
        "${RUN_DIR}/config_meta.json" \
        "${LANG}" \
        "search_50k" \
        "${CONFIG_NAME}" \
        "${TARGET_GROUP}" \
        "${RANK}" \
        "${ALPHA}" \
        "${LR}" \
        "${DROPOUT}" \
        "${TARGET}" \
        "${SEARCH_MAX_SAMPLES}"

      run_lora_train "${RUN_DIR}" "${LOG}" "${YAML}" "${SELECTION_METRIC}"
    done

    python "${BASE_DIR}/select_best_lora.py" \
      --language "${LANG}" \
      --search-root "${RUN_ROOT}/lora_search_50k" \
      --log-root "${LOG_ROOT}" \
      --output-dir "${RUN_ROOT}/lora_search_50k/${LANG}" \
      --metric "${SELECTION_METRIC}"
  done
else
  echo "[SKIP] RUN_LORA_SEARCH=0"
fi

# ============================================================
# 2. Final LoRA on full data
# ============================================================

if [[ "${RUN_FINAL_LORA}" == "1" ]]; then
  for LANG in "${LANGUAGES[@]}"; do
    echo "================ Final LoRA full data: ${LANG} ================"

    TRAIN_DATASET="MT_En_${LANG}"
    VALID_DATASET="MT_En_${LANG}_valid"

    BEST_ENV="${RUN_ROOT}/lora_search_50k/${LANG}/best_lora_config.env"

    if [[ ! -f "${BEST_ENV}" ]]; then
      echo "[ERROR] Missing best LoRA config for ${LANG}: ${BEST_ENV}"
      echo "[ERROR] Run with RUN_LORA_SEARCH=1 first."
      exit 1
    fi

    source "${BEST_ENV}"

    RUN_DIR="${RUN_ROOT}/final_lora_full/${LANG}"
    YAML="${YAML_ROOT}/final_lora_full/${LANG}.yaml"
    LOG="${LOG_ROOT}/final_lora_full_${LANG}.log"

    echo "[BEST] ${LANG}: ${BEST_CONFIG_NAME}, ${BEST_SELECTION_METRIC}=${BEST_SELECTION_VALUE}"
    echo "[BEST] group=${BEST_TARGET_GROUP}, rank=${BEST_LORA_RANK}, alpha=${BEST_LORA_ALPHA}, lr=${BEST_LORA_LR}, dropout=${BEST_LORA_DROPOUT}"
    echo "[BEST] target=${BEST_LORA_TARGET}"

    write_lora_final_yaml \
      "${YAML}" \
      "${RUN_DIR}" \
      "${TRAIN_DATASET}" \
      "${VALID_DATASET}" \
      "${BEST_LORA_RANK}" \
      "${BEST_LORA_ALPHA}" \
      "${BEST_LORA_LR}" \
      "${BEST_LORA_DROPOUT}" \
      "${BEST_LORA_TARGET}"

    write_lora_meta \
      "${RUN_DIR}/config_meta.json" \
      "${LANG}" \
      "final_full_data" \
      "${BEST_CONFIG_NAME}" \
      "${BEST_TARGET_GROUP}" \
      "${BEST_LORA_RANK}" \
      "${BEST_LORA_ALPHA}" \
      "${BEST_LORA_LR}" \
      "${BEST_LORA_DROPOUT}" \
      "${BEST_LORA_TARGET}" \
      "FULL_DATA_NO_MAX_SAMPLES"

    run_lora_train "${RUN_DIR}" "${LOG}" "${YAML}" "${SELECTION_METRIC}"
  done
else
  echo "[SKIP] RUN_FINAL_LORA=0"
fi

# ============================================================
# 3. Collect LoRA-only results
# ============================================================

SUMMARY_CSV="${RUN_ROOT}/lora_only_search50k_full_summary.csv"

python "${BASE_DIR}/collect_lora_results.py" \
  --run-root "${RUN_ROOT}" \
  --log-root "${LOG_ROOT}" \
  --output "${SUMMARY_CSV}"

echo "============================================================"
echo "Done."
echo "Summary: ${SUMMARY_CSV}"
echo "LoRA search dirs: ${RUN_ROOT}/lora_search_50k/{Language}/{config}"
echo "Final LoRA dirs: ${RUN_ROOT}/final_lora_full/{Language}"
echo "No Full SFT was run."
echo "Skip rule used: skip if ${SELECTION_METRIC} exists."
echo "============================================================"