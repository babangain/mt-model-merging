#!/bin/bash
set -euo pipefail

# Do not run on V100 etc. vLLM may fall back and crash depending on attention backend.
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_V1=0

languages=("hi" "bn" "ta" "te")

model=anonymous/AnonMT_Hindi_English
SAFE_NAME=$(echo "$model" | sed 's/[^A-Za-z0-9._-]/_/g')

# Path where masked data already exists (built with Qwen template + teacher forcing)
DATA_DIR="data_masked_qwen/indic_en"

# Where to store activation outputs

# Collect masked activations on Indic source tokens
SPAN="src"
OUT_DIR="acts_${SAFE_NAME}/indic_en/$SPAN"
mkdir -p "$OUT_DIR"

echo "Model: $model"
echo "SAFE_NAME: $SAFE_NAME"
echo "DATA_DIR: $DATA_DIR"
echo "OUT_DIR: $OUT_DIR"
echo "SPAN: $SPAN"
echo

for lang in "${languages[@]}"; do
  echo "Collecting activations for lang=$lang"
  python src/activations_indic_en_granular.py \
    --model "$model" \
    --data_dir "$DATA_DIR" \
    --save_dir "$OUT_DIR" \
    --languages "$lang" \
    --span "$SPAN" \
    --batch_size 8 \
    --max_tokens 1
done

echo
echo "Done. Activations saved under: $OUT_DIR"
echo "Example file: $OUT_DIR/activation.hi.${SPAN}.pt"
