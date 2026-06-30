#!/bin/bash
set -euo pipefail

# Do not run on V100 etc. vLLM may fall back and crash depending on attention backend.
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_V1=0

languages=("hi" "bn" "ta" "te")

model=anonymous/AnonMT_English_Tamil
# model=Qwen/Qwen2.5-3B-Instruct
SPAN="src"
######## SRC #####
SAFE_NAME=$(echo "$model" | sed 's/[^A-Za-z0-9._-]/_/g')

# Path where masked data already exists (built with Qwen template + teacher forcing)
DATA_DIR="data_masked_qwen/en_indic"

# Where to store activation outputs

# Collect masked activations on Indic source tokens
OUT_DIR="acts_${SAFE_NAME}/en_indic/$SPAN"
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


mkdir -p  activation_mask/en_indic/$SAFE_NAME
OUT_DIR="acts_${SAFE_NAME}/en_indic/$SPAN"
python src/identify_indic_en_granular.py \
  --act_dir $OUT_DIR \
  --span $SPAN \
  --save_path activation_mask/en_indic/$SAFE_NAME/$SPAN.pt




#### TGT#######
SPAN=tgt
SAFE_NAME=$(echo "$model" | sed 's/[^A-Za-z0-9._-]/_/g')

# Path where masked data already exists (built with Qwen template + teacher forcing)
DATA_DIR="data_masked_qwen/en_indic"

# Where to store activation outputs

# Collect masked activations on Indic source tokens
OUT_DIR="acts_${SAFE_NAME}/en_indic/$SPAN"
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


mkdir -p  activation_mask/en_indic/$SAFE_NAME
OUT_DIR="acts_${SAFE_NAME}/en_indic/$SPAN"
python src/identify_indic_en_granular.py \
  --act_dir $OUT_DIR \
  --span $SPAN \
  --save_path activation_mask/en_indic/$SAFE_NAME/$SPAN.pt


#######END ####
python src/indic_en_compare_src_tgt.py \
  --src activation_mask/en_indic/$SAFE_NAME/src.pt \
  --tgt activation_mask/en_indic/$SAFE_NAME/tgt.pt \
  --languages "hi bn ta te"
