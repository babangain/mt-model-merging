#!/bin/bash

HF_TOKEN=*

# huggingface-cli login --token $HF_TOKEN --add-to-git-credential

export CUDA_VISIBLE_DEVICES=1

# rates=(0.01 0.02 0.03 0.04 0.05)

# for i in "${!rates[@]}"; do
#     RATE=${rates[$i]}
#     SAVE_PATH="qwen-$((i + 1))"
#     echo "Running with top_rate=$RATE, save_path=$SAVE_PATH" # llama_3-1 llama-3.1
#     python src/identify.py --top_rate $RATE --activations "qwen_flores_indic_en nemo" --save_path "$SAVE_PATH"
# done
model=Qwen/Qwen2.5-3B-Instruct
SAFE_NAME=$(echo "$model" | sed 's/[^A-Za-z0-9._-]/_/g')
SPAN=tgt
mkdir -p  activation_mask/indic_en/$SAFE_NAME
OUT_DIR="acts_${SAFE_NAME}/indic_en/$SPAN"
python src/identify_indic_en_granular.py \
  --act_dir $OUT_DIR \
  --span $SPAN \
  --save_path activation_mask/indic_en/$SAFE_NAME/$SPAN.pt
