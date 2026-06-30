#!/bin/bash

#HF_TOKEN=*

# huggingface-cli login --token $HF_TOKEN --add-to-git-credential

# python src/data.py

# python src/data_indic_en_granular.py \
#   --tokenizer_name Qwen/Qwen2.5-3B-Instruct \
#   --save_dir data_masked_qwen/indic_en \
#   --seq_len 1024 \
#   --target_mb 100



python src/data_en_indic_granular.py \
  --tokenizer_name Qwen/Qwen2.5-3B-Instruct \
  --save_dir data_masked_qwen/en_indic \
  --seq_len 1024 \
  --target_mb 100
