#!/bin/bash

# do not run on V100 or so. on this kind of GPUs, VLLM backs up to using an alternative for flash attention and everything crashes
# pip install vllm
# pip install -U transformers

# HF_TOKEN=*
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_V1=0

# meta-llama/Llama-3.1-8B 
# meta-llama/Meta-Llama-3-8B 
# meta-llama/Llama-3.1-70B
# CohereLabs/aya-expanse-8b
# CohereLabs/aya-expanse-32b
# mistralai/Mistral-Nemo-Base-2407

# huggingface-cli login --token $HF_TOKEN --add-to-git-credential

languages=("hi" "bn" "ta" "te")

# model="Qwen/Qwen2.5-3B-Instruct"
model=$1
SAFE_NAME=$(echo "$model" | sed 's/[^A-Za-z0-9._-]/_/g')

# for lang in "${languages[@]}"
# do
#     echo "Running activation.py for language: $lang"
#     python src/activation.py -m $model -l $lang -s "qwen nemo"
# done
mkdir data_$SAFE_NAME
set -euo pipefail

cp data_qwen_flores_indic_en/id.{hi,bn,ta,te}.valid.nemo data_$SAFE_NAME/
for lang in "${languages[@]}"
do
    echo "Running activation.py for language: $lang"
    python src/activation.py -m $model -l $lang -s "$(echo $SAFE_NAME) nemo"
done
