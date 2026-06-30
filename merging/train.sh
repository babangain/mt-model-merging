export BASE_DIR=/workspace
YAML_FILE=$1
pip install datasets
#python download_data.py
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation
cd ..
torchrun --nnodes=1 --node_rank=0 --nproc_per_node=8 --rdzv_id=101 --rdzv_endpoint=$MASTER_ADDR:23456 LLaMA-Factory/src/llamafactory/launcher.py yamls_qwen/$YAML_FILE
# llamafactory-cli train \
#     --stage sft \
#     --do_train True \
#     --model_name_or_path meta-llama/Llama-3.2-1B \
#     --preprocessing_num_workers 16 \
#     --finetuning_type full \
#     --template llama3 \
#     --rope_scaling llama3 \
#     --flash_attn auto \
#     --use_unsloth True \
#     --dataset_dir $BASE_DIR/data \
#     --dataset MT_En_Hindi \
#     --cutoff_len 2048 \
#     --learning_rate 5e-05 \
#     --num_train_epochs 3.0 \
#     --max_samples -1 \
#     --per_device_train_batch_size 32 \
#     --gradient_accumulation_steps 8 \
#     --lr_scheduler_type inverse_sqrt \
#     --max_grad_norm 1.0 \
#     --logging_steps 5 \
#     --save_steps 5000 \
#     --warmup_steps 0 \
#     --packing False \
#     --enable_thinking False \
#     --report_to tensorboard \
#     --output_dir $BASE_DIR/saves/Llama-3.2-1B/MT_En_Hindi \
#     --pure_bf16 True \
#     --plot_loss True \
#     --trust_remote_code True \
#     --ddp_timeout 180000000 \
#     --include_num_input_tokens_seen True \
#     --optim adamw_torch