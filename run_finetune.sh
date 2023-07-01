#!/bin/bash

# torchrun --nproc_per_node=8 train.py \
#     --model_name_or_path decapoda-research/llama-7b-hf \
#     --data_path ./alpaca_data.json \
#     --bf16 True \
#     --output_dir finetune_results \
#     --num_train_epochs 3 \
#     --per_device_train_batch_size 4 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 8 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 2000 \
#     --save_total_limit 1 \
#     --learning_rate 2e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --fsdp "full_shard auto_wrap" \
#     --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
#     --tf32 True

initial_context_length=2048
max_position_embeddings_scale_factor=1
block_size=$((initial_context_length * max_position_embeddings_scale_factor))


# stage1
# WANDB_DISABLED=true \
# deepspeed --num_gpus=8 run_clm.py \
#     --deepspeed ~/bwen/Finetune_LLMs/finetuning_repo/ds_config_stage1.json \
#     --model_name_or_path decapoda-research/llama-7b-hf \
#     --train_file ~/bwen/llama_variation/GPT-4-LLM/data/alpaca_gpt4_data.json \
#     --do_train \
#     --fp16 \
#     --overwrite_cache \
#     --evaluation_strategy no \
#     --output_dir finetune_results \
#     --num_train_epochs 1 \
#     --eval_steps 20 \
#     --gradient_accumulation_steps 32 \
#     --per_device_train_batch_size 1 \
#     --use_fast_tokenizer False \
#     --learning_rate 5e-06 \
#     --warmup_steps 10 \
#     --save_total_limit 1 \
#     --save_steps 20 \
#     --save_strategy steps \
#     --block_size $block_size \
#     --max_position_embeddings_scale_factor $max_position_embeddings_scale_factor

# stage2/3
WANDB_DISABLED=true \
deepspeed --num_gpus=8 run_clm.py \
    --deepspeed ./deepspeed_configs/ds_config_stage2.json \
    --model_name_or_path decapoda-research/llama-7b-hf \
    --train_file ./alpaca_gpt4_data.json \
    --do_train \
    --fp16 \
    --overwrite_cache \
    --evaluation_strategy no \
    --output_dir finetune_results_zero2_ctrl \
    --num_train_epochs 1 \
    --eval_steps 20 \
    --gradient_accumulation_steps 16 \
    --per_device_train_batch_size 2 \
    --use_fast_tokenizer False \
    --learning_rate 5e-06 \
    --warmup_steps 10 \
    --save_total_limit 1 \
    --save_steps 20 \
    --save_strategy steps \
    --block_size $block_size \
    --max_position_embeddings_scale_factor $max_position_embeddings_scale_factor


    # --train_file ./alpaca_data.json \
    # --do_eval \
    # --validation_file ~/bwen/Finetune_LLMs/finetuning_repo/validation.csv \
    # --load_best_model_at_end=True \
