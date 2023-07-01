#!/bin/bash

ORIG_MODEL_MAX_LEN=2048
MAX_POSITION_EMBEDDINGS_SCALE_FACTOR=16

NEW_MODEL_MAX_LEN=$((ORIG_MODEL_MAX_LEN * MAX_POSITION_EMBEDDINGS_SCALE_FACTOR))

torchrun --nproc_per_node=8 train_condense.py \
    $MAX_POSITION_EMBEDDINGS_SCALE_FACTOR \
    --model_name_or_path "decapoda-research/llama-7b-hf" \
    --data_path ./alpaca_gpt4_data.json \
    --bf16 True \
    --output_dir finetune_result_alpaca_gpt4 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --deepspeed "./configs/default_offload_opt_param.json" \
    --report_to none \
    --model_max_length $NEW_MODEL_MAX_LEN 

    # Not needed when using LongChat
    # --max_position_embeddings_scale_factor $MAX_POSITION_EMBEDDINGS_SCALE_FACTOR

    # --fsdp "full_shard auto_wrap offload" \
    # --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \