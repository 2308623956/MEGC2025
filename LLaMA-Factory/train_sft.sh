#!/bin/bash
nohup deepspeed --include localhost:1,2,3,4 ./src/train.py \
--stage sft   \
--model_name_or_path /workspace/llm_weight/Qwen2.5-VL-7B-Instruct/  \
--do_train  \
--dataset_dir ./data/sft/20250626 \
--dataset  vqa_samm_test \
--template qwen2_vl  \
--finetuning_type full \
--output_dir save_models/qwen2.5_vl_7b_sft_model_250626/ \
--per_device_train_batch_size 2   \
--gradient_accumulation_steps 2 \
--warmup_ratio 0.05  \
--lr_scheduler_type cosine   \
--logging_steps 5   \
--save_steps 50   \
--learning_rate 2.0e-5 \
--num_train_epochs 20 \
--cutoff_len 4096 \
--overwrite_output_dir  \
--plot_loss \
--fp16  \
--deepspeed ./examples/deepspeed/ds_z3_config.json >train_sft.log 2>&1 &

# --val_size 0.1 \
# --per_device_eval_batch_size 1 \
# --eval_strategy  steps \
# --eval_steps  100 \