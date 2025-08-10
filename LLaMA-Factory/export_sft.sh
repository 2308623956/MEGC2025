# 原始模型
modelPath=/workspace/llm_weight/Qwen2.5-VL-7B-Instruct/
# 上一步微调得到的 LoRA 权重
adapterModelPath=/workspace/LLaMA-Factory-main/save_models/sft-lora/qwen2.5_vl_7b_sft_lora_model_250616/checkpoint-200/

llamafactory-cli export \
  --model_name_or_path $modelPath \
  --adapter_name_or_path $adapterModelPath \
  --template qwen \
  --finetuning_type lora \
  --export_dir /workspace/LLaMA-Factory-main/save_models/sft_lora/qwen2.5_vl_7b_sft_lora_model_250616/ \
  --export_size 2 \
  --export_device cpu \
  --export_legacy_format False
