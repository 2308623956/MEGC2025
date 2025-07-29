# MEGC2025 Micro-Expression Recognition

This repository provides the inference code for our solution to the **MEGC2025 Micro-Expression Recognition Challenge**.

## Inference

Run the following scripts to perform inference on the respective datasets（Before running, please ensure that the paths to the dataset and model checkpoint files are correctly set in the configuration. Incorrect paths may lead to file not found errors or failed inference.）:

### 1. CASME3 Test Set

```bash
python infer_fix.py
```

### 2. SAMM Long Test Set

```bash
python infer_fix_samm_long.py
```

The output files will be saved in the directory:data/7b/

------

## Model Weights

Pretrained model weights are available on Hugging Face:

 [ALex230/MEGC2025 · Hugging Face](https://huggingface.co/ALex230/MEGC2025)

quark：
链接：https://pan.quark.cn/s/d97c04b9a934
提取码：wit6

------

## Required Dataset

Please make sure you have access to the required dataset before running the inference:

[Unseen Dataset for VQA](https://megc2025.github.io/challenge.html)

Required Dataset

## Required Package

```
torch
transformers
qwen_vl_utils
flash-attn
```

