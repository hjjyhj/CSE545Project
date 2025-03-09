# Evaluate LLMs on AIME

This repository contains a minimal implementation of the evaluation code for LLMs on AIME.

## Requirements
PyTorch, transformers, numpy, pandas, sklearn, tqdm

## Example Usage

```bash
MODEL=meta-llama/Llama-2-7b-hf
device=0
CUDA_VISIBLE_DEVICES=$device python main.py \
    --model_name_or_path $MODEL \
    --output_dir outputs/$MODEL
```

## References

https://huggingface.co/datasets/di-zhang-fdu/AIME_1983_2024/viewer/default/train?p=2&views%5B%5D=train
