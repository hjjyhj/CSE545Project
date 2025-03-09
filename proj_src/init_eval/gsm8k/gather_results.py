import os
import re

model_list = [
    "stabilityai/stablelm-zephyr-3b",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    # "mistralai/Mistral-7B-Instruct-v0.3",
    # "Qwen/Qwen2.5-7B-Instruct",
    # "meta-llama/Llama-3.1-8B-Instruct",
    # "Qwen/Qwen2.5-Math-7B",
]

root_path = './output'
out_path = './output/all_results.csv'

with open(out_path, 'w') as fw:
    fw.write('model,acc\n')
    for model in model_list:
        model = model.split('/')[-1]
        result_path = os.path.join(root_path, model, 'scores.txt')
        with open(result_path, 'r') as fr:
            text = fr.read().strip()
            match = re.search(r'Accuracy: (.+)', text[:-1])
            acc = match.group(1)
            fw.write(f"{model},{acc}\n")
            
            