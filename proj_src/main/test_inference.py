"""
Adapted from John's script
"""
import os
os.environ['HF_HOME'] = "/scratch/eecs545w25_class_root/eecs545w25_class/cse545_reasoning/hf"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM

model_list = [
    # "meta-llama/Llama-3.1-8B-Instruct",
    "stabilityai/stablelm-zephyr-3b",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    # "mistralai/Mistral-7B-Instruct-v0.3",
    # "Qwen/Qwen2.5-7B-Instruct",
    # "Qwen/Qwen2.5-Math-7B"
]

prompt = "I have 3 apples, and my dad has 2 more apples than me. How many apples do we have in total?"

for model_name in model_list:
    print("=" * 50)
    print(f"Generating answers from: {model_name}")
    print("=" * 50)
    
    if "llama" in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  
            device_map="auto"           
        )
    else:
        # qwen2.5-3b
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    outputs = model.generate(
        input_ids,
        max_new_tokens=200,
        num_beams=3,
        num_return_sequences=3,
        early_stopping=True,
        pad_token_id=tokenizer.eos_token_id
    )
    for i, beam_output in enumerate(outputs, start=1):
        answer = tokenizer.decode(beam_output, skip_special_tokens=True)
        print(f"Beam {i}:\n{answer}\n")