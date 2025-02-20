"""
Adapted from John's script
"""
import os
os.environ['HF_HOME'] = "/scratch/eecs545w25_class_root/eecs545w25_class/cse545_reasoning/hf"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM

model_paths = {
    "llama-3.2-3b": "/scratch/eecs487w25_class_root/eecs487w25_class/shared_data/johnkimm_dir/models/llama-3.2-3b",
    "zephyr-3b": "/scratch/eecs487w25_class_root/eecs487w25_class/shared_data/johnkimm_dir/models/stablelm-zephyr-3b",
    "qwen2.5-3b": "/scratch/eecs487w25_class_root/eecs487w25_class/shared_data/johnkimm_dir/models/qwen2.5-3b"
}

prompt = "I have 3 apples, and my dad has 2 more apples than me. How many apples do we have in total?"

for model_name, model_path in model_paths.items():
    print("=" * 50)
    print(f"Generating answers from: {model_name}")
    print("=" * 50)
    
    if "llama" in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,  
            device_map="auto"           
        )
    else:
        # qwen2.5-3b
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    outputs = model.generate(
        input_ids,
        max_new_tokens=512,
        num_beams=10,
        num_return_sequences=10,
        early_stopping=True,
        pad_token_id=tokenizer.eos_token_id
    )
    for i, beam_output in enumerate(outputs, start=1):
        answer = tokenizer.decode(beam_output, skip_special_tokens=True)
        print(f"Beam {i}:\n{answer}\n")