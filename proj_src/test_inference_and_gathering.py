"""
Adapted from John's script
"""
import os
os.environ['HF_HOME'] = "/scratch/eecs545w25_class_root/eecs545w25_class/cse545_reasoning/hf"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM

# Change top k HERE
k = 2

model_list = [
    # "meta-llama/Llama-3.1-8B-Instruct",
    "stabilityai/stablelm-zephyr-3b",
    # "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    # "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    # "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    # "mistralai/Mistral-7B-Instruct-v0.3",
    # "Qwen/Qwen2.5-7B-Instruct",
    # "Qwen/Qwen2.5-Math-7B"
]

prompt = "I have 3 apples, and my dad has 2 more apples than me. How many apples do we have in total?"

all_outputs, top_k = [], []

for model_name in model_list:
    print("=" * 50)
    print(f"Generating answers from: {model_name}")
    print("=" * 50)
    
    if "qwen" in model_name.lower():
        # qwen2.5-3b
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  
            device_map="auto"           
        )
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenized_input = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = tokenized_input.input_ids.to(model.device)
    attention_mask = tokenized_input.attention_mask.to(model.device)

    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=512,
        num_beams=5,
        num_return_sequences=5,
        early_stopping=True,
        pad_token_id=tokenizer.pad_token_id
    )
    
    # Gather output
    current_model_outputs = []
    for i, beam_output in enumerate(outputs, start=1):
        answer = tokenizer.decode(beam_output, skip_special_tokens=True).strip()
        output_entry = {
            "model": model_name,
            "beam": i,
            "output": answer
        }
        all_outputs.append(output_entry)
        current_model_outputs.append(output_entry)

    # print for checking
    # for entry in all_outputs:
    #     print("Model: ", entry["model"])
    #     print("Beam: ", entry["beam"])
    #     print("Output: ", entry["output"])
    #     print("-" * 50)

    # Sort the outputs by the length
    # and get the top k longest outputs
    sorted_outputs = sorted(current_model_outputs, key=lambda entry: len(entry["output"]), reverse=True)
    top_k.extend(sorted_outputs[:k])


# Here, we will give a bigger model the aggregate of the answers we got from small models
# and output the "popular" answer

final_prompt = f"""Below are candidate answers for the question: "{prompt}"

Candidate Answers:
"""

for entry in top_k:
    final_prompt += f"- {entry['output']}\n"

final_prompt += """
Please analyze the candidate answers above and determine the most frequent final answer. 
Do not repeat the candidate answers or the question. 
Based solely on the candidate answers, provide one concise final answer along with a detailed explanation of your reasoning.
Please do not provide the candidate answers on the answer.
Your final answer should be in the following format:

Final Answer: <your answer>
Reasoning: <detailed explanation>
"""

# llama-3.1-8B model does not work well
# final_model_name = "meta-llama/Llama-3.1-8B-Instruct"
# final_tokenizer = AutoTokenizer.from_pretrained(final_model_name)
# final_model = AutoModelForCausalLM.from_pretrained(
#     final_model_name,
#     torch_dtype=torch.float16,
#     device_map="auto"
# )

final_model_name = "Qwen/Qwen2.5-7B-Instruct"
final_tokenizer = AutoTokenizer.from_pretrained(final_model_name, trust_remote_code=True)
final_model = AutoModelForCausalLM.from_pretrained(
    final_model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

if final_tokenizer.pad_token_id is None:
    final_tokenizer.pad_token_id = final_tokenizer.eos_token_id

tokenized_final_input = final_tokenizer(final_prompt, return_tensors="pt", padding=True)
final_input_ids = tokenized_final_input.input_ids.to(final_model.device)
final_attention_mask = tokenized_final_input.attention_mask.to(final_model.device)

# Maybe it's a good idea to explore more, using higher number for `num_beams`
final_output = final_model.generate(
    final_input_ids,
    attention_mask=final_attention_mask,
    max_new_tokens=1024,
    num_beams=10,
    early_stopping=True,
    pad_token_id=final_tokenizer.pad_token_id
)

# Take the input prompt out
prompt_length = final_input_ids.shape[-1]
generated_tokens = final_output[0][prompt_length:]
final_answer = final_tokenizer.decode(generated_tokens, skip_special_tokens=True)

print("=" * 50)
print("Final Answer from model:")
print(final_answer)
print("=" * 50)
