import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM

# Change this parameter to select the top k longest outputs per model.
k = 3

# Define the candidate-generation model paths
model_paths = {
    "llama-3.2-3b": "/scratch/eecs487w25_class_root/eecs487w25_class/shared_data/johnkimm_dir/models/llama-3.2-3b",
    "zephyr-3b": "/scratch/eecs487w25_class_root/eecs487w25_class/shared_data/johnkimm_dir/models/stablelm-zephyr-3b",
    "qwen2.5-3b": "/scratch/eecs487w25_class_root/eecs487w25_class/shared_data/johnkimm_dir/models/qwen2.5-3b"
}

# Define the prompt for candidate models
prompt = "I have 3 apples, and my dad has 2 more apples than me. How many apples do we have in total?"

# Container for candidate answers
candidate_answers = []

# For each candidate model, generate outputs and select the top k longest beams.
for model_name, model_path in model_paths.items():
    print("=" * 50)
    print(f"Generating answers from: {model_name}")
    print("=" * 50)
    
    # Load the candidate model and its tokenizer based on model type.
    if "qwen" in model_name.lower():
        tokenizer_candidate = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model_candidate = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        # llama and zephyr
        tokenizer_candidate = AutoTokenizer.from_pretrained(model_path)
        model_candidate = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    # Set pad_token_id if not defined.
    if tokenizer_candidate.pad_token_id is None:
        tokenizer_candidate.pad_token_id = tokenizer_candidate.eos_token_id

    # Tokenize the prompt and generate outputs.
    input_ids = tokenizer_candidate(prompt, return_tensors="pt").input_ids.to(model_candidate.device)
    outputs = model_candidate.generate(
        input_ids,
        max_new_tokens=512,
        num_beams=10,
        num_return_sequences=10,
        early_stopping=True,
        pad_token_id=tokenizer_candidate.eos_token_id
    )
    
    # Decode the beams.
    beams = []
    for i, beam_output in enumerate(outputs, start=1):
        answer_text = tokenizer_candidate.decode(beam_output, skip_special_tokens=True).strip()
        beams.append(answer_text)
        print(f"Beam {i} from {model_name}:\n{answer_text}\n")
    
    # Sort beams by length (number of characters) and select the top k longest.
    beams_sorted = sorted(beams, key=lambda x: len(x), reverse=True)
    top_k_beams = beams_sorted[:k]
    
    # Append candidate answers with a label for their source.
    for ans in top_k_beams:
        candidate_answers.append(f"{model_name}: {ans}")

# Prepare the prompt for the llama-8b-Instruct model.
llama8b_model_path = "/scratch/eecs487w25_class_root/eecs487w25_class/shared_data/johnkimm_dir/models/llama-8b-Instruct"
candidates_text = "\n".join(candidate_answers)
final_prompt = (
    "Below are candidate answers from various models (each labeled by its source).\n"
    "Count the frequency of the final answers (ignoring the source labels) and output the answer that appears most frequently. "
    "If there is a tie, choose the answer that appears first.\n"
    "Candidates:\n"
    f"{candidates_text}\n\n"
    "Final Answer:"
)

print("=" * 50)
print("Final prompt for llama-8b-Instruct:")
print(final_prompt)
print("=" * 50)

# Load the llama-8b-Instruct model and its tokenizer.
tokenizer_llama8b = AutoTokenizer.from_pretrained(llama8b_model_path)
model_llama8b = AutoModelForCausalLM.from_pretrained(
    llama8b_model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

if tokenizer_llama8b.pad_token_id is None:
    tokenizer_llama8b.pad_token_id = tokenizer_llama8b.eos_token_id

# Tokenize the final prompt.
input_ids_llama8b = tokenizer_llama8b(final_prompt, return_tensors="pt").input_ids.to(model_llama8b.device)

# Generate the final answer with the llama-8b-Instruct model.
output_final = model_llama8b.generate(
    input_ids_llama8b,
    max_new_tokens=100,
    num_beams=5,
    early_stopping=True,
    pad_token_id=tokenizer_llama8b.eos_token_id
)

final_answer = tokenizer_llama8b.decode(output_final[0], skip_special_tokens=True)
print("=" * 50)
print("Final Answer from llama-8b-Instruct:")
print(final_answer)
print("=" * 50)
