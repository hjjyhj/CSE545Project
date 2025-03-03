import os
os.environ['HF_HOME'] = "/scratch/eecs545w25_class_root/eecs545w25_class/cse545_reasoning/hf"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM

# Params
k = 2
MAX_ITER = 3

# Models in use (SLMs)
model_list = [
    "stabilityai/stablelm-zephyr-3b",
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
]

original_prompt = "I have 3 apples, and my dad has 2 more apples than me. How many apples do we have in total?"
current_prompt = [original_prompt] * len(model_list) # This will be updated, when there is disagreement between SLMs.

# Judge model
# Qwen2.5-7B-Instruct model is used here. We may need a stronger model in the future

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

# Iterate until max iteration is reached or the models comes to consensus
it = 0
while True:
    print("=" * 80)
    print(f"Iteration {it + 1}: Generating candidate answers")
    print("=" * 80)

    top_k_outputs = []
    for i, model_name in enumerate(model_list):
        print("-" * 80)
        print(f"Generating answers from: {model_name}")
        print("-" * 80)
        # Load candidate model and tokenizer
        if "qwen" in model_name.lower():
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                # trust_remote_code=True
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        input_ids = tokenizer(current_prompt[i], return_tensors="pt").input_ids.to(model.device)

        outputs = model.generate(
            input_ids,
            max_new_tokens=512,
            num_beams=5,
            num_return_sequences=5,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id
        )

        current_model_outputs = []
        for b, beam_output in enumerate(outputs, start=1):
            answer = tokenizer.decode(beam_output, skip_special_tokens=True).strip()
            output_entry = {
                "model": model_name,
                "beam": b,
                "output": answer
            }
            current_model_outputs.append(output_entry)
            print(f"Model: {model_name}, Beam {b}: {answer}\n")

        # Get the longest k outputs from each model.
        sorted_outputs = sorted(current_model_outputs, key=lambda entry: len(entry["output"]), reverse=True)
        top_k_outputs.extend(sorted_outputs[:k])

    judge_prompt = f"""Below are candidate answers for the question: "{original_prompt}"
Candidate Answers:
"""
    # If models came to consensus or i == MAX_ITER, output the answer
    # Else, go back to the small models and 
    for entry in top_k_outputs:
        judge_prompt += f"- {entry['output']}\n"
    if it == MAX_ITER - 1:

        judge_prompt += """
        Please analyze the candidate answers above and determine the most frequent final answer. 
        Do not repeat the candidate answers or the question. 
        Based solely on the candidate answers, provide one concise final answer along with a detailed explanation of your reasoning.
        Please do not provide the candidate answers on the answer.
        Your final answer should be in the following format:

        Final Answer: <your answer>
        Reasoning: <detailed explanation>
        """
    else:
        judge_prompt += """
        Please analyze the candidate answers above and decide whether there is consensus among them.
        If there is consensus, based solely on the candidate answers, provide one concise final answer along with a detailed explanation of your reasoning.
        Please do not provide the candidate answers on the answer.
        Output your final answer in the following format:

        Final Answer: <your answer>
        Reasoning: <detailed explanation>

        If there is no consensus, output in the following format:

        No Consensus. 
        Specific Info:
        <model_name_1> answered <majority_answer1>. Other models answered <majority_answer2> and <majority_answer3>. Let's check our answer again. Make sure to show your reasoning process.
        <model_name_2> answered <majority_answer2>. Other models answered <majority_answer1> and <majority_answer3>. Let's check our answer again. Make sure to show your reasoning process.
        <model_name_3> answered <majority_answer3> Other models answered <majority_answer1> and <majority_answer2>. Let's check our answer again. Make sure to show your reasoning process.

        Note that each model can output multiple answers using beam search. For the majority_answer, give the most popular answer from each model.
        If there is a tie, give any number out of the most popular candidates.
        Do not repeat the candidate answers or the question.
        You MUST follow the output format I provided you.
        """
    final_input_ids = final_tokenizer(judge_prompt, return_tensors="pt").input_ids.to(final_model.device)
    final_output = final_model.generate(
        final_input_ids,
        max_new_tokens=2048,
        num_beams=10,
        early_stopping=True,
        pad_token_id=final_tokenizer.pad_token_id
    )
    
    # Remove the prompt tokens from the output.
    prompt_length = final_input_ids.shape[-1]
    generated_tokens = final_output[0][prompt_length:]
    judge_response = final_tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # End the process if i == MAX_ITER
    # Or the models came to consensus
    if it == MAX_ITER - 1 or judge_response.strip().startswith("Final Answer:"):
        print(judge_response.strip())
        break
    
    # Make small models re-check their answers
    if judge_response.strip().startswith("No Consensus"):

        # Using regex to get the next prompt for the models
        pattern = r"(.+?)\s+answered\s+(.+?)\.\s+Other models answered\s+(.+?)\s+and\s+(.+?)\.\s+Let's check our answer again\.\s+Make sure to show your reasoning process\."
        feedback_lines = re.findall(pattern, judge_response)
        
        print("Extracted lines for debugging:")
        for line in feedback_lines:
            print(line)

        for i, feedback in enumerate(feedback_lines):
            current_prompt[i] = original_prompt + feedback

    it += 1

