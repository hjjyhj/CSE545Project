import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM

# Set Hugging Face cache directory
os.environ['HF_HOME'] = "/scratch/eecs545w25_class_root/eecs545w25_class/cse545_reasoning/hf"

# Configuration parameters
K_OUTPUTS_PER_MODEL = 2  # Number of top outputs to select from each model
MAX_ITERATIONS = 3       # Maximum number of consensus-seeking iterations

# Small language models (SLMs) to use for generating candidate answers
MODEL_LIST = [
    "stabilityai/stablelm-zephyr-3b",
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
]

# Judge model for evaluating consensus (larger model)
JUDGE_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# Input problem statement
ORIGINAL_PROMPT = "I have 3 apples, and my dad has 2 more apples than me. How many apples do we have in total?"


def load_model_and_tokenizer(model_name):
    """
    Load a model and its tokenizer with appropriate configurations based on model type.
    
    Args:
        model_name (str): The name or path of the model to load
        
    Returns:
        tuple: (tokenizer, model) The loaded tokenizer and model
    """
    print(f"Loading model: {model_name}")
    
    # Common parameters for all models
    model_kwargs = {
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "trust_remote_code": True
    }
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    
    # Ensure pad token is set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    return tokenizer, model


def generate_model_outputs(model, tokenizer, prompt, num_beams=5, max_new_tokens=512):
    """
    Generate outputs from a model using beam search.
    
    Args:
        model: The language model
        tokenizer: The tokenizer for the model
        prompt (str): The input prompt
        num_beams (int): Number of beams for beam search
        max_new_tokens (int): Maximum number of tokens to generate
        
    Returns:
        list: List of dictionaries containing model outputs
    """
    # Tokenize input with proper attention mask
    tokenized_input = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = tokenized_input.input_ids.to(model.device)
    attention_mask = tokenized_input.attention_mask.to(model.device)
    
    # Generate outputs using beam search
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        num_return_sequences=num_beams,
        early_stopping=True,
        pad_token_id=tokenizer.pad_token_id
    )
    
    # Process and collect outputs
    model_outputs = []
    for beam_idx, beam_output in enumerate(outputs, start=1):
        answer = tokenizer.decode(beam_output, skip_special_tokens=True).strip()
        output_entry = {
            "model": model.__class__.__name__,
            "beam": beam_idx,
            "output": answer
        }
        model_outputs.append(output_entry)
        print(f"Beam {beam_idx}: {answer}\n")
        
    return model_outputs


def create_consensus_prompt(original_prompt, candidate_answers, is_final_iteration):
    """
    Create a prompt for the judge model to evaluate consensus among candidate answers.
    
    Args:
        original_prompt (str): The original problem statement
        candidate_answers (list): List of candidate answer dictionaries
        is_final_iteration (bool): Whether this is the final iteration
        
    Returns:
        str: The prompt for the judge model
    """
    # Base prompt with original question and candidate answers
    prompt = f"""Below are candidate answers for the question: "{original_prompt}"
Candidate Answers:
"""
    # Add each candidate answer
    for entry in candidate_answers:
        prompt += f"- {entry['output']}\n"
    
    # Add instructions based on whether it's the final iteration
    if is_final_iteration:
        prompt += """
        Please analyze the candidate answers above and determine the most frequent final answer. 
        Do not repeat the candidate answers or the question. 
        Based solely on the candidate answers, provide one concise final answer along with a detailed explanation of your reasoning.
        Please do not provide the candidate answers on the answer.
        Your final answer should be in the following format:

        Final Answer: <your answer>
        Reasoning: <detailed explanation>
        """
    else:
        prompt += """
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
        You MUST follow the output format I provided to you.
        """
    
    return prompt


def main():
    """
    Main function that orchestrates the consensus-seeking process between multiple models.
    """
    # Initialize current prompts for each model
    current_prompts = [ORIGINAL_PROMPT] * len(MODEL_LIST)
    
    # Load judge model once at the beginning
    judge_tokenizer, judge_model = load_model_and_tokenizer(JUDGE_MODEL_NAME)
    
    # Iterate until max iterations or consensus is reached
    for iteration in range(MAX_ITERATIONS):
        print("=" * 80)
        print(f"Iteration {iteration + 1}: Generating candidate answers")
        print("=" * 80)
        
        top_candidate_answers = []
        
        # Generate answers from each SLM
        for model_idx, model_name in enumerate(MODEL_LIST):
            print("-" * 80)
            print(f"Generating answers from: {model_name}")
            print("-" * 80)
            
            # Load model and tokenizer
            tokenizer, model = load_model_and_tokenizer(model_name)
            
            # Generate and collect outputs
            model_outputs = generate_model_outputs(
                model, 
                tokenizer, 
                current_prompts[model_idx]
            )
            
            # Sort outputs by length (assuming longer answers might be more detailed)
            sorted_outputs = sorted(
                model_outputs, 
                key=lambda output: len(output["output"]), 
                reverse=True
            )
            
            # Keep top K outputs from this model
            top_candidate_answers.extend(sorted_outputs[:K_OUTPUTS_PER_MODEL])
            
            # Free up memory
            # Keep it if the memory is the issue
            # If you have enough memory, COMMENT IT to prevent the cost of loading the model again
            del model, tokenizer
            torch.cuda.empty_cache()
        
        # Create prompt for judge to evaluate consensus
        is_final_iteration = (iteration == MAX_ITERATIONS - 1)
        judge_prompt = create_consensus_prompt(
            ORIGINAL_PROMPT, 
            top_candidate_answers, 
            is_final_iteration
        )
        
        # Get judge's evaluation
        judge_response = get_judge_evaluation(
            judge_model, 
            judge_tokenizer, 
            judge_prompt
        )
        
        # Check if we have a final answer or need another iteration
        if is_final_iteration or judge_response.strip().startswith("Final Answer:"):
            print("Final evaluation:")
            print(judge_response.strip())
            break
            
        # If no consensus, update prompts for next iteration
        if judge_response.strip().startswith("No Consensus"):
            current_prompts = update_prompts_from_feedback(
                ORIGINAL_PROMPT, 
                judge_response
            )
    
    print("=" * 80)
    print("Process completed")


def get_judge_evaluation(judge_model, judge_tokenizer, judge_prompt):
    """
    Get evaluation from the judge model.
    
    Args:
        judge_model: The judge language model
        judge_tokenizer: The tokenizer for the judge model
        judge_prompt (str): The prompt for the judge
        
    Returns:
        str: The judge's response
    """
    # Tokenize judge prompt
    tokenized_input = judge_tokenizer(judge_prompt, return_tensors="pt", padding=True)
    input_ids = tokenized_input.input_ids.to(judge_model.device)
    attention_mask = tokenized_input.attention_mask.to(judge_model.device)
    
    # Generate judge's evaluation
    output = judge_model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=2048,
        num_beams=10,
        early_stopping=True,
        pad_token_id=judge_tokenizer.pad_token_id
    )
    
    # Extract only the generated part (excluding the prompt)
    prompt_length = input_ids.shape[-1]
    generated_tokens = output[0][prompt_length:]
    
    return judge_tokenizer.decode(generated_tokens, skip_special_tokens=True)


def update_prompts_from_feedback(original_prompt, judge_response):
    """
    Extract feedback from judge response and update prompts for next iteration.
    
    Args:
        original_prompt (str): The original problem statement
        judge_response (str): The judge's response with feedback
        
    Returns:
        list: Updated prompts for each model
    """
    # Extract feedback using regex
    pattern = r"(.+?)\s+answered\s+(.+?)\.\s+Other models answered\s+(.+?)\s+and\s+(.+?)\.\s+Let's check our answer again\.\s+Make sure to show your reasoning process\."
    feedback_lines = re.findall(pattern, judge_response)
    
    print("Feedback extracted for next iteration:")
    for line in feedback_lines:
        print(line)
    
    # Create new prompts with feedback
    updated_prompts = []
    for feedback in feedback_lines:
        new_prompt = original_prompt + " ".join(feedback)
        updated_prompts.append(new_prompt)
    
    # If we didn't get enough feedback for all models, duplicate the last one
    while len(updated_prompts) < len(MODEL_LIST):
        updated_prompts.append(updated_prompts[-1] if updated_prompts else original_prompt)
    
    return updated_prompts


if __name__ == "__main__":
    main()

