import torch
import os
from openai import OpenAI

# Import configuration
from proj_src.utils.config import (
    MODEL_LIST, 
    JUDGE_MODEL_NAME, 
    ORIGINAL_PROMPT, 
    K_OUTPUTS_PER_MODEL, 
    MAX_ITERATIONS
)

# Import utility functions
from proj_src.utils.model_utils import (
    load_model_and_tokenizer, 
    generate_model_outputs, 
    get_judge_evaluation
)
from proj_src.utils.prompt_utils import (
    create_consensus_prompt, 
    update_prompts_from_feedback
)


def main():
    """
    Main function that orchestrates the consensus-seeking process between multiple models.
    """
    # Initialize current prompts for each model
    current_prompts = [ORIGINAL_PROMPT] * len(MODEL_LIST)
    
    # Load judge model once at the beginning
    # judge_tokenizer, judge_model = load_model_and_tokenizer(JUDGE_MODEL_NAME)
    judge_model = OpenAI(api_key="sk-4673fd7bbbd445f380b30ab883a43b05", base_url="https://api.deepseek.com")
    
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
            judge_prompt
        )

        # Check if we have a final answer or need another iteration
        if is_final_iteration or judge_response.strip().startswith("Final Answer:"):
            print("Final evaluation:")
            print(judge_response.strip())
            break
            
        # If no consensus, update prompts for next iteration
        if "No Consensus" in judge_response:
            
            current_prompts = update_prompts_from_feedback(
                ORIGINAL_PROMPT, 
                judge_response
            )
            print(current_prompts[0])
    
    print("=" * 80)
    print("Process completed")


if __name__ == "__main__":
    main()
