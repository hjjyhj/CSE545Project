import torch
import os

# Import configuration
from proj_src.utils.config import (
    MODEL_LIST, 
    JUDGE_MODEL_NAME, 
    ORIGINAL_PROMPT, 
    K_OUTPUTS_PER_MODEL, 
    MAX_ITERATIONS,
    USE_OPENROUTER,
    OPENROUTER_MODEL_NAME,
    NUM_OPENROUTER_MODELS
)

# Import environment variables
from proj_src.utils.env_utils import load_environment_variables

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

# Import OpenRouter utilities
from proj_src.utils.openrouter_utils import (
    load_openrouter_model,
    generate_openrouter_outputs
)


def main():
    """
    Main function that orchestrates the consensus-seeking process between multiple models.
    """
    # Load environment variables if using OpenRouter
    if USE_OPENROUTER:
        env_vars = load_environment_variables()
        
    # Initialize current prompts for each model
    if USE_OPENROUTER:
        current_prompts = [ORIGINAL_PROMPT] * NUM_OPENROUTER_MODELS
    else:
        current_prompts = [ORIGINAL_PROMPT] * len(MODEL_LIST)
    
    # Load judge model once at the beginning
    if USE_OPENROUTER:
        judge_model, judge_tokenizer = load_openrouter_model(
            env_vars["openrouter_api_key"],
            env_vars["site_url"],
            env_vars["site_name"],
            OPENROUTER_MODEL_NAME
        )
    else:
        judge_tokenizer, judge_model = load_model_and_tokenizer(JUDGE_MODEL_NAME)
    
    # Iterate until max iterations or consensus is reached
    for iteration in range(MAX_ITERATIONS):
        print("=" * 80)
        print(f"Iteration {iteration + 1}: Generating candidate answers")
        print("=" * 80)
        
        top_candidate_answers = []
        
        # Generate answers from each model
        if USE_OPENROUTER:
            # For OpenRouter, we use the same model multiple times to simulate different models
            for model_idx in range(NUM_OPENROUTER_MODELS):
                print("-" * 80)
                print(f"Generating answers from OpenRouter model instance {model_idx+1}")
                print("-" * 80)
                
                # Load model
                model, _ = load_openrouter_model(
                    env_vars["openrouter_api_key"],
                    env_vars["site_url"],
                    env_vars["site_name"],
                    OPENROUTER_MODEL_NAME
                )
                
                # Generate and collect outputs
                model_outputs = generate_openrouter_outputs(
                    model, 
                    None,  # No tokenizer needed for OpenRouter
                    current_prompts[model_idx],
                    num_outputs=K_OUTPUTS_PER_MODEL
                )
                
                # Add to candidates
                top_candidate_answers.extend(model_outputs)
                
                # No need to free memory for OpenRouter models
        else:
            # Original HuggingFace models approach
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
        if USE_OPENROUTER:
            judge_response = judge_model.generate_response(judge_prompt)
        else:
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


if __name__ == "__main__":
    main()

