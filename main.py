import torch
import os
import gc
from openai import OpenAI

os.environ['HF_HOME'] = "/scratch/eecs545w25_class_root/eecs545w25_class/cse545_reasoning/hf"

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
    create_summary_prompt, 
    create_consensus_prompt, 
    update_prompts_from_feedback
)
from proj_src.utils.output_utils import (
    extract_consensus_final
)


def main():
    """
    Main function that orchestrates the consensus-seeking process between multiple models.
    """
    # Initialize current prompts for each model
    current_prompts = [ORIGINAL_PROMPT] * len(MODEL_LIST)
    
    # Load judge model once at the beginning
    judge_tokenizer, judge_model = None, None # load later
    # judge_model = OpenAI(api_key="sk-4673fd7bbbd445f380b30ab883a43b05", base_url="https://api.deepseek.com")
    
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
            
            if model_name != JUDGE_MODEL_NAME:
                # Load model and tokenizer
                tokenizer, model = load_model_and_tokenizer(model_name)
            else:
                if judge_model == None:
                    judge_tokenizer, judge_model = load_model_and_tokenizer(JUDGE_MODEL_NAME)
                tokenizer, model = judge_tokenizer, judge_model
            
            # Generate and collect outputs
            try:
                model_outputs = generate_model_outputs(
                    model, 
                    tokenizer, 
                    current_prompts[model_idx]
                )
            except RuntimeError as e:
                print("Error, freeing judge model and re-running")
                print(e)
                if "out of memory" not in str(e).lower():
                    raise
                assert model_name != JUDGE_MODEL_NAME
                # try again without judge model loaded
                del judge_model, judge_tokenizer
                # del judge_model, judge_tokenizer, tokenizer, model
                judge_tokenizer, judge_model = None, None
                gc.collect()
                torch.cuda.empty_cache()
                a
                # tokenizer, model = load_model_and_tokenizer(model_name)
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
            if model_name != JUDGE_MODEL_NAME:
                del model, tokenizer
                gc.collect()
                torch.cuda.empty_cache()
        

        # Load judge model if not loaded
        if judge_model == None:
            judge_tokenizer, judge_model = load_model_and_tokenizer(JUDGE_MODEL_NAME)

        # Use judge to summarize answers
        summarized_answers = {}
        for answer in top_candidate_answers:
            assert answer["model"] not in summarized_answers # not gonna deal with multiple beams per model
            summary_prompt = create_summary_prompt(ORIGINAL_PROMPT, answer["output"])
            summarized_answers[answer["model"]] = get_judge_evaluation(judge_model, judge_tokenizer, summary_prompt)

        # Create prompt for judge to evaluate consensus
        is_final_iteration = (iteration == MAX_ITERATIONS - 1)
        judge_prompt = create_consensus_prompt(
            ORIGINAL_PROMPT, 
            list(summarized_answers.values())
        )
        
        # Get judge's evaluation
        judge_response = get_judge_evaluation(judge_model, judge_tokenizer, judge_prompt)
        print(judge_response)
        consensus,final_answer = extract_consensus_final(judge_response)

        # Check if we have a final answer or need another iteration
        if is_final_iteration or consensus:
            print("Final evaluation:")
            print(final_answer)
            break
            
        # If no consensus, update prompts for next iteration
        if not consensus:
            current_prompts = [update_prompts_from_feedback(
                ORIGINAL_PROMPT, 
                list(summarized_answers.values()),
            )] * len(current_prompts)
            print(current_prompts[0])
    
    print("=" * 80)
    print("Process completed")


if __name__ == "__main__":
    main()

