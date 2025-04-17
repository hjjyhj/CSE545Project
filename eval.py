import os
os.environ['HF_HOME'] = "/scratch/eecs545w25_class_root/eecs545w25_class/cse545_reasoning/hf"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

import warnings
warnings.filterwarnings("ignore")

import argparse
import re
import json
import gc
from symeval import EvaluatorMathBatch

from datasets import load_dataset
from tqdm import tqdm
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from proj_src.init_eval.gsm8k.utils import download_url, load_jsonl, record_wrong_responses
import os

# Import configuration
from proj_src.utils.config import (
    MODEL_LIST, 
    JUDGE_MODEL_NAME,
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

# # Define the prompt template
# INSTRUCTION_TEMPLATE = """
# Solve the following math problem step by step. The last line of your response should be of the form: 'Answer: $ANSWER' (without quotes), where $ANSWER is the answer to the problem.
# The correct final answer is guaranteed to be an integer. Your answer should not have any units, just the number. Like: 'Answer: 3'.
# Remember to put your final answer on its own line after "Answer: ", and you DO NOT need to use a \\boxed command. The final answer should be as brief as possible.
# """

# def format_prompt(question, num_shots):
#     return INSTRUCTION_TEMPLATE + "\nQuestion:\n" + question + '\n\nResponse:\n'

def extract_answer_from_response(response):
    """
    Extracts the answer from the model's response.
    Assumes the answer follows the 'Answer:' keyword.
    """
    match = re.search(r'Final Answer: \s*(.*)', response)
    if match:
        return match.group(1).strip()
    return None

def math_equal(model_answer, true_answer, evaluator):
    """
    Compares two mathematical expressions for equivalence.
    """
    # print(f"[Model answer]: {model_answer}")
    # print(f"[True answer]: {true_answer}")
    if model_answer is None or true_answer is None:
        return False
    try:
        return evaluator.eq(model_answer, true_answer)
    except Exception as e:
        print(e)
        return model_answer == true_answer


def get_response_from_whole_system(original_prompt):
    """
    Main function that orchestrates the consensus-seeking process between multiple models.
    """
    # Initialize current prompts for each model
    current_prompts = [original_prompt] * len(MODEL_LIST)
    
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

            print(model_outputs[0]["output"])
            
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
            gc.collect()
            torch.cuda.empty_cache()
        

        # Load judge model
        judge_tokenizer, judge_model = load_model_and_tokenizer(JUDGE_MODEL_NAME)

        # Use judge to summarize answers
        summarized_answers = {}
        for answer in top_candidate_answers:
            assert answer["model"] not in summarized_answers # not gonna deal with multiple beams per model
            summary_prompt = create_summary_prompt(answer["output"])
            summarized_answers[answer["model"]] = get_judge_evaluation(judge_model, judge_tokenizer, summary_prompt)
            gc.collect()
            torch.cuda.empty_cache()

        # Create prompt for judge to evaluate consensus
        is_final_iteration = (iteration == MAX_ITERATIONS - 1)
        judge_prompt = create_consensus_prompt(
            original_prompt, 
            list(summarized_answers.values())
        )
        
        # Get judge's evaluation
        judge_response = get_judge_evaluation(judge_model, judge_tokenizer, judge_prompt)
        consensus,final_answer = extract_consensus_final(judge_response)
        print("=== judge response ===")
        print('\n'.join(list(summarized_answers.values())))
        print(judge_response)

        # Check if we have a final answer or need another iteration
        if is_final_iteration or consensus:
            print("=== Final evaluation: ===")
            # final response by the whole system, given by the judge model, ends with "Final Answer: XXX"
            final_judge_response = f"Final Answer: {final_answer}"
            break
            
        # If no consensus, update prompts for next iteration
        if not consensus:
            current_prompts = [update_prompts_from_feedback(
                original_prompt, 
                list(summarized_answers.values()),
            )] * len(current_prompts)

        # Free memory used for judge model
        del judge_model, judge_tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    
    print("=" * 80)
    print("Process completed")
    return final_judge_response


def evaluate_model_on_gsm8k(dataset, evaluator):
    """
    Evaluates the given model on the gsm8k dataset.
    """

    cur_count = 0
    correct_count = 0
    total_count = len(dataset)

    for sample in tqdm(dataset):
        try:
            question = sample['instruction']
            true_answer = sample['gt_answer']

            # Generate the model's response
            # prompt = format_prompt(question, num_shots)
            original_prompt = question

            final_response = get_response_from_whole_system(original_prompt)

            # Extract the model's answer
            final_answer = extract_answer_from_response(final_response)

            # Compare the model's answer to the true answer
            correct = False
            if final_answer and math_equal(final_answer, true_answer, evaluator):
                correct = True
                correct_count += 1
            cur_count += 1

            summary_str = (
                '=' * 50 + '\n\n' + \
                '### Question:\n' + question + '\n\n' + \
                f'### [MODEL_ANSWER]: {final_answer}\n' + \
                f'### [TRUE_ANSWER]: {true_answer}\n\n' + \
                f'%%%%% IS_CORRECT: {str(correct)} %%%%%\n\n' + \
                f'%%%%% Acc for now: {correct_count}/{cur_count}={correct_count/cur_count} %%%%%\n\n'
            )

            print(summary_str)
        except Exception as e:
            print(f"Exception Occurs: {e}")
            continue

    accuracy = correct_count / cur_count

    results = {
        "total": total_count,
        "correct": correct_count,
        "exception": total_count - cur_count,
        "accuracy": accuracy,
    }

    return results

if __name__ == '__main__':

    # eval on the pipeline
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument('--data_root', default="/scratch/eecs545w25_class_root/eecs545w25_class/cse545_reasoning/data")
    parser.add_argument('--save_dir', default='./proj_src/results')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Load the gsm8k dataset
    test_filepath = os.path.join(args.data_root, "gsm8k_test.jsonl")
    if not os.path.exists(test_filepath):
        download_url(
            "https://raw.githubusercontent.com/openai/"
            "grade-school-math/2909d34ef28520753df82a2234c357259d254aa8/"
            "grade_school_math/data/test.jsonl",
            args.data_root,
        )
        os.rename(os.path.join(args.data_root, "test.jsonl"), test_filepath)

    dataset = load_jsonl(test_filepath, instruction="question", output="answer")

    # math expression evaluator
    evaluator = EvaluatorMathBatch()

    results = evaluate_model_on_gsm8k(dataset, evaluator)
    
    print(results)
    with open(os.path.join(args.save_dir, f'scores.json'), 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


    
    # ==================================================




