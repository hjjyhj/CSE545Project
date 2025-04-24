import os
os.environ['HF_HOME'] = "/scratch/eecs545w25_class_root/eecs545w25_class/cse545_reasoning/hf"

import warnings
warnings.filterwarnings("ignore")

import argparse
import re
import json
from symeval import EvaluatorMathBatch
from resumehelpers import build_json

from datasets import load_dataset
from tqdm import tqdm
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from proj_src.init_eval.gsm8k.utils import download_url, load_jsonl, record_wrong_responses
import torch
import os
from openai import OpenAI

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
    create_consensus_prompt, 
    update_prompts_from_feedback
)

# # Define the prompt template
# INSTRUCTION_TEMPLATE = """
# Solve the following math problem step by step. The last line of your response should be of the form: 'Answer: $ANSWER' (without quotes), where $ANSWER is the answer to the problem.
# The correct final answer is guaranteed to be an integer. Your answer should not have any units, just the number. Like: 'Answer: 3'.
# Remember to put your final answer on its own line after "Answer: ", and you DO NOT need to use a \\boxed command. The final answer should be as brief as possible.
# """

INSTRUCTION_TEMPLATE = """
Solve the following SAT math problem step by step. The last line of your response should be of the form: 'Answer: $ANSWER' (without quotes), where $ANSWER is the answer to the problem.
The correct final answer is guaranteed to be an one of the alphabet choices. Your answer should not have any units, just the letter option. Like: 'Answer: B'.
Remember to put your final answer on its own line after "Answer: ", and you DO NOT need to use a \\boxed command. The final answer should be as brief as possible.
"""

def format_prompt(question, num_shots):
     return INSTRUCTION_TEMPLATE + "\nQuestion:\n" + question + '\n\nResponse:\n'




def extract_answer_from_response(response):
    """
    Extracts a valid multiple-choice letter (A–D) from the model's response.
    Prioritizes \boxed{X} format if present.
    """
    # Case 1: In the case of an answer written as '\boxed{C}-like answer'
    match = re.search(r'\\boxed\{\s*([A-Da-d])\s*\}', response)
    if match:
        return match.group(1).upper()

    # Case 2: In the case of something like C, Answer: (C), Answer: **C**, etc.
    match = re.search(
        r'Answer:\s*(?:\\boxed\s*\{)?(?:\\text\s*\{)?\**\(?\s*([A-Da-d])\s*\)?\**\}?',
        response,
        re.IGNORECASE
    )
    if match:
        return match.group(1).upper()

    # Case 3: As a fallback, we look for a lone capital letter near the end of the response.
    match = re.search(r'\b([A-Da-d])\b[\s\)\}]*$', response.strip())
    if match:
        return match.group(1).upper()

    # No letter, we assume model failed, and no response.
    return None

def check_label(model_answer, true_answer):
    """
    Compares the model's predicted letter choice with the correct letter answer.
    """
    # Ensures it must exist
    if model_answer is None or true_answer is None:
        return False
    # Only returns true if they are both equal, else false
    return model_answer.strip().upper() == true_answer.strip().upper()

def get_response_from_whole_system(original_prompt):
    """
    Main function that orchestrates the consensus-seeking process between multiple models.
    """
    # Initialize current prompts for each model
    current_prompts = [original_prompt] * len(MODEL_LIST)
    
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
            torch.cuda.empty_cache()
        
        # Create prompt for judge to evaluate consensus
        is_final_iteration = (iteration == MAX_ITERATIONS - 1)
        judge_prompt = create_consensus_prompt(
            original_prompt, 
            top_candidate_answers, 
            is_final_iteration
        )
        
        # Get judge's evaluation
        judge_response = get_judge_evaluation(
            judge_model, 
            judge_prompt
        )
        print("=== judge response ===")
        print(judge_response)

        # Check if we have a final answer or need another iteration
        if is_final_iteration or judge_response.strip().startswith("Final Conclusion:"):
            print("=== Final evaluation: ===")
            # final response by the whole system, given by the judge model, ends with "Final Answer: XXX"
            final_judge_response = judge_response.strip()   
            break
            
        # If no consensus, update prompts for next iteration
        if "No Consensus" in judge_response:
            current_prompts = update_prompts_from_feedback(
                original_prompt,
                judge_response
            )
            # print(current_prompts[0])
    
    print("=" * 80)
    print("Process completed")
    return final_judge_response

def evaluate_model_on_agieval(dataset, progress, start_index):
    cur_count = start_index
    correct_count = progress["correct"]
    total_count = progress["total"]

    # Sanity checks to make sure progress is consistent
    assert total_count - cur_count == progress["exception"], \
    f"Exception mismatch: expected {progress['exception']}, got {total_count - cur_count}"

    assert (cur_count > 0 and abs(correct_count / cur_count - progress["accuracy"]) < 1e-6) or \
       (cur_count == 0 and progress["accuracy"] == 0), \
    f"Accuracy mismatch: expected {progress['accuracy']}, got {correct_count / cur_count if cur_count else 0}"
    

    for sample in tqdm(dataset):
        try:
            question = sample['instruction']
            true_answer = sample['output']

            # Generate the model's response
            # prompt = format_prompt(question, num_shots)
            original_prompt = format_prompt(question,0)
 
            final_response = get_response_from_whole_system(original_prompt)

            # Extract the model's answer
            final_answer = extract_answer_from_response(final_response)


            correct = False
            if check_label(final_answer, true_answer):
                correct = True
                correct_count += 1
            cur_count += 1

            # Save intermediate results every 22 examples
            if cur_count % 22 == 0:
                interim_results = {
                    "total": total_count,
                    "correct": correct_count,
                    "exception": total_count - cur_count,
                    "accuracy": correct_count / cur_count if cur_count else 0,
                }
                with open(os.path.join(args.save_dir, 'scores.json'), 'w') as f:
                    json.dump(interim_results, f, ensure_ascii=False, indent=2)


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

    if cur_count == 0:
        accuracy=0
    else:
     accuracy = correct_count / cur_count

    results = {
        "total": total_count,
        "correct": correct_count,
        "exception": total_count - cur_count,
        "accuracy": accuracy,
    }
    return results


def prepare_agi_eval_dataset(args):
    """Downloads and loads the AGIEval SAT Math dataset if it doesn't already exist.
     We also append the multiple-choice options to the question text"""
    test_filepath = os.path.join(args.data_root, "sat-math.jsonl")
    if not os.path.exists(test_filepath):
        download_url(
            "https://raw.githubusercontent.com/ruixiangcui/AGIEval/main/data/v1_1/sat-math.jsonl",
            args.data_root,
        )
        downloaded_file = os.path.join(args.data_root, "sat-math.jsonl")
        if os.path.exists(downloaded_file):
            os.rename(downloaded_file, test_filepath)

    dataset = []
    # I append the options to the end of each question
    with open(test_filepath, "r", encoding="utf-8") as f:
        for line in f:
                item = json.loads(line)
                question = item["question"]
                options = "\n".join(item["options"])
                full_question = f"{question}\n{options}"
                dataset.append({
                "instruction": full_question, 
                "output": item["label"]
            })
    return dataset



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
            original_prompt =format_prompt(question)

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
    parser.add_argument("--max_new_tokens", type=int, default=1600)
    parser.add_argument('--data_root', default="/scratch/eecs545w25_class_root/eecs545w25_class/cse545_reasoning/data")
    parser.add_argument('--save_dir', default='./proj_src/results')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Load the agieval dataset and append the option choices to the question itself
    progress = build_json(args)
    dataset = prepare_agi_eval_dataset(args=args)

    # Manually set where the index should begin at
    start_index = start_index = progress["total"] - progress["exception"]
    dataset = dataset[start_index:]
    print(f"Loaded dataset with {len(dataset)} samples.")
    results = evaluate_model_on_agieval(dataset=dataset, progress=progress, start_index=start_index)
    
    print(results)
    with open(os.path.join(args.save_dir, f'scores.json'), 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)    
    # ==================================================




