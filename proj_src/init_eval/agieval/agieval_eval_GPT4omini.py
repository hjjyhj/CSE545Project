"""
Author: Larnell Moore
Purpose: This script is used to evaluate the performance of GPT4o mini on the AGIEval sat math dataset.
"""

import os
os.environ['HF_HOME'] = "/scratch/eecs545w25_class_root/eecs545w25_class/cse545_reasoning/hf"
os.environ["PYTORCH_USE_SDPA"] = "0"
from dotenv import load_dotenv
import argparse
import re
import json
from tqdm import tqdm
from openai import OpenAI
from utils import download_url, load_jsonl, record_wrong_responses

from proj_src.init_eval.agieval.helpers import (
 determine_reasoning_process_agieval,
 build_json,
 GPT4oMiniResponse,
 prepare_agi_eval_dataset,
 extract_answer_from_response,
 check_label
)

# As recommended by the professor during our poster presentation, we can give a model more chances to respond than once given feedback
# to make the comparsions more fair. However, by default, we set this to one due to API costs.
MAX_ITERATIONS = 1

# Define the prompt template
INSTRUCTION_TEMPLATE = """
Solve the following SAT math problem step by step. The last line of your response should be of the form: 'Answer: $ANSWER' (without quotes), where $ANSWER is the answer to the problem.
The correct final answer is guaranteed to be an one of the alphabet choices. Your answer should not have any units, just the letter option. Like: 'Answer: B'.
Remember to put your final answer on its own line after "Answer: ", and you DO NOT need to use a \\boxed command. The final answer should be as brief as possible.
"""

# max: 3
few_shot_examples = [
    "Question:\nBetty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?\nOptions:\nA) $10\nB) $5\nC) $15\nD) $0\n\nResponse:\nIn the beginning, Betty has 100 / 2 = 50. Her parents gave her $15. Her grandparents gave her 15 * 2 = 30. Total money Betty has: 50 + 15 + 30 = 95. She still needs 100 - 95 = 5 more.\nAnswer: B",
    "Question:\nLisa, Jack, and Tommy earned $60 from washing cars all week. However, half of the $60 was earned by Lisa. Tommy earned half of what Lisa earned. How much more money did Lisa earn than Tommy?\nOptions:\nA) $30\nB) $10\nC) $15\nD) $20\n\nResponse:\nLisa earned 60 * 1/2 = 30. Tommy earned 30 * 1/2 = 15. So, Lisa earned 30 - 15 = 15 more than Tommy.\nAnswer: C",
    "Question:\nGretchen has 110 coins. There are 30 more gold coins than silver coins. How many gold coins does Gretchen have?\nOptions:\nA) 50\nB) 60\nC) 70\nD) 80\n\nResponse:\nLet x be the number of silver coins. Then gold coins = x + 30. So, x + (x + 30) = 110 → 2x + 30 = 110 → 2x = 80 → x = 40. Gold coins = 40 + 30 = 70.\nAnswer: C"
]

# Can either be 0 shot or 3 shot at max in this program. 
def format_prompt(question, num_shots):
    """
    If zero shot, append instuction template alongside question.
    If few shot, give the model some examples of how to answer.    
    """
    if num_shots == 0:
        return INSTRUCTION_TEMPLATE + "\nQuestion:\n" + question + '\n\nResponse:\n'
    prompt = INSTRUCTION_TEMPLATE + '\nExamples:\n'
    for example in few_shot_examples:
        prompt += (example + '\n')
    return prompt + "\nQuestion:\n" + question + '\n\nResponse:\n'


def get_reasoning_path(original_prompt):
    """
    Main function that orchestrates the feedback process between for model, prompted three times.
    """

    load_dotenv()
    judge_model = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
    prev_attempts = {}
    final_judge_response = ""

    for iteration in range(MAX_ITERATIONS):
        print("=" * 80)
        print(f"Iteration {iteration + 1}: Generating reasoning path {iteration + 1} of {MAX_ITERATIONS}")
        print("=" * 80)

        # Build current prompt with previous attempts included
        current_prompt = original_prompt
        if iteration > 0:
            last_attempt = prev_attempts.get(iteration - 1, "")
            current_prompt += f"Possible reasoning path discovered by a student(May be correct or false):{last_attempt}\n"
            print("=== Model response ===")
            print("Augmented Prompt:")
            print(current_prompt)
        else:
            print("=== Model response ===")
            print("Original Prompt:")
            print(current_prompt)

        # Generate and collect outputs
        model_output = GPT4oMiniResponse(
                API_KEY=os.getenv("OPENAI_API_KEY"), 
                model="gpt-4o-mini-2024-07-18",
                prompt=current_prompt,
            )
        output_text = model_output
        print("=== Model response ===")
        print(output_text)
            
        is_final_iteration = (iteration == MAX_ITERATIONS - 1)

        if not is_final_iteration:
            prev_attempts[iteration] =  determine_reasoning_process_agieval(output=output_text, judge_model=judge_model,num=1)
            judge_response = prev_attempts[iteration]
        else:
            prev_attempts[iteration] =  determine_reasoning_process_agieval(output=output_text, judge_model=judge_model,num=1)
            judge_response = prev_attempts[iteration]
            
        print("=== judge response ===")
        print(judge_response[0].strip())

        if is_final_iteration:
            print("=== Final evaluation: ===")
            final_judge_response = judge_response[0].strip()
            break

    print("=" * 80)
    print("Process completed")
    return final_judge_response

def evaluate_model_on_agieval(dataset, progress, start_index):
    cur_count = start_index
    correct_count = progress["correct"]
    total_count = progress["total"]
    
    for sample in tqdm(dataset):
        try:
         while True:
            print(f"Evaluating sample {cur_count + 1}...")
            question = sample['instruction']
            true_answer = sample['output']

            # Generate the model's response
            # prompt = format_prompt(question, num_shots)
            original_prompt = format_prompt(question,0)
 
            final_response = get_reasoning_path(original_prompt)

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
            break
        except Exception as e:
           # Instead of skipping failed results, we force GPT4o to retry the same sample.
           # This is to ensure that we get a response for every sample.
           print(f"Exception Occurs: {e}")
           print("Retrying same sample...")

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



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B')
    parser.add_argument("--run_all_models", action='store_true', help='run all models in the list')
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument('--num_shots', type=int, choices=[0, 3], default=0)
    parser.add_argument('--record_wrong', action='store_true', help='record the answers that are wrong')
    parser.add_argument('--data_root', default="/scratch/eecs545w25_class_root/eecs545w25_class/cse545_reasoning/data")
    parser.add_argument('--output_dir', default='./output')
    parser.add_argument('--save_dir', default='./proj_src/results')
    
    args = parser.parse_args()

    # Only do one model at a time, do not run all models or progam may be unstable.  Keep this list length of 1.
    model_list = [
       "GPT4-o-Mini-2024-07-18",
    ]   

    os.makedirs(args.output_dir, exist_ok=True)

    # Set to variable
    dataset = prepare_agi_eval_dataset(args=args)
    print( f"Loaded dataset with {len(dataset)} samples.")
    progress = build_json(args, dataset)
    print("Total:", progress["total"], "Correct", progress["correct"], "Exception", progress["exception"])
   
    generate_kwargs = dict(max_new_tokens=args.max_new_tokens, do_sample=False) # for deterministic generation

    if args.run_all_models:
        for full_model_name in model_list:
            save_name = os.path.join(args.output_dir, full_model_name.split('/')[-1])
            os.makedirs(save_name, exist_ok=True)
            print(f'Evaluating on {full_model_name}')

            # Update args.save_dir to this model-specific folder
            args.save_dir = save_name

            # Manually set where the index should begin at
            # The total samples minus the number of samples we have left to evaluate.
            start_index = progress["total"] - progress["exception"]
            print(f"Start index: {start_index}")
            dataset = dataset[start_index:]
            print(f"Loaded dataset with {len(dataset)} samples.")

            with open(os.path.join(save_name, f'responses_{args.max_new_tokens}_{args.num_shots}shot.txt'), 'w') as f:
                results = evaluate_model_on_agieval(dataset=dataset, start_index=start_index, progress=progress)
            print(results)
            with open(os.path.join(save_name, f'scores_{args.max_new_tokens}_{args.num_shots}shot.json'), 'w') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

        


