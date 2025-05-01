import os
os.environ['HF_HOME'] = "/scratch/eecs545w25_class_root/eecs545w25_class/cse545_reasoning/hf"

import argparse
import re
import json
from symeval import EvaluatorMathBatch
from dotenv import load_dotenv

from datasets import load_dataset
from tqdm import tqdm
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from utils import download_url, load_jsonl, record_wrong_responses
from openai import OpenAI

MAX_ITERATIONS = 3

# Define the prompt template
INSTRUCTION_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form: 'Answer: $ANSWER' (without quotes), where $ANSWER is the answer to the problem.
The correct final answer is guaranteed to be an integer. Your answer should not have any units, just the number. Like: 'Answer: 3'.
Remember to put your final answer on its own line after "Answer: ", and you DO NOT need to use a \\boxed command. The final answer should be as brief as possible.
"""

# max: 3
few_shot_examples = [
    "Question:\n Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?\n\nResponse:\n In the beginning, Betty has only 100 / 2 = 50. Betty's grandparents gave her 15 * 2 = 30. This means, Betty needs 100 - 50 - 30 - 15 = 5 more.\nAnswer: 5",
    "Question:\n Lisa, Jack, and Tommy earned $60 from washing cars all week. However, half of the $60 was earned by Lisa. Tommy earned half of what Lisa earned. How much more money did Lisa earn than Tommy?\n\nResponse:\n Lisa earned 60 * 1/2 = 30. Tommy earned 30 * 1/2 = 15. Lisa earned 30 - 15 = 15 more than Tommy.\nAnswer: 15",
    "Question:\n Gretchen has 110 coins. There are 30 more gold coins than silver coins. How many gold coins does Gretchen have?\n\nResponse\n: Let x be the number of silver coins Gretchen has Gretchen has x+30 gold coins. x+x+30=110 2*x=80 x=<<40=40>>40 Gretchen has 40+30=70 gold coins\nAnswer: 70."
]

def format_prompt(question, num_shots):
    if num_shots == 0:
        return INSTRUCTION_TEMPLATE + "\nQuestion:\n" + question + '\n\nResponse:\n'
    prompt = INSTRUCTION_TEMPLATE + '\nExamples:\n'
    for example in few_shot_examples:
        prompt += (example + '\n')
    return prompt + "\nQuestion:\n" + question + '\n\nResponse:\n'

def build_json(args, dataset):
    """Resumes progress if it exists scores.json exist, else start fresh
    Args:
        args: command line arguments
        dataset: the dataset to evaluate
        total: the total number of samples in the dataset
        correct: the number of correct samples
        exception: the number of remaining samples
        accuracy: the accuracy of the model
    """
    resume_path = os.path.join(args.save_dir, args.model_name.split('/')[-1], 'scores.json')
    os.makedirs(os.path.dirname(resume_path), exist_ok=True)
    if os.path.exists(resume_path):
        with open(resume_path, 'r') as f:
            resume_data = json.load(f)
        print(f"[RESUME] Loaded existing progress from {resume_path}")
    else:
        resume_data = {
            "total": len(dataset),
            "correct": 0,
            "exception": len(dataset),
            "accuracy": 0.0
        }
        print(f"[NEW RUN] No previous scores.json found. Starting fresh.")
    return resume_data

def GPT4oMiniResponse(API_KEY, model, prompt):
    
    student = OpenAI(api_key=API_KEY)
    response = student.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        max_tokens=2048,
        temperature=0.0,
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ],
        stream=False
    )
    return response.choices[0].message.content

def determine_reasoning_process(output, judge_model, num):
    """
    Summarizes the reasoning behind why the judge model chose a particular answer.
    """
    # Step 1: Ask for the final answer choice only
    answer_prompt = f"""
Read the student's work below carefully. Identify and extract only the FINAL numerical ANSWER they chose (Ex. 0.25, 2x+5, etc.).
Output ONLY the single ANSWER (0.25, 2x+2, 1/2, 5, etc), with no extra text. If unclear, choose the most recent piece of work they chose.

Student's work:
{output}
"""
    answer_response = judge_model.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": answer_prompt},
        ],
        stream=False
    )
    answer = answer_response.choices[0].message.content.strip()

    # Step 2: Ask for the reasoning separately
    reasoning_prompt = f"""
Read the student's work below carefully. Summarize their reasoning process in 2-4 sentences.
Be concise and clear. Do not mention their final answer..

Student's work:
{output}
"""
    reasoning_response = judge_model.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": reasoning_prompt},
        ],
        stream=False
    )
    reasoning = reasoning_response.choices[0].message.content.strip()

    # Step 3: Format it yourself
    reasoning_text = f"<model_{num}> answered {answer}. Here is the reasoning process for this answer: {reasoning}"
    reasoning_text += f"\nAnswer: {answer}"  

    return reasoning_text, answer

def extract_answer_from_response(response):
    """
    Extracts the answer from the model's response.
    Assumes the answer follows the 'Answer:' keyword.
    """
    match = re.search(r'Answer:\s*(.*)', response)
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


def get_reasoning_path_from_whole_system(original_prompt):
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
            current_prompt += f"Possible reasoning path discovered by a student. Propose a new solution if you think it is wrong:{last_attempt}\n"
            print("=== Model response ===")
            print("Augmented Prompt:")
            print(current_prompt)
        else:
            print("=== Model response ===")
            print("Original Prompt:")
            print(current_prompt)

        # Generate and collect outputs
        load_dotenv()
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
            prev_attempts[iteration] =  determine_reasoning_process(output=output_text, judge_model=judge_model,num=1)
            judge_response = prev_attempts[iteration]
        else:
            prev_attempts[iteration] =  determine_reasoning_process(output=output_text, judge_model=judge_model,num=1)
            judge_response = prev_attempts[iteration]
            

        print("=== judge response ===")
        print(judge_response)

        if is_final_iteration:
            print("=== Final evaluation: ===")
            final_judge_response = judge_response[0].strip()
            break

    print("=" * 80)
    print("Process completed")
    return final_judge_response


def evaluate_model_on_gsm8k(dataset, evaluator, num_shots, generate_kwargs, record_wrong, f):
    """
    Evaluates the given model on the gsm8k dataset.
    """

    cur_count = 0
    correct_count = 0
    trunc_wrong_count = 0
    total_count = len(dataset)

    wrong_responses = []

    for sample in tqdm(dataset):
      try:
       while True:
        question = sample['instruction']
        true_answer = sample['gt_answer']

        # Generate the model's response
        prompt = format_prompt(question, 0)
        response = get_reasoning_path_from_whole_system(prompt)
        model_answer = extract_answer_from_response(response)

        print(f"Model answer: {model_answer}")
        print(f"True answer: {true_answer}")

        # Compare the model's answer to the true answer
        correct = False
        if model_answer and math_equal(model_answer, true_answer, evaluator):
            correct = True
            correct_count += 1
        else:
            if not model_answer: # wrong because the model didn't output a formatted answer, most likely to be truncated
                trunc_wrong_count += 1

            if record_wrong:
                wrong_response = {
                    "idx": str(cur_count),
                    "question": question,
                    "answer": sample["output"],
                    "model_response": response,
                    "gt_answer": str(true_answer),
                    "model_answer": str(model_answer)
                }
                wrong_responses.append(wrong_response)

        cur_count += 1

        summary_str = (
            '=' * 50 + '\n\n' + \
            '### Question:\n' + question + '\n\n' + \
            '### Model Response:\n' + response + '\n\n' + \
            f'### [MODEL_ANSWER]: {model_answer}\n' + \
            f'### [TRUE_ANSWER]: {true_answer}\n\n' + \
            f'%%%%% IS_CORRECT: {str(correct)} %%%%%\n\n' + \
            f'%%%%% Acc for now: {correct_count/cur_count} %%%%%\n\n'
        )
        print(summary_str)
        f.write(summary_str)
        break
      except Exception as e:
         print(f"Exception Occurs: {e}")
         print("Retrying same sample...")

    if cur_count == 0:
     accuracy = 0
    else:
     accuracy = correct_count / cur_count

    if total_count - correct_count == 0:
     trunc_wrong_ratio = 0
    else:
     trunc_wrong_ratio = trunc_wrong_count / (total_count - correct_count)

    results = {
        "total": total_count,
        "correct": correct_count,
        "trunc_wrong": trunc_wrong_count,
        "trunc_wrong_ratio": trunc_wrong_ratio,
        "accuracy": accuracy,
        "wrong_responses": wrong_responses
    }

    return results

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_all_models", action='store_true', help='run all models in the list')
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument('--num_shots', type=int, choices=[0, 3], default=0)
    parser.add_argument('--record_wrong', action='store_true', help='record the answers that are wrong')
    parser.add_argument('--data_root', default="/scratch/eecs545w25_class_root/eecs545w25_class/cse545_reasoning/data")
    parser.add_argument('--output_dir', default='./output')
    
    args = parser.parse_args()


    model_list = [
        "GPT4oMini2",
    ]   

    os.makedirs(args.output_dir, exist_ok=True)

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

    # generate_kwargs = dict(max_new_tokens=args.max_new_tokens, top_p=0.95, temperature=0.8)
    generate_kwargs = dict(max_new_tokens=args.max_new_tokens, do_sample=False) # for deterministic generation

    if args.run_all_models:
        for full_model_name in model_list:
            save_name = os.path.join(args.output_dir, full_model_name.split('/')[-1])
            os.makedirs(save_name, exist_ok=True)
            print(f'Evaluating on {full_model_name}')

            with open(os.path.join(save_name, f'responses_{args.max_new_tokens}_{args.num_shots}shot.txt'), 'w') as f:
                results = evaluate_model_on_gsm8k(dataset=dataset, evaluator=evaluator, num_shots=args.num_shots, generate_kwargs=generate_kwargs, record_wrong=record_wrong_responses, f=f)
            
            wrong_responses = results["wrong_responses"]
            del results["wrong_responses"]

            print(results)
            with open(os.path.join(save_name, f'scores_{args.max_new_tokens}_{args.num_shots}shot.json'), 'w') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            if args.record_wrong:
                with open(os.path.join(save_name, f'wrong_responses_{args.max_new_tokens}_{args.num_shots}shot.jsonl'), 'w') as f:
                    for item in wrong_responses:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")

        


