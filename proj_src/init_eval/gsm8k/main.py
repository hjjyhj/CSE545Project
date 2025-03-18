import os
os.environ['HF_HOME'] = "/scratch/eecs545w25_class_root/eecs545w25_class/cse545_reasoning/hf"

import argparse
import re
import json
from symeval import EvaluatorMathBatch

from datasets import load_dataset
from tqdm import tqdm
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from utils import download_url, load_jsonl, record_wrong_responses

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

def load(model_name_or_path):
    print(f"Loading model from {model_name_or_path} ...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    if "gemma-3" in model_name_or_path.lower():
        from transformers import Gemma3ForCausalLM
        model = Gemma3ForCausalLM.from_pretrained(
            model_name_or_path,
            # device_map="auto",
            torch_dtype=torch.bfloat16,
            # trust_remote_code=True,
        ).cuda().eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0

    model.eval()

    return model, tokenizer

def evaluate_model_on_gsm8k(model, tokenizer, dataset, evaluator, num_shots, generate_kwargs, record_wrong, f):
    """
    Evaluates the given model on the gsm8k dataset.
    """

    cur_count = 0
    correct_count = 0
    trunc_wrong_count = 0
    total_count = len(dataset)

    wrong_responses = []

    for sample in tqdm(dataset):
        question = sample['instruction']
        true_answer = sample['gt_answer']

        # Generate the model's response
        prompt = format_prompt(question, num_shots)

        input_text = tokenizer(
            prompt,
            padding=False,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids = input_text.input_ids.cuda()
        attention_mask = input_text.attention_mask.cuda()

        output_ids = model.generate(
            input_ids=input_ids, attention_mask=attention_mask, **generate_kwargs
        )

        response = []
        for i in range(output_ids.shape[0]):
            response.append(
                tokenizer.decode(
                    output_ids[i][input_ids.shape[1] :],
                    skip_special_tokens=True,
                    ignore_tokenization_space=True,
                )
            )

        if not len(response) > 1:
            response = response[0]

        # Extract the model's answer
        model_answer = extract_answer_from_response(response)

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

    accuracy = correct_count / total_count

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
    parser.add_argument('--model_name', default='deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B')
    parser.add_argument("--run_all_models", action='store_true', help='run all models in the list')
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument('--num_shots', type=int, choices=[0, 3], default=0)
    parser.add_argument('--record_wrong', action='store_true', help='record the answers that are wrong')
    parser.add_argument('--data_root', default="/scratch/eecs545w25_class_root/eecs545w25_class/cse545_reasoning/data")
    parser.add_argument('--output_dir', default='./output')
    
    args = parser.parse_args()


    model_list = [
        "stabilityai/stablelm-zephyr-3b",
        "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        # "mistralai/Mistral-7B-Instruct-v0.3",
        # "Qwen/Qwen2.5-7B-Instruct",
        # "meta-llama/Llama-3.1-8B-Instruct",
        # "Qwen/Qwen2.5-Math-7B",
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
            model, tokenizer = load(full_model_name)
            print(f'Evaluating on {full_model_name}')

            with open(os.path.join(save_name, f'responses_{args.max_new_tokens}_{args.num_shots}shot.txt'), 'w') as f:
                results = evaluate_model_on_gsm8k(model, tokenizer, dataset, evaluator, args.num_shots, generate_kwargs, f)
            
            wrong_responses = results["wrong_responses"]
            del results["wrong_responses"]

            print(results)
            with open(os.path.join(save_name, f'scores_{args.max_new_tokens}_{args.num_shots}shot.json'), 'w') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            if args.record_wrong:
                with open(os.path.join(save_name, f'wrong_responses_{args.max_new_tokens}_{args.num_shots}shot.jsonl'), 'w') as f:
                    for item in wrong_responses:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")

    else:
        full_model_name = args.model_name
        save_name = os.path.join(args.output_dir, full_model_name.split('/')[-1])
        os.makedirs(save_name, exist_ok=True)
        model, tokenizer = load(full_model_name)
        print(f'Evaluating on {full_model_name}')

        with open(os.path.join(save_name, f'responses_{args.max_new_tokens}_{args.num_shots}shot.txt'), 'w') as f:
            results = evaluate_model_on_gsm8k(model, tokenizer, dataset, evaluator, args.num_shots, generate_kwargs, args.record_wrong, f)
        
        wrong_responses = results["wrong_responses"]
        del results["wrong_responses"]

        print(results)
        with open(os.path.join(save_name, f'scores_{args.max_new_tokens}_{args.num_shots}shot.json'), 'w') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        if args.record_wrong:
            with open(os.path.join(save_name, f'wrong_responses_{args.max_new_tokens}_{args.num_shots}shot.jsonl'), 'w') as f:
                for item in wrong_responses:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

        


