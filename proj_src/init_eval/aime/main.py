import os
os.environ['HF_HOME'] = "/scratch/eecs545w25_class_root/eecs545w25_class/cse545_reasoning/hf"

import argparse
import re
import pandas as pd
from symeval import EvaluatorMathBatch

from datasets import load_dataset
from tqdm import tqdm
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from utils import download_url, load_jsonl

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

def evaluate_model_on_aime(model, tokenizer, dataset, evaluator, num_shots, generate_kwargs, f):
    """
    Evaluates the given model on the aime dataset.
    """

    cur_count = 0
    correct_count = 0
    total_count = len(dataset)
    print(f'=== Total: {total_count} ===')

    for row in tqdm(dataset.itertuples(index=True, name='Row')):
        sample = {
            "instruction": row.Question,
            "output": row.Answer
        }

        question = sample['instruction']
        true_answer = sample['output']

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
    return accuracy

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B')
    parser.add_argument("--run_all_models", action='store_true', help='run all models in the list')
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument('--num_shots', type=int, choices=[0, 3], default=0)
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

    # Load the aime dataset
    test_filepath = os.path.join(args.data_root, "AIME_Dataset_1983_2024.csv") 
    if not os.path.exists(test_filepath):
        download_url(
            "https://huggingface.co/datasets/di-zhang-fdu/AIME_1983_2024/resolve/main/AIME_Dataset_1983_2024.csv?download=true",
            args.data_root,
        )
        # os.rename(os.path.join(args.data_root, "test.jsonl"), test_filepath)

    list_data_dict = pd.read_csv(test_filepath)
    list_data_dict.rename(columns={"Question": "instruction", "Answer": "output"})
    dataset = list_data_dict

    # math expression evaluator
    evaluator = EvaluatorMathBatch()

    generate_kwargs = dict(max_new_tokens=args.max_new_tokens, top_p=0.95, temperature=0.8)

    if args.run_all_models:
        for full_model_name in model_list:
            save_name = os.path.join(args.output_dir, full_model_name.split('/')[-1])
            os.makedirs(save_name, exist_ok=True)
            model, tokenizer = load(full_model_name)
            print(f'Evaluating on {full_model_name}')

            with open(os.path.join(save_name, f'responses_{args.max_new_tokens}_{args.num_shots}shot.txt'), 'w') as f:
                acc = evaluate_model_on_aime(model, tokenizer, dataset, evaluator, args.num_shots, generate_kwargs, f)
            
            print(f'Acc: {acc}')
            with open(os.path.join(save_name, f'scores_{args.max_new_tokens}_{args.num_shots}shot.txt'), 'w') as f:
                f.write(f"Acc: {acc}\n")

    else:
        full_model_name = args.model_name
        save_name = os.path.join(args.output_dir, full_model_name.split('/')[-1])
        os.makedirs(save_name, exist_ok=True)
        model, tokenizer = load(full_model_name)
        print(f'Evaluating on {full_model_name}')

        with open(os.path.join(save_name, f'responses_{args.max_new_tokens}_{args.num_shots}shot.txt'), 'w') as f:
            acc = evaluate_model_on_aime(model, tokenizer, dataset, evaluator, args.num_shots, generate_kwargs, f)
        
        print(f'Acc: {acc}')
        with open(os.path.join(save_name, f'scores_{args.max_new_tokens}_{args.num_shots}shot.txt'), 'w') as f:
            f.write(f"Acc: {acc}\n")


        


