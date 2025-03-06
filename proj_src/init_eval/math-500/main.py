import os
os.environ['HF_HOME'] = "/scratch/eecs545w25_class_root/eecs545w25_class/cse545_reasoning/hf"

import argparse
import re
from symeval import EvaluatorMathBatch

from datasets import load_dataset
from tqdm import tqdm
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

# Define the prompt template
INSTRUCTION_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form Answer: $ANSWER (without quotes), where $ANSWER is the answer to the problem. 
For fractions, sqrts, inequalities, matrices and other possible math expressions, please use latex formats in your answer, and you MUST use double slash notations like '\\frac{1}{2}', '3\\sqrt{13}', etc.
Remember to put your final answer on its own line after "Answer:", and you DO NOT need to use a \\boxed command. The final answer should be as brief as possible.
"""

# max: 3
few_shot_examples = [
    "Question:\n Solve for $x$: $$5^{x + 4} = 125^x.$\n\nResponse:\n Writing the right side with $5$ as the base, we have $125^x = (5^3)^x = 5^{3x}$, so our equation is: $$5^{x + 4} = 5^{3x}.$$Then, setting the exponents equal, we obtain $$x + 4 = 3x.$$This yields $2x = 4 \\implies \boxed{x = 2}$\nAnswer: 2",
    "Question:\n Nathan will roll two six-sided dice. What is the probability that he will roll a number less than three on the first die and a number greater than three on the second die? Express your answer as a common fraction.\n\nResponse:\n For the first die to be less than three, it must be a 1 or a 2, which occurs with probability $\\frac{1}{3}$. For the second die to be greater than 3, it must be a 4 or a 5 or a 6, which occurs with probability $\\frac{1}{2}$. The probability of both of these events occuring, as they are independent, is $\\frac{1}{3} \\cdot \\frac{1}{2} = \\boxed{\\frac{1}{6}}$.\nAnswer: \\frac{1}{6}",
    "Question:\n Golf balls are packaged by stacking three balls vertically in a box. Given that the height of the box is 13.5 cm, and that the golf balls are touching each other and the ends of the box, what is the number of centimeters in the circumference of each golf ball? Express your answer as a common fraction in terms of $\\pi$.\n\nResponse\n: Let the diameter of each golf ball be $d$; we have $3d=13.5$ so $d=4.5$. The circumference of each golf ball is $\\pi d = 4.5\\pi = \boxed{\\frac{9\\pi}{2}}.\nAnswer: \\frac{9\\pi}{2}$."
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

    print(f"[Model answer]: {model_answer}")
    print(f"[True answer]: {true_answer}")

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

def evaluate_model_on_math500(model, tokenizer, dataset, evaluator, num_shots, generate_kwargs):
    """
    Evaluates the given model on the MATH-500 dataset.
    """
    correct_count = 0
    total_count = len(dataset['test'])

    for problem in tqdm(dataset['test']):
        question = problem['problem']
        true_answer = problem['answer']

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

        print('='*50)
        # print("[Prompt]: ", prompt)
        print("[Response]: ", response)
        print('-'*30)

        # Extract the model's answer
        model_answer = extract_answer_from_response(response)

        # Compare the model's answer to the true answer
        if model_answer and math_equal(model_answer, true_answer, evaluator):
            print('%%%%% Correct %%%%%')
            correct_count += 1
        else:
            print('%%%%% Wrong %%%%%')

    accuracy = correct_count / total_count
    return accuracy

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_size', choices=['big', 'small'], default='small')
    parser.add_argument('--num_shots', type=int, choices=[0, 3], default=3)
    parser.add_argument('--output_dir', default='./output')
    parser.add_argument('--save_name', default='scores_small_origin.csv')
    args = parser.parse_args()

    if args.model_size == 'small':
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
    else:
        model_list =  [
            # "stabilityai/stablelm-zephyr-3b",
            # "HuggingFaceTB/SmolLM2-1.7B-Instruct",
            # "meta-llama/Llama-3.2-1B-Instruct",
            # "meta-llama/Llama-3.2-3B-Instruct",
            # "Qwen/Qwen2.5-1.5B-Instruct",
            # "Qwen/Qwen2.5-3B-Instruct",
            # "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
            "mistralai/Mistral-7B-Instruct-v0.3",
            "Qwen/Qwen2.5-7B-Instruct",
            "meta-llama/Llama-3.1-8B-Instruct",
            "Qwen/Qwen2.5-Math-7B",
        ]       

    os.makedirs(args.output_dir, exist_ok=True)
    save_name = os.path.join(args.output_dir, args.save_name)
    if not os.path.exists(save_name):
        with open(save_name, 'a') as f:
            f.write('model,acc\n')

    # Load the MATH-500 dataset
    dataset = load_dataset("HuggingFaceH4/MATH-500")

    # math expression evaluator
    evaluator = EvaluatorMathBatch()

    generate_kwargs = dict(max_new_tokens=512, top_p=0.95, temperature=0.8)

    with open(save_name, 'a') as f:
        for full_model_name in model_list:
            model, tokenizer = load(full_model_name)
            print(f'Evaluating on {full_model_name}')
            acc = evaluate_model_on_math500(model, tokenizer, dataset, evaluator, args.num_shots, generate_kwargs)
            print(f'Acc: {acc}')
            f.write(f'{full_model_name.split('/')[-1]},{acc}\n')


        


