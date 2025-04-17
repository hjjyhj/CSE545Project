import os
os.environ['HF_HOME'] = "/scratch/eecs545w25_class_root/eecs545w25_class/cse545_reasoning/hf"
from helpers import get_dataset
import argparse
import re
import json

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

def evaluate_model_on_agieval(model, tokenizer, dataset,num_shots, generate_kwargs, record_wrong, f):
    """
    Evaluates the given model on the agieval dataset.
    """

    # All the metrics used throughout the run.
    cur_count = 0
    correct_count = 0
    trunc_wrong_count = 0
    total_count = len(dataset)
    wrong_responses = []

    for sample in tqdm(dataset):
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
        if check_label(model_answer, true_answer):
            correct = True
            correct_count += 1
        else:
            if not model_answer: # wrong because the model didn't output a formatted answer, most likely to be truncated
                trunc_wrong_count += 1

            if record_wrong:
                wrong_response = {
                    "idx": str(cur_count),
                    "question": question,
                    "answer": sample["label"],
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

    # Needed to avoid divide by zero error.
    if (total_count - correct_count) > 0:
     trunc_wrong_ratio = trunc_wrong_count / (total_count - correct_count)
    else:
     trunc_wrong_ratio = 0.0

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
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument('--num_shots', type=int, choices=[0, 3], default=0)
    parser.add_argument('--record_wrong', action='store_true', help='record the answers that are wrong')
    parser.add_argument('--data_root', default="/scratch/eecs545w25_class_root/eecs545w25_class/cse545_reasoning/data")
    parser.add_argument('--output_dir', default='./output')
    args = parser.parse_args()


    model_list = [
        "stabilityai/stablelm-zephyr-3b",
        # "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        # "meta-llama/Llama-3.2-1B-Instruct",
        # "meta-llama/Llama-3.2-3B-Instruct",
        # "Qwen/Qwen2.5-1.5B-Instruct",
        # "Qwen/Qwen2.5-3B-Instruct",
        # "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        # "mistralai/Mistral-7B-Instruct-v0.3",
        # "Qwen/Qwen2.5-7B-Instruct",
        # "meta-llama/Llama-3.1-8B-Instruct",
        # "Qwen/Qwen2.5-Math-7B",
    ]   

    os.makedirs(args.output_dir, exist_ok=True)

   # Set to variable
    dataset = get_dataset(args)

    # generate_kwargs = dict(max_new_tokens=args.max_new_tokens, top_p=0.95, temperature=0.8)
    generate_kwargs = dict(max_new_tokens=args.max_new_tokens, do_sample=False) # for deterministic generation

    if args.run_all_models:
        for full_model_name in model_list:
            save_name = os.path.join(args.output_dir, full_model_name.split('/')[-1])
            os.makedirs(save_name, exist_ok=True)
            model, tokenizer = load(full_model_name)
            print(f'Evaluating on {full_model_name}')

            with open(os.path.join(save_name, f'responses_{args.max_new_tokens}_{args.num_shots}shot.txt'), 'w') as f:
                results = evaluate_model_on_agieval(model, tokenizer, dataset, args.num_shots, generate_kwargs, f)
            
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
            results = evaluate_model_on_agieval(model, tokenizer, dataset, args.num_shots, generate_kwargs, args.record_wrong, f)
        
        wrong_responses = results["wrong_responses"]
        del results["wrong_responses"]

        print(results)
        with open(os.path.join(save_name, f'scores_{args.max_new_tokens}_{args.num_shots}shot.json'), 'w') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        if args.record_wrong:
            with open(os.path.join(save_name, f'wrong_responses_{args.max_new_tokens}_{args.num_shots}shot.jsonl'), 'w') as f:
                for item in wrong_responses:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

        


