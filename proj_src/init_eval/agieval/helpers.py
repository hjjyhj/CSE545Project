import os
import json
from proj_src.init_eval.agieval.utils import download_url, load_jsonl, record_wrong_responses
from openai import OpenAI
import re

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Gemma3ForCausalLM, 
    AutoConfig
)

#############################################################################
#             Grabs the sat-math dataset for used in Agieeval               #
#############################################################################

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

#############################################################################
#                Saves and tracks progress for each trial                   #
#############################################################################
def build_json(args, dataset):
    """Resumes progress if it exists scores.json exist, else start fresh
    Args:
        args: command line arguments
        dataset: the dataset to evaluate
        total: the total number of samples in the dataset
        correct: the number of correct samples
        exception: the number of remaining samples. If not 0 at the end, it means samples were skipped.
        accuracy: the accuracy of the model
    """
    resume_path = os.path.join(args.save_dir, 'scores.json')
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

#############################################################################
#             Special Functions used in all Agieval tests                   #
#############################################################################

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


#############################################################################
#  Special Functions for parsing the reasoning process more cleanly         #
#############################################################################
def determine_reasoning_process_agieval(output, judge_model, num):
    """
    Summarizes the reasoning behind why the a student chose a particular answer
    and their answer. This used used to send to the other students as external
    feedback. This version specifically looks for letters since the agieval dataset
    uses multiple choice letters (A, B, C, D) as the answer.

    Returns a tuple with the reasoning process and the final answer formatted in the following manner:
    <model_1> answered A. Here is the reasoning process for this answer: <reasoning process>
    """
    # Step 1: Ask for the final answer choice only
    answer_prompt = f"""
Read the student's work below carefully. Identify and extract only the FINAL ANSWER they chose (like A, B, C, D). 
Output ONLY the single letter (A, B, C, or D), with no extra text. If unclear, choose the answer closest to their work.

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
Be concise and clear. Do not mention the final answer letter.

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


#############################################################################
#                       GPT40 mini function call                            #
#############################################################################

def GPT4oMiniResponse(API_KEY, model, prompt):
    """Makes a call to Gpt4o mini model and returns the response.
    Args:
        API_KEY (str): The API key for OpenAI.
        model (str): The model to use.
        prompt (str): The prompt to send to the model."""
    
    student = OpenAI(api_key=API_KEY)
    response = student.chat.completions.create(
        model=model,
        max_tokens=2048,
        temperature=0.0,
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ],
        stream=False
    )
    return response.choices[0].message.content


#############################################################################
#                       Model loading functions                             #
#############################################################################
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

#############################################################################
#             Functions used in the agieval pipeline                        #                                                        #
#############################################################################
def load_model_and_tokenizer_singlemodels(model_name):
    """
    Load a model and its tokenizer with appropriate configurations based on model type.
    
    Args:
        model_name (str): The name or path of the model to load
        
    Returns:
        tuple: (tokenizer, model) The loaded tokenizer and model
    """
    print(f"Loading model: {model_name}")
    
    if "gemma" in model_name.lower():
        model = Gemma3ForCausalLM.from_pretrained(
            model_name,
             device_map=None,
            # device_map="auto",
            torch_dtype=torch.bfloat16,
            # trust_remote_code=True,
        ).cuda().eval()
    else:

        # Common parameters for all models
        model_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "auto",
            "trust_remote_code": True
        }
        
        # Load tokenizer and model
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # Ensure pad token is set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    return tokenizer, model

def load_model_and_tokenizer(model_name):
    """
    Load a model and its tokenizer with appropriate configurations based on model type.
    
    Args:
        model_name (str): The name or path of the model to load
        
    Returns:
        tuple: (tokenizer, model) The loaded tokenizer and model
    """
    print(f"Loading model: {model_name}")
    
    if "gemma" in model_name.lower():
        model = Gemma3ForCausalLM.from_pretrained(
            model_name,
             device_map=None,
            # device_map="auto",
            torch_dtype=torch.bfloat16,
            # trust_remote_code=True,
        ).cuda().eval()
    else:

        # Common parameters for all models
        model_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "auto",
            "trust_remote_code": True
        }
        
        # Load tokenizer and model
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # Ensure pad token is set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    return tokenizer, model

def load_model_and_tokenizercase2(model_name):
    """
    Load a model and its tokenizer with appropriate configurations based on model type.
    
    Args:
        model_name (str): The name or path of the model to load
        
    Returns:
        tuple: (tokenizer, model) The loaded tokenizer and model
    """
    print(f"Loading model: {model_name}")
    
    if "gemma" in model_name.lower():
        model = Gemma3ForCausalLM.from_pretrained(
            model_name,
             device_map=None,
            # device_map="auto",
            torch_dtype=torch.bfloat16,
            # trust_remote_code=True,
        ).to("cpu").eval()
    else:

        # Common parameters for all models
        model_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "auto",
            "trust_remote_code": True
        }
        
        # Load tokenizer and model
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs).to('cpu')

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # Ensure pad token is set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    return tokenizer, model


def load_model_and_tokenizer16(model_name):
    """
    Load model and tokenizer, forcing float16 and eager attention.
    """
    print(f"Loading model (second try): {model_name}")

    # Load config first
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    if hasattr(config, "attn_implementation"):
        config.attn_implementation = "eager"

    # Now load model with config
    model = Gemma3ForCausalLM.from_pretrained(
        model_name,
        config=config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer, model

def generate_model_outputs(model, tokenizer, prompt, num_beams=1, max_new_tokens=2048):
    """
    Generate outputs from a model using beam search.
    
    Args:
        model: The language model
        tokenizer: The tokenizer for the model
        prompt (str): The input prompt
        num_beams (int): Number of beams for beam search
        max_new_tokens (int): Maximum number of tokens to generate
        
    Returns:
        list: List of dictionaries containing model outputs
    """

    model_is_gemma = "gemma" in model.__class__.__name__.lower()

    if model_is_gemma:
        # For Gemma models, use the appropriate tokenizer
        tokenizer.padding_side = "left"  # Important for causal models
        tokenized_input = tokenizer(
            prompt,
            return_tensors="pt",
            padding="longest",
            pad_to_multiple_of=8,
        )
    else:
         # Tokenize input with proper attention mask
        tokenized_input = tokenizer(prompt, 
                                    return_tensors="pt", padding=True)
        
    input_ids = tokenized_input.input_ids.to(model.device)
    attention_mask = tokenized_input.attention_mask.to(model.device)

    # Generate outputs using beam search
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0,
        num_beams=num_beams,
        num_return_sequences=num_beams,
        # early_stopping=True,
        pad_token_id=tokenizer.pad_token_id
    )
    
    # Process and collect outputs
    model_outputs = []
    for beam_idx, beam_output in enumerate(outputs, start=1):
        answer = tokenizer.decode(beam_output, skip_special_tokens=True).strip()
        output_entry = {
            "model": model.__class__.__name__,
            "beam": beam_idx,
            "output": answer
        }
        model_outputs.append(output_entry)
        # print(f"Beam {beam_idx}: {answer}\n")
        
    return model_outputs


# Replace with deepseek
def get_judge_evaluation(judge_model, judge_prompt):
    # """
    # Get evaluation from the judge model.
    
    # Args:
    #     judge_model: The judge language model
    #     judge_tokenizer: The tokenizer for the judge model
    #     judge_prompt (str): The prompt for the judge
        
    # Returns:
    #     str: The judge's response
    # """
    # # Tokenize judge prompt
    # tokenized_input = judge_tokenizer(judge_prompt, return_tensors="pt", padding=True)
    # input_ids = tokenized_input.input_ids.to(judge_model.device)
    # attention_mask = tokenized_input.attention_mask.to(judge_model.device)
    
    # # Generate judge's evaluation
    # output = judge_model.generate(
    #     input_ids,
    #     attention_mask=attention_mask,
    #     max_new_tokens=2048,
    #     num_beams=10,
    #     early_stopping=True,
    #     pad_token_id=judge_tokenizer.pad_token_id
    # )
    
    # # Extract only the generated part (excluding the prompt)
    # prompt_length = input_ids.shape[-1]
    # generated_tokens = output[0][prompt_length:]
    
    # return judge_tokenizer.decode(generated_tokens, skip_special_tokens=True)
    #  
    response = judge_model.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": judge_prompt},
        ],
        stream=False
    )
    # print(response.choices[0].message.content) 
    return (response.choices[0].message.content)


def create_consensus_prompt_agieval(original_prompt, candidate_answers, is_final_iteration, reasoning_dict=None, num_of_models=2):
    """
    Create a prompt for the judge model to evaluate consensus among candidate answers.

    Args:
        original_prompt (str): The original problem statement
        candidate_answers (list): List of candidate answer dictionaries
        is_final_iteration (bool): Whether this is the final iteration
        reasoning_dict (list of dict, optional): List of reasoning details per model
        num_of_models (int, optional): Number of models to include if no consensus

    Returns:
        str: The prompt for the judge model
    """
    # Base prompt with original question and candidate answers
    prompt = f"""Below are candidate answers for the question: "{original_prompt}"
Candidate Answers:
"""
    for entry in candidate_answers:
        prompt += f"- {entry['output']}\n"

    if is_final_iteration:
        prompt += """
        IMPORTANT: Imagine you are just a summarizer, and you don't have any reasoning ability. Make sure to only summarize the answer from the inputs given.
        Please analyze the candidate answers above and determine the most frequent final answer. 
        Do not repeat the candidate answers or the question. 
        Based solely on the candidate answers, provide one concise final answer along with a detailed explanation of your reasoning.
        Please do not provide the candidate answers on the answer. If there is no consensus on the candidate answers, output the most popular answer.
        Do not add any reasoning of your own. Only use the output from the input given.
        Your response should be in the following format (Don't use $\box$ for the Final Answer, just put the number there):

        Final Conclusion:
        Reasoning: <detailed explanation>
        Final Answer: <the letter answer>
        """
    else:
        prompt += """
        IMPORTANT: Imagine you are just a summarizer, and you don't have any reasoning ability. Make sure to only summarize the answer from the inputs given.
        Please analyze the candidate answers above and decide whether there is consensus among them.
        Only when there is a complete consensus amongst the models, based solely on the candidate answers, provide one concise final answer along with a detailed explanation of your reasoning.
        Please do not provide the candidate answers on the answer. Do not add any reasoning of your own. Only use the output from the input given.
        Output your final answer in the following format (Don't use $\box$ for the Final Answer, just put the number there):

        Final Conclusion:
        Reasoning: <detailed explanation>
        Final Answer: <the letter answer>

        If there is no consensus, we would like to prompt the small models again with the original question + reasoning process and answer from those models.

        Please output in the following format:

        No Consensus. Refer to the output from the models and rethink about the reasoning process. 
        Specific Info:
        """

        # Add specific model info if needed
        if reasoning_dict and num_of_models:
            for i in range(min(num_of_models, len(reasoning_dict))):
                model_info = reasoning_dict[i]
                prompt += f"""{model_info['reasoning']}\n"""

        prompt += """
        Note that each model can output multiple answers using beam search. For the majority_answer, give the most popular answer from each model.
        If there is a tie, give any number out of the most popular candidates. Do not add any reasoning of your own. Only use the output from the input given.
        Do not repeat the candidate answers or the question.
        You MUST follow the output format I provided to you.
        """

    return prompt


def update_prompts_from_feedback(original_prompt, reasoning_dict, num_models=2):
    """
    Build updated prompts for the next iteration using the collected reasoning data.
    
    Args:
        original_prompt (str): The original problem statement.
        reasoning_dict (list): A list of dicts {"model_name": int, "output": str, "reasoning": str}.
        num_models (int): Number of models.
        
    Returns:
        list: Updated prompts for each model.
    """

    updated_prompt_dic = {}
    
    for i in range(1, num_models + 1):
        prompt_text = f"""
Here is the question: {original_prompt}
These are answers generated by some students.
"""
        for entry in reasoning_dict:
            prompt_text += entry["reasoning"] + "\n"

        prompt_text += """Review each student's answer and select the answer you think is right. 
If you think all the answers are wrong, propose a new answer and give the reasoning process behind it. Your answer:
"""
        updated_prompt_dic[f'model_{i}'] = prompt_text

    updated_prompt = [updated_prompt_dic[key] for key in updated_prompt_dic.keys()]
    return updated_prompt

