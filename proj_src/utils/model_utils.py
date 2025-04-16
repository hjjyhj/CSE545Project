import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Gemma3ForCausalLM
from openai import OpenAI

# torch.backends.cuda.enable_mem_efficient_sdp(False)
# torch.backends.cuda.enable_flash_sdp(False)
# torch.backends.cuda.enable_math_sdp(True)

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


def generate_model_outputs(model, tokenizer, prompt, num_beams=1, max_new_tokens=1024):
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
    # Tokenize input with proper attention mask
    tokenized_input = tokenizer(prompt, return_tensors="pt", padding=True)
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
