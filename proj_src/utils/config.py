import os

# Set Hugging Face cache directory
os.environ['HF_HOME'] = "/scratch/eecs545w25_class_root/eecs545w25_class/cse545_reasoning/hf"

# Configuration parameters
K_OUTPUTS_PER_MODEL = 2  # Number of top outputs to select from each model
MAX_ITERATIONS = 3       # Maximum number of consensus-seeking iterations

# Small language models (SLMs) to use for generating candidate answers
MODEL_LIST = [
    "stabilityai/stablelm-zephyr-3b",
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
]

# Judge model for evaluating consensus (larger model)
JUDGE_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# Input problem statement
ORIGINAL_PROMPT = "I have 3 apples, and my dad has 2 more apples than me. How many apples do we have in total?" 