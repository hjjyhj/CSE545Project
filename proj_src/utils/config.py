import os

# Set Hugging Face cache directory
os.environ['HF_HOME'] = "/scratch/eecs545w25_class_root/eecs545w25_class/cse545_reasoning/hf"

# Configuration parameters
K_OUTPUTS_PER_MODEL = 2  # Number of top outputs to select from each model
MAX_ITERATIONS = 5       # Maximum number of consensus-seeking iterations

# Small language models (SLMs) to use for generating candidate answers
MODEL_LIST = [
    "google/gemma-3-4b-it",
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
]

# Judge model for evaluating consensus (larger model)
JUDGE_MODEL_NAME = "deepseek-chat"

# Input problem statement
ORIGINAL_PROMPT = "Charlie wants to sell beeswax candles.  For every pound of beeswax, he can make 10 tapered candles.  One pound of beeswax and the wicks cost $10.00 in supplies.   If he sells each candle for $2.00 each, what is his net profit if he makes and sells 20 candles?"