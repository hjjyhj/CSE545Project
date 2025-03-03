# Init_Eval

### Models & Benchmarks to test
Before building our framework, first we need to evaluate every single model we want to incorporate, on each benchmark we aim to use.

> Note: To pull llama models, you might have to require access on their huggingface page (e.g. https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct), and log in to your huggingface account using `huggingface-cli login` (`pip install huggingface-hub` to use this command)


**Small Models (~3B):**
- <del>Gemma-2-2B (google/gemma-2-2b-it)</del>
- Zephyr-3B (stabilityai/stablelm-zephyr-3b)
- SmolLM2-1.7B (HuggingFaceTB/SmolLM2-1.7B-Instruct)
- Llama-3.2-1B (meta-llama/Llama-3.2-1B-Instruct)
- Llama-3.2-3B (meta-llama/Llama-3.2-3B-Instruct)
- Qwen2.5-1.5B (Qwen/Qwen2.5-1.5B-Instruct)
- Qwen2.5-3B (Qwen/Qwen2.5-3B-Instruct)
- R1-Distill-Qwen-1.5B (deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)

**Medium Models (~7B)**
- Llama-3.1-8B (meta-llama/Llama-3.1-8B-Instruct)
- Mistral-7B-v0.3 (mistralai/Mistral-7B-Instruct-v0.3)
- Qwen2.5-7B (Qwen/Qwen2.5-7B-Instruct)
- Qwen2.5-7B-Math (Qwen/Qwen2.5-Math-7B)


**Math Benchmarks**
> A list of candidate math benchmarks: https://github.com/huggingface/evaluation-guidebook/blob/main/contents/automated-benchmarks/some-evaluation-datasets.md 
- GSM8K (openai/gsm8k) ([Reference evaluation code](https://github.com/Guangxuan-Xiao/GSM8K-eval))
- MATH-500 (HuggingFaceH4/MATH-500) ([Reference evaluation code](https://github.com/openai/simple-evals/blob/main/math_eval.py))
- AIME (di-zhang-fdu/AIME_1983_2024) ([Reference evaluation code](https://github.com/huggingface/open-r1/blob/main/src/open_r1/evaluate.py))
- AGIEval (hails/agieval-sat-math) ([Reference evaluation code](https://github.com/ruixiangcui/AGIEval))
