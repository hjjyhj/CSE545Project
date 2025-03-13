# ECE 545 Project

### Setup
#### Basic Environment
Clone this repo:
```sh
git clone https://github.com/hjjyhj/CSE545Project.git
cd CSE545Project
```

Create conda environment and install dependencies:
```sh
conda create -n ece545 python==3.12
conda activate ece545
pip install -e .
```

##### Additional requirements (for specific usage)
- For evaluation on GSM8k, AIME, MATH-500, please additionally install some other dependencies:
  ```sh
  pip install "git+https://github.com/tongyx361/symeval.git"
  ```

- If you want to run the new Gemma-3 models, please re-install the nighty version of transformers:
  ```sh
  pip install git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3
  ```
  And load the model like:
  ```python
  from transformers import Gemma3ForCausalLM
  model = Gemma3ForCausalLM.from_pretrained(
      model_name_or_path,
      # device_map="auto",
      torch_dtype=torch.bfloat16,
      # trust_remote_code=True,
  ).cuda().eval()
  ```



#### Use our Shared Models & Datasets
To use our shared models and datasets by default, first set the environment variable `HF_HOME` in `~/.bashrc`:
```sh
export HF_HOME='/scratch/eecs545w25_class_root/eecs545w25_class/cse545_reasoning/hf'
```
Then run the following command to let it take effect:
```sh
source ~/.bashrc
```

Or, you could alternatively add the following code at the top of your python scripts (be sure to write it before `import transformers` and `import torch`):
```python
import os
os.environ['HF_HOME'] = "/scratch/eecs545w25_class_root/eecs545w25_class/cse545_reasoning/hf"
```

After these steps, you could use our downloaded models, by directly specifying the model name in your code like:
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", device_map='auto', torch_dtype=torch.float16)

prompt = "9.8 and 9.11, which is bigger?"
in_tokens = tokenizer(prompt, return_tensors="pt")['input_ids'].to(model.device)

out = model.generate(in_tokens, max_new_tokens=50)
output = tokenizer.batch_decode(out)
print(output[0])
```

#### Test
Try to run the following script to check if everything works:
```sh
cd ./proj_src/main
python test_inference.py
```
This should be able to use our cached models (without automatic downloading).

### Directory Structure:
- `proj_src/main`: entry point for running our pipeline.
- `proj_src/scripts`: useful scripts (like model download).
- `proj_src/init_eval`: evaluation of single models (before building our own pipeline).
- `proj_src/pipeline`: main implementation of our pipeline.
- `proj_src/utils`: some auxiliary scripts for running the pipeline.

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
- Gemma-3-1b (google/gemma-3-1b-it)
- Gemma-3-4b (google/gemma-3-4b-it)

**Medium Models (~7B)**
- Llama-3.1-8B (meta-llama/Llama-3.1-8B-Instruct)
- Mistral-7B-v0.3 (mistralai/Mistral-7B-Instruct-v0.3)
- Qwen2.5-7B (Qwen/Qwen2.5-7B-Instruct)
- Qwen2.5-7B-Math (Qwen/Qwen2.5-Math-7B)


**Math Benchmarks**
> A list of candidate math benchmarks: https://github.com/huggingface/evaluation-guidebook/blob/main/contents/automated-benchmarks/some-evaluation-datasets.md 
- [GSM8K](https://huggingface.co/datasets/openai/gsm8k) (openai/gsm8k) ([Reference evaluation code](https://github.com/Guangxuan-Xiao/GSM8K-eval))
- [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) (HuggingFaceH4/MATH-500) ([Reference evaluation code](https://github.com/openai/simple-evals/blob/main/math_eval.py))
- [AIME](https://huggingface.co/datasets/di-zhang-fdu/AIME_1983_2024) (di-zhang-fdu/AIME_1983_2024) ([Reference evaluation code](https://github.com/huggingface/open-r1/blob/main/src/open_r1/evaluate.py))
- [AGIEval](https://huggingface.co/datasets/hails/agieval-sat-math) (hails/agieval-sat-math) ([Reference evaluation code](https://github.com/ruixiangcui/AGIEval))


### TODO
- Run initial evaluation of every single model on each benchmark (while seeking more possible models and benchmarks). Work in `proj_src/init_eval`.
- Begin building the pipeline. Work in `proj_src/pipeline`.
