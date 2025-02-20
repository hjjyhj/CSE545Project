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

After there steps, you could use our downloaded models, by directly specifying the model name in your code like:
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
