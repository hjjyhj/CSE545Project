# Author: Larnell Moore
# Creation Date: 3/9/2025
# Purpose: To perform the evaluations for our EECS 545 project with Agieval, I created a template.sh file that describes the job to GreatLakes. Then, this program
# edits that script to swap the details for each model. 

import json
# A dictionary where the key is the model name, and the value is the directory to the model.
model_dict = {
    "stabilityai--stablelm-zephyr-3b": "/scratch/eecs545w25_class_root/eecs545w25_class/cse545_reasoning/hf/hub/models--stabilityai--stablelm-zephyr-3b/snapshots/fe1fd5e36ccb8cb2cb1dc225ab6d2962692b9837",
    "SmolLM2-1.7B-Instruct": "/scratch/eecs545w25_class_root/eecs545w25_class/cse545_reasoning/hf/hub/models--HuggingFaceTB--SmolLM2-1.7B-Instruct/snapshots/4db41b7c29f5e1f883ab90b5129c21519ccb526e",
    "meta-llama--Llama-3.2-1B-Instruct": "/scratch/eecs545w25_class_root/eecs545w25_class/cse545_reasoning/hf/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6",
    "meta-llama--Llama-3.2-3B-Instruct": "/scratch/eecs545w25_class_root/eecs545w25_class/cse545_reasoning/hf/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95",
    "meta-llama--Llama-3.1-8B-Instruct": "/scratch/eecs545w25_class_root/eecs545w25_class/cse545_reasoning/hf/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
    "Qwen2.5-1.5B-Instruct": "/scratch/eecs545w25_class_root/eecs545w25_class/cse545_reasoning/hf/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    "Qwen2.5-3B-Instruct": "/scratch/eecs545w25_class_root/eecs545w25_class/cse545_reasoning/hf/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1",
    "deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B": "/scratch/eecs545w25_class_root/eecs545w25_class/cse545_reasoning/hf/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562",
    "google--gemma-3-1b-it": "/scratch/eecs545w25_class_root/eecs545w25_class/cse545_reasoning/hf/hub/models--google--gemma-3-1b-it/snapshots/9b99be88fdd7a2496bf644baade44348ad736c95",
    "google--gemma-3-4b-it": "/scratch/eecs545w25_class_root/eecs545w25_class/cse545_reasoning/hf/hub/models--google--gemma-3-4b-it/snapshots/dbd91bbaf64a0e591f4340ce8b66fd1dba9ab6bd",
    "mistralai--Mistral-7B-Instruct-v0.3": "/scratch/eecs545w25_class_root/eecs545w25_class/cse545_reasoning/hf/hub/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/e0bc86c23ce5aae1db576c8cca6f06f1f73af2db",
    "Qwen2.5-7B-Instruct": "/scratch/eecs545w25_class_root/eecs545w25_class/cse545_reasoning/hf/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28",
    "Qwen2.5-Math-7B": "/scratch/eecs545w25_class_root/eecs545w25_class/cse545_reasoning/hf/hub/models--Qwen--Qwen2.5-Math-7B/snapshots/b101308fe89651ea5ce025f25317fea6fc07e96e"
}

# This piece of code reads in the template.sh and then edits specific lines in the file.
with open("template.sh", "r") as file:
    template_content = file.read()

# For each model in the dictonary
for model_name, model_path in model_dict.items():
    # We look into the file for specific lines of text to replace
    script_content = template_content.replace("#SBATCH --job-name=my_job_name", f"#SBATCH --job-name={model_name}")
    script_content = script_content.replace('JOB_NAME=""', f'JOB_NAME="{model_name}"')
    script_content = script_content.replace('MODEL_PATH=""', f'MODEL_PATH="{model_path}"')

    # Then, we create a sh file for both the 0-shot and 3-shot scenarios
    output_filename = f"{model_name}.sh"

    with open(output_filename, "w") as file:
        file.write(script_content)

    print(f"Generated SLURM script: {output_filename}")