#!/bin/bash
#SBATCH --account=eecs545w25_class  
#SBATCH --partition=spgpu  
#SBATCH --gres=gpu:1 
#SBATCH --job-name=my_job_name        # Job name
#SBATCH --output=output.txt           # Standard output file
#SBATCH --error=error.txt             # Standard error file
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=1           # Number of tasks per node
#SBATCH --mem-per-cpu=16G  
#SBATCH --cpus-per-task=4            # Number of CPU cores per task
#SBATCH --time=1:00:00  
#SBATCH --mail-type=BEGIN,END    
#SBATCH --output=results/%x_%j.out   

# Load any of your required modules in your environment here
module load python3.11-anaconda/2024.02    

# Ensure Conda is initialized
eval "$(conda shell.bash hook)"
source /home/larnell/miniconda3/etc/profile.d/conda.sh

# This accesses the shared folder
export HF_HOME='/scratch/eecs545w25_class_root/eecs545w25_class/cse545_reasoning/hf'
export HUGGINGFACE_HUB_CACHE='/scratch/eecs545w25_class_root/eecs545w25_class/cse545_reasoning/hf'

# Activate your conda environment
conda activate ece545

# Allows great lakes access to the lm-evaluation-harness program
export PYTHONPATH=/home/larnell/lm-evaluation-harness:$PYTHONPATH

# We specify the agieval_sat_math task name
TASK_NAME=${TASK_NAME:-"agieval_sat_math"}
JOB_NAME=""
MODEL_PATH=""

# Zero shot
python -m lm_eval \
    --model_args pretrained=$MODEL_PATH,trust_remote_code=True,dtype=auto \
    --tasks $TASK_NAME \
    --device cuda:0 \
    --num_fewshot 0 \
    --output_path results/${JOB_NAME}_0.json

wait

# Three shot
python -m lm_eval \
    --model_args pretrained=$MODEL_PATH,trust_remote_code=True,dtype=auto \
    --tasks $TASK_NAME \
    --device cuda:0 \
    --num_fewshot 3 \
    --output_path results/${JOB_NAME}_3.json