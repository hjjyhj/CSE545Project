from transformers import AutoTokenizer, AutoModelForCausalLM

SAVE_PATH = "/scratch/eecs487w25_class_root/eecs487w25_class/shared_data/johnkimm_dir/models/llama-3.2-3b"
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

print("Downloading tokenizer for Llama-3.2-3B-Instruct...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Downloading Llama-3.2-3B-Instruct model weights... This may take some time.")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=SAVE_PATH)

print("Saving tokenizer and model to disk...")
tokenizer.save_pretrained(SAVE_PATH)
model.save_pretrained(SAVE_PATH)

print(f"Llama-3.2-3B-Instruct model saved to: {SAVE_PATH}")
