from transformers import AutoTokenizer, AutoModelForCausalLM

SAVE_PATH = "/scratch/eecs487w25_class_root/eecs487w25_class/shared_data"  

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

print("🚀 Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("🚀 Downloading model weights... This may take time.")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)
model.save_pretrained(SAVE_PATH)

print(f"Model saved to: {SAVE_PATH}")
