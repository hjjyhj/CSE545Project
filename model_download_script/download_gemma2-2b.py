from transformers import AutoTokenizer, AutoModelForCausalLM

SAVE_PATH = "/scratch/eecs487w25_class_root/eecs487w25_class/shared_data/johnkimm_dir/models/gemma-2-2b"
MODEL_NAME = "google/gemma-2-2b"

print("Downloading tokenizer for gemma-2-2b...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Downloading gemma-2-2b model weights... This may take time.")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=SAVE_PATH)

print("Saving tokenizer and model to disk...")
tokenizer.save_pretrained(SAVE_PATH)
model.save_pretrained(SAVE_PATH)

print(f"Gemma-2-2B model saved to: {SAVE_PATH}")
