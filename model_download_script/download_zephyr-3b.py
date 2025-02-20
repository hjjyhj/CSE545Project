from transformers import AutoTokenizer, AutoModelForCausalLM

SAVE_PATH = "/scratch/eecs487w25_class_root/eecs487w25_class/shared_data/johnkimm_dir/models/stablelm-zephyr-3b"
MODEL_NAME = "stabilityai/stablelm-zephyr-3b"

print("Downloading tokenizer for StabilityAI StableLM-Zephyr-3B...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Downloading StabilityAI StableLM-Zephyr-3B model weights... This may take some time.")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=SAVE_PATH)

print("Saving tokenizer and model to disk...")
tokenizer.save_pretrained(SAVE_PATH)
model.save_pretrained(SAVE_PATH)

print(f"StabilityAI StableLM-Zephyr-3B model saved to: {SAVE_PATH}")
