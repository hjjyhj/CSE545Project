from transformers import AutoTokenizer, AutoModel

# Define a unique save path for Qwen2.5-VL-3B-Instruct
SAVE_PATH = "/scratch/eecs487w25_class_root/eecs487w25_class/shared_data/johnkimm_dir/models/qwen2.5-3b"
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

print("Downloading tokenizer for Qwen2.5-VL-3B-Instruct...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Downloading Qwen2.5-VL-3B-Instruct model weights... This may take time.")
model = AutoModel.from_pretrained(MODEL_NAME, cache_dir=SAVE_PATH)

print("Saving tokenizer and model to disk...")
tokenizer.save_pretrained(SAVE_PATH)
model.save_pretrained(SAVE_PATH)

print(f"Qwen2.5-VL-3B-Instruct model saved to: {SAVE_PATH}")
