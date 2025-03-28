import json
import sys
import os
import csv

def get_subdirectory_name(path, parent_dir):
    parts = path.split(os.sep)  # Split the path into components
    if parent_dir in parts:
        parent_index = parts.index(parent_dir)
        if parent_index + 1 < len(parts):  # Ensure there's a subdirectory after the parent
            return parts[parent_index + 1]
    return None  # Return None if not found

def process_files(benchmark_folder_name, file_paths):
    idx_dict = {}
    
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        idx = int(data.get("idx"))
                        if idx not in idx_dict:
                            idx_dict[idx] = []
                        idx_dict[idx].append(get_subdirectory_name(file_path, benchmark_folder_name))
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON in file {file_path}")
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
    
    idx_dict = dict(sorted(idx_dict.items()))
    
    output_csv = benchmark_folder_name + '.csv'
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Incorrect Question Index", "Models"])
        for idx, models in idx_dict.items():
            csv_writer.writerow([idx, ", ".join(models)])
    print(f"CSV output saved to {output_csv}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Creates a CSV containing which questions were answered incorrectly by which models")
        print("Usage: python script.py <benchmark_folder_name> <file1.jsonl> <file2.jsonl> ...")
        print("Example: python3 combine_analysis.py gsm8k /scratch/eecs545w25_class_root/eecs545w25_class/cse545_reasoning/eval_results/init_eval/gsm8k/gemma-3-4b-it/wrong_responses_1024_0shot.jsonl /scratch/eecs545w25_class_root/eecs545w25_class/cse545_reasoning/eval_results/init_eval/gsm8k/Llama-3.2-3B-Instruct/wrong_responses_1024_0shot.jsonl /scratch/eecs545w25_class_root/eecs545w25_class/cse545_reasoning/eval_results/init_eval/gsm8k/Qwen2.5-3B-Instruct/wrong_responses_1024_0shot.jsonl /scratch/eecs545w25_class_root/eecs545w25_class/cse545_reasoning/eval_results/init_eval/gsm8k/DeepSeek-R1-Distill-Qwen-1.5B/wrong_responses_1024_0shot.jsonl")
    else:
        process_files(sys.argv[1], sys.argv[2:])
