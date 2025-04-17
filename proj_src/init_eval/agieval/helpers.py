import os
import json
from utils import download_url, load_jsonl, record_wrong_responses

def get_dataset(args):
    """Downloads and loads the AGIEval SAT Math dataset if it doesn't already exist.
     We also append the multiple-choice options to the question text"""
    test_filepath = os.path.join(args.data_root, "sat-math.jsonl")
    if not os.path.exists(test_filepath):
        download_url(
            "https://raw.githubusercontent.com/ruixiangcui/AGIEval/main/data/v1_1/sat-math.jsonl",
            args.data_root,
        )
        downloaded_file = os.path.join(args.data_root, "sat-math.jsonl")
        if os.path.exists(downloaded_file):
            os.rename(downloaded_file, test_filepath)

    dataset = []
    # I append the options to the end of each question
    with open(test_filepath, "r", encoding="utf-8") as f:
        for line in f:
                item = json.loads(line)
                question = item["question"]
                options = "\n".join(item["options"])
                full_question = f"{question}\n{options}"
                dataset.append({
                "instruction": full_question, 
                "output": item["label"]
            })
    return dataset

def print_sample_outputs(dataset, num_samples=3):
    """
    Prints a few sample input-output pairs from the dataset.
    """
    print(f"Showing {num_samples} sample(s):\n")
    for i, example in enumerate(dataset[:num_samples]):
        print(f"Example {i+1}:")
        print("Input:\n", example["input"])
        print("Output:", example["output"])
        print("-" * 40)

