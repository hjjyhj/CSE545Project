import os
import json

def build_json(args):
    """Resumes progress if it exists scores.json exist, else start fresh"""
    resume_path = os.path.join(args.save_dir, 'scores.json')
    if os.path.exists(resume_path):
        with open(resume_path, 'r') as f:
            resume_data = json.load(f)
        print(f"[RESUME] Loaded existing progress from {resume_path}")
    else:
        resume_data = {
            "total": 0,
            "correct": 0,
            "exception": 0,
            "accuracy": 0.0
        }
        print(f"[NEW RUN] No previous scores.json found. Starting fresh.")
    return resume_data