import json
import argparse
import os

def compare_answers(dataset):
    """
    Compare answers between original and distilled datasets.
    Auto-detect 'answer' or 'output' keys without extra parameters.
    Saves indices of differing answers and accuracy to a log file.
    """
    # Base directory path where data/<dataset> is located
    # Place this script in your project root, alongside the 'data/' folder.
    # e.g.,
    # /path/to/project/
    # ├── compare_answers.py
    # └── data/
    #     └── <dataset>/
    #         ├── <dataset>_train.json
    #         └── distilled_<dataset>.json

    base_dir = os.path.join("data", dataset)
    train_file = os.path.join(base_dir, f"{dataset}_train.json")
    distilled_file = os.path.join(base_dir, f"distilled_{dataset}.json")
    log_file = os.path.join(base_dir, f"{dataset}_diff.log")

    # Load JSON data
    with open(train_file, 'r', encoding='utf-8') as f:
        original = json.load(f)
    with open(distilled_file, 'r', encoding='utf-8') as f:
        distilled = json.load(f)

    # Possible answer keys
    key_candidates = ['answer', 'output']

    # Helper to extract the first matching key value
    def get_answer(item):
        for key in key_candidates:
            if key in item:
                return str(item[key]).strip()
        return ""

    # Compare answers
    total = min(len(original), len(distilled))
    diffs = []
    for idx in range(total):
        orig_ans = get_answer(original[idx])
        dist_ans = get_answer(distilled[idx])
        if orig_ans != dist_ans:
            diffs.append(idx + 1)  # 1-based index

    # Calculate accuracy
    matches = total - len(diffs)
    accuracy = ((1-matches / total) * 100) if total > 0 else 0.0

    # Write log file
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Total questions compared: {total}\n")
        f.write(f"Differing answers count: {len(diffs)}\n")
        f.write(f"Accuracy: {accuracy:.2f}%\n")
        f.write("Questions with differing answers (1-based indices):\n")
        f.write(", ".join(map(str, diffs)) + "\n")

    print(f"[Done] Compared {total} entries for dataset '{dataset}'.")
    print(f"Found {len(diffs)} differences. Accuracy: {accuracy:.2f}%")
    print(f"Log saved to {log_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare original vs. distilled answers in data/<dataset>/"
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        help="Name of dataset folder under 'data/' (e.g., gsm8k)"
    )
    args = parser.parse_args()
    compare_answers(args.dataset)
