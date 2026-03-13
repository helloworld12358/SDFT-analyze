import json
import argparse
import datasets
from eval_math import check_math
from eval_openfunction import check_openfunction
from eval_magicoder import check_magicoder
from utils import strip_dict
import sys

def main(dataset, predict_jsonl):
    with open(f"data/{dataset}/{dataset}_train.json") as f:
        origin_dataset = json.load(f)
    check_func = get_check_func(dataset)
    distilled_dataset_name = f"distilled_{dataset}"
    output_data_list = get_output_data_list(
        origin_dataset, predict_jsonl, check_func
    )
    output_file = f"data/{dataset}/{distilled_dataset_name}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data_list, f, ensure_ascii=False, indent=4)


def get_check_func(dataset):
    if "gsm8k" in dataset or "MultiArith" in dataset:
        return check_math
    if "openfunction" in dataset:
        return check_openfunction
    if "magicoder" in dataset:
        return check_magicoder
    return lambda x: True


def get_output_data_list(origin_dataset, predict_jsonl, check_func):
    output_data_list = []
    answer_key = None
    with open(predict_jsonl, "r", encoding="utf-8") as f:
        for origin_data, line in zip(origin_dataset, f):
            predict_data = json.loads(line)
            strip_dict(origin_data)
            strip_dict(predict_data)
            if not answer_key:
                answer_key = find_answer_key(origin_data, predict_data)
            if verify(predict_data) and check_func(predict_data):
                origin_data[answer_key] = predict_data["predict"]
            output_data_list.append(origin_data)
    return output_data_list


def find_answer_key(origin_data, predict_data):
    label = predict_data.get("label", "")
    for key, value in origin_data.items():
        # 用子串包含判断，而不是全等
        if label in value:
            return key
    # 再兜底全等（防万一）
    for key, value in origin_data.items():
        if value == label:
            return key

    # 如果还是没找到，就抛错或跳过
    raise ValueError("answer key not found!")


def verify(predict_data):
    ban_set = {
        "reference answer",
        "your response",
        "my response",
        "your own response",
        "now it's your turn"
    }
    for ban in ban_set:
        if ban.lower() in predict_data["predict"].lower():
            return False
    if predict_data["predict"].startswith("Your turn"):
        return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate distilled dataset")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument(
        "--predict_jsonl",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    # 如果 dataset 参数以 '_train' 结尾，则去掉该后缀
    dataset = args.dataset
    if dataset.endswith("_train"):
        dataset = dataset[:-len("_train")]

    # 调用主函数
    main(dataset, args.predict_jsonl)
