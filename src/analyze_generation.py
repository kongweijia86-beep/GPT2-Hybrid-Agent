import json
import re
import csv
import os
import argparse

def tokenize(text):
    """将文本转换为小写单词序列。"""
    return re.findall(r"\b[\w']+\b", text.lower())

def get_ngrams(tokens, n):
    """提取 n-gram。将一段 Token 序列按照连续的 N 个 Token 为一组进行切分，从而提取文本中的连续词组。"""
    return [
        tuple(tokens[i:i+n])
        for i in range(len(tokens) - n + 1)
    ]

def calculate_metrics(text, prompt=""):
    """计算单条生成文本的评价指标。"""
    # 去掉 Prompt，只评价模型生成的部分
    if prompt and text.startswith(prompt):
        text = text[len(prompt):]
    tokens = tokenize(text)
    if not tokens:
        return {
            "length": 0,
            "distinct_1": 0,
            "distinct_2": 0,
            "repetition_3": 0
        }
    # 一个 Token 为一组。
    unigrams = get_ngrams(tokens, 1)
    # 两个连续 Token 为一组。
    bigrams = get_ngrams(tokens, 2)
    # 三个连续 Token 为一组。
    trigrams = get_ngrams(tokens, 3)
    #生成文本中不同单个 Token 的比例。它们用来判断模型生成的文本是否丰富，是否存在大量重复。
    distinct_1 = (
        len(set(unigrams)) / len(unigrams)
    )
    #生成文本中不同连续两个 Token 的比例。它们用来判断模型生成的文本是否丰富，是否存在大量重复。
    distinct_2 = (
        len(set(bigrams)) / len(bigrams)
        if bigrams else 0
    )
    # 重复 Trigram 比例
    repetition_3 = (
        1 - len(set(trigrams)) / len(trigrams)
        if trigrams else 0
    )
    return {
        "length": len(tokens),
        "distinct_1": distinct_1,
        "distinct_2": distinct_2,
        "repetition_3": repetition_3
    }

def analyze_file(file_path):
    results = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            metrics = calculate_metrics(
                record["text"],
                record.get("prompt", "")
            )
            results.append(metrics)
    if not results:
        raise ValueError(f"文件中没有有效文本: {file_path}")
    avg = {} #平均评价指标
    for key in results[0]:
        avg[key] = sum(
            item[key] for item in results
        ) / len(results)
    return avg, len(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default="generation_results"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="generation_metrics.csv"
    )
    args = parser.parse_args()
    rows = []
    for filename in sorted(os.listdir(args.input_dir)):
        if not filename.endswith(".jsonl"):
            continue
        path = os.path.join(
            args.input_dir,
            filename
        )
        metrics, num_samples = analyze_file(path)
        row = {
            "experiment": filename.replace(".jsonl", ""),
            "num_samples": num_samples,
            "avg_length": metrics["length"],
            "distinct_1": metrics["distinct_1"],
            "distinct_2": metrics["distinct_2"],
            "repetition_3": metrics["repetition_3"]
        }
        rows.append(row)
    if not rows:
        raise ValueError("没有找到 JSONL 实验文件！")
    fieldnames = list(rows[0].keys())
    with open(
        args.output,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(f"\n实验: {row['experiment']}")
        print(f"Samples      : {row['num_samples']}")
        print(f"Average Length: {row['avg_length']:.2f}")
        print(f"Distinct-1   : {row['distinct_1']:.4f}")
        print(f"Distinct-2   : {row['distinct_2']:.4f}")
        print(f"Repetition-3 : {row['repetition_3']:.4f}")
    print(f"\n结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
