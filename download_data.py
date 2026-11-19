from datasets import load_dataset
import os

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

print("开始下载 TinyStories validation...")

dataset = load_dataset(
    "roneneldan/TinyStories",
    split="validation"
)

print(f"validation stories: {len(dataset)}")

val_path = os.path.join(DATA_DIR, "val.txt")

with open(val_path, "w", encoding="utf-8") as f:
    for item in dataset:
        f.write(item["text"])
        f.write("\n")

print("下载并转换完成！")
print("保存位置：")
print(os.path.abspath(val_path))