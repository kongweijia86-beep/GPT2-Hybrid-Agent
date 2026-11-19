import os
import pandas as pd
import matplotlib.pyplot as plt

log_path = r"C:\Users\DELL\Desktop\python 1\from 0 to GPT-2\outputs\gpt2-medium\log.csv"
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(log_path)
print("CSV columns:")
print(df.columns.tolist())
print("\n前五行数据:")
print(df.head())

# 清除列名可能存在的空格
df.columns = df.columns.str.strip()
# 检查必要列
required = ["step", "train_loss", "val_loss"]
for col in required:
    if col not in df.columns:
        raise ValueError(
            f"CSV 中缺少 {col} 列，请检查实际列名！"
        )
# 转换数值
for col in required:
    df[col] = pd.to_numeric(df[col], errors="coerce")

plt.figure(figsize=(12, 6))
train_df = df.dropna(subset=["step", "train_loss"])
val_df = df.dropna(subset=["step", "val_loss"])
plt.plot(
    train_df["step"],
    train_df["train_loss"],
    label="Training Loss",
    linewidth=1.5
)

plt.plot(
    val_df["step"],
    val_df["val_loss"],
    label="Validation Loss",
    linewidth=1.5,
    marker="o",
    markersize=3
)

plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    os.path.join(output_dir, "loss_curve.png"),
    dpi=300
)
plt.show()

print("Total records:", len(df))
print(
    "Minimum training loss:",
    train_df["train_loss"].min()
)
print(
    "Minimum validation loss:",
    val_df["val_loss"].min()
)
print(
    "Final training loss:",
    train_df.iloc[-1]["train_loss"]
)
print(
    "Final validation loss:",
    val_df.iloc[-1]["val_loss"]
)
print("\nLoss curve saved to results/loss_curve.png")
