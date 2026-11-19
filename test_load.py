import torch
from pathlib import Path

# 假设你的 ckpt 在项目根目录的 outputs/gpt2-medium/ 下
ckpt_path = Path("outputs/gpt2-medium/ckpt_final.pt")
print(f"路径: {ckpt_path.resolve()}")
print(f"存在: {ckpt_path.exists()}")
print(f"大小: {ckpt_path.stat().st_size / 1e6:.1f} MB")

# 尝试加载
ckpt = torch.load(ckpt_path, map_location="cpu")
print(f"\n✅ 加载成功！")
print(f"ckpt 包含的键: {list(ckpt.keys())}")
print(f"训练步数: {ckpt.get('step', '未知')}")

# 检查模型版本（完整版还是简化版）
h0_keys = [k for k in ckpt["model"].keys() if k.startswith("transformer.h.0")]
print(f"\nh.0 的键名:")
for k in h0_keys[:5]:
    print(f"  {k}")