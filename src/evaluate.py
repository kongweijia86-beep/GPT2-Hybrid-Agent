import argparse
import math
import time
import os
import torch

from model import GPT, GPTConfig
from data import DataLoaderLite

@torch.no_grad()
def compute_val_loss(model, val_loader, eval_steps, device):
    """计算模型的验证 Loss 和 Perplexity。"""
    model.eval()
    val_loader.reset()
    losses = []
    for _ in range(eval_steps):
        x, y = val_loader.next_batch()

        x = x.to(device)
        y = y.to(device)

        _, loss = model(x, y)

        losses.append(loss.item())

    avg = sum(losses) / len(losses)
    ppl = math.exp(avg)
    print(f"Evaluation steps : {eval_steps}")
    print(f"val loss         : {avg:.6f}")
    print(f"perplexity       : {ppl:.4f}")
    return avg, ppl

@torch.no_grad()
def benchmark_speed(model, device, seq_len=128,
                    batch_size=4, n_iters=20):
    """评价模型前向传播速度。"""
    model.eval()
    x = torch.randint(
        0, 50257,
        (batch_size, seq_len),
        device=device
    )
    # Warmup
    for _ in range(3):
        _ = model(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_iters):
        _ = model(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = (time.time() - t0) / n_iters
    tokens = batch_size * seq_len
    print(f"Forward time : {dt * 1000:.2f} ms/iter")
    print(f"Tokens/sec   : {tokens / dt:,.0f}")
    return dt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=str,
        default=r"C:\Users\DELL\Desktop\python 1\from 0 to GPT-2\outputs\gpt2-medium\ckpt_final_1.pt",
        help="checkpoint 文件路径"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "test"],
        help="评估哪个数据集"
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=100,
        help="评估使用多少个 batch"
    )
    parser.add_argument(
        "--no_benchmark",
        action="store_true",
        help="关闭速度测试"
    )
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"当前设备: {device}")

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(
            f"找不到权重文件，请检查路径: {args.ckpt}"
        )
    print(f"\n加载 checkpoint: {args.ckpt}")
    ckpt = torch.load(
        args.ckpt,
        map_location=device
    )

    cfg = ckpt.get("config", {})
    model_cfg_dict = cfg.get("model", cfg)
    model_cfg = GPTConfig(**model_cfg_dict)
    model = GPT(model_cfg)
    state_dict = ckpt["model"]
    # 兼容 DDP 保存的权重
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {
            k.replace("module.", "", 1): v
            for k, v in state_dict.items()
        }

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"Checkpoint step : {ckpt.get('step', '未知')}")
    print(f"Number of layers: {model_cfg.n_layer}")
    print(f"Number of heads : {model_cfg.n_head}")
    print(f"Embedding dim   : {model_cfg.n_embd}")
    print(f"Block size      : {model_cfg.block_size}")
    print(f"Vocabulary size : {model_cfg.vocab_size}")

    training_cfg = cfg.get("training", {})

    B = training_cfg.get("batch_size", 8)
    T = training_cfg.get(
        "block_size",
        model_cfg.block_size
    )
    print(f"\n使用数据集: {args.split}")
    print(f"数据文件: {os.path.join(args.data_dir, args.split + '.txt')}")
    print(f"Batch size: {B}")
    print(f"Block size: {T}")

    data_file = os.path.join(
        args.data_dir,
        f"{args.split}.txt"
    )

    if not os.path.exists(data_file):
        print(
            f"\n警告: 本地没有找到 {data_file}，"
            "跳过 Loss 计算。"
        )
    else:
        print("\n开始计算 Validation Loss")
        val_loader = DataLoaderLite(
            B=B,
            T=T,
            split=args.split,
            data_dir=args.data_dir
        )
        val_loss, ppl = compute_val_loss(
            model,
            val_loader,
            args.eval_steps,
            device
        )

    if not args.no_benchmark:
        print("\n前向传播速度测试")
        benchmark_speed(model, device)
    print("\nEvaluation completed successfully!")
    if os.path.exists(data_file):
        print(f"Final validation loss: {val_loss:.6f}")
        print(f"Final perplexity: {ppl:.4f}")

if __name__ == "__main__":
    main()
