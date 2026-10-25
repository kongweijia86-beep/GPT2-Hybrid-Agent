import argparse
import math
import time
import torch
from model import GPT,GPTConfig
from data import DataLoaderLite
@torch.no_grad()
def compare_with_hf(model,device,seq_len=64,batch_size=2):
    from transformers import GPT2LMHeadModel
    print("loading HuggingFace GPT-2 for comparison")
    model_hf = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    model_hf.eval()
    torch.manual_seed(0)
    x = torch.randint(0,50257,(batch_size,seq_len),device=device)
    model.eval()
    out_mine,_ = model(x)
    out_hf = model_hf(x).logits
    diff = (out_mine-out_hf).abs()
    print(f"max abs diff:{diff.max().item():.6e}")
    print(f"mean abs diff:{diff.mean().item():.6e}")
    return diff.max().item()
@torch.no_grad()
def compute_val_loss(model,val_loader,eval_steps,device):
    model.eval()
    val_loader.reset()
    losses = []
    for _ in range(eval_steps):
        x,y = val_loader.next_batch()
        x,y = x.to(device),y.to(device)
        _,loss = model(x,y)
        losses.append(loss.item())
    avg = sum(losses)/len(losses)  #计算多个batch的loss平均值
    ppl = math.exp(avg)
    print(f"val loss:{avg:.4f},perplexity:{ppl:.2f}")
    return avg,ppl
@torch.no_grad()
def benchmark_speed(model,device,seq_len=128,batch_size=4,n_iters=20):
    model.eval()
    x = torch.randint(0,50257,(batch_size,seq_len),device=device)
    for _ in range(3):
        _ = model(x)  #不记录时间的跑3次，让所有初始化完成，GPU进入稳定态
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_iters):
        _ = model(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = (time.time()-t0)/n_iters
    tokens = batch_size*seq_len
    print(f"forward: {dt * 1000:.2f} ms/iter, {tokens / dt:,.0f} tokens/sec")
    return dt
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--compare_hf", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.ckpt, map_location=device)
    model_cfg = GPTConfig(**ckpt["config"]["model"])
    model = GPT(model_cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    print(f"loaded checkpoint from step {ckpt.get('step', '?')}")

    # 1. validation loss
    B = ckpt["config"]["training"]["batch_size"]
    T = ckpt["config"]["training"]["block_size"]
    val_loader = DataLoaderLite(B=B, T=T, split="val", data_dir=args.data_dir)
    compute_val_loss(model, val_loader, args.eval_steps, device)

    # 2. compare with HF
    if args.compare_hf and model_cfg.n_layer == 12 and model_cfg.n_embd == 768:
        compare_with_hf(model, device)

    # 3. speed benchmark
    if args.benchmark:
        benchmark_speed(model, device)
if __name__ == '__main__':
    main()