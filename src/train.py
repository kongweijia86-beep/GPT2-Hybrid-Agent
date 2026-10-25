import os
import math
import time
import argparse
import yaml
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from model import GPT, GPTConfig
from data import DataLoaderLite


def get_lr(it, max_lr, min_lr, warmup_steps, max_steps):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


@torch.no_grad()
def evaluate(model, val_loader, eval_steps, device):
    model.eval()
    val_loader.reset()
    losses = []
    for _ in range(eval_steps):
        x, y = val_loader.next_batch()
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    # 【修改点】：支持通过命令行传入数据路径
    parser.add_argument("--data_dir", type=str, default="data")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        assert torch.cuda.is_available(), "DDP requires CUDA"
        dist.init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if master_process:
        print(f"using device: {device}, ddp: {ddp}, world_size: {ddp_world_size}")

    torch.manual_seed(cfg['training']['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg['training']['seed'])

    B = cfg['training']['batch_size']
    T = cfg['training']['block_size']
    total_batch_size = cfg['training']['total_batch_size']

    assert total_batch_size % (B * T * ddp_world_size) == 0, \
        "total_batch_size must be divisible by B*T*world_size"
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)

    if master_process:
        print(f"total batch size: {total_batch_size}")
        print(f"grad accum steps: {grad_accum_steps}")

    train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='train',
                                  data_dir=args.data_dir)
    val_loader = DataLoaderLite(B=B, T=T, process_rank=0, num_processes=1, split='val', data_dir=args.data_dir)

    torch.set_float32_matmul_precision('high')

    model_cfg = GPTConfig(**cfg['model'])
    model = GPT(model_cfg)
    model.to(device)

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model

    optimizer = raw_model.configure_optimizers(
        weight_decay=cfg['training']['weight_decay'],
        learning_rate=cfg['training']['max_lr'],
        device=device
    )

    out_dir = os.path.join(cfg['output']['dir'], cfg['output']['run_name'])
    if master_process:
        os.makedirs(out_dir, exist_ok=True)
        log_path = os.path.join(out_dir, 'log.csv')
        with open(log_path, 'w') as f:
            f.write("step,train_loss,val_loss,lr,grad_norm,dt_ms,tokens_per_sec\n")

    max_steps = cfg["training"]["max_steps"]
    eval_interval = cfg["training"]["eval_interval"]
    eval_steps = cfg["training"]["eval_steps"]
    save_interval = cfg["training"]["save_interval"]
    log_interval = cfg["training"]["log_interval"]
    grad_clip = cfg["training"]["grad_clip"]
    max_lr = cfg["training"]["max_lr"]
    min_lr = cfg["training"]["min_lr"]
    warmup_steps = cfg["training"]["warmup_steps"]

    model.train()
    for step in range(max_steps):
        t0 = time.time()
        optimizer.zero_grad()
        loss_accum = 0.0

        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)

            logits, loss = model(x, y)
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()

        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        lr = get_lr(step, max_lr, min_lr, warmup_steps, max_steps)
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        optimizer.step()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        dt = time.time() - t0
        tokens_per_sec = B * T * grad_accum_steps * ddp_world_size / dt

        val_loss = float('NaN')
        if step % eval_interval == 0 or step == max_steps - 1:
            val_loss = evaluate(model, val_loader, eval_steps, device)
            if master_process:
                print(f"[eval] step {step}, val_loss {val_loss:.4f}")

        if master_process and (step % log_interval == 0 or step == max_steps - 1):
            print(
                f"step {step:5d} | "
                f"train_loss {loss_accum.item():.4f} | "
                f"val_loss {val_loss:.4f} | "
                f"lr {lr:.2e} | "
                f"norm {norm:.3f} | "
                f"dt {dt * 1000:.1f}ms | "
                f"tok/s {tokens_per_sec:,.0f}"
            )
            with open(log_path, "a") as f:
                f.write(
                    f"{step},{loss_accum.item():.6f},{val_loss:.6f},"
                    f"{lr:.6e},{norm:.4f},{dt * 1000:.2f},{tokens_per_sec:.2f}\n"
                )

        if master_process and step > 0 and step % save_interval == 0:
            ckpt = {
                "step": step,
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": cfg,
            }
            torch.save(ckpt, os.path.join(out_dir, f"ckpt_{step:06d}.pt"))
            print(f"saved checkpoint at step {step}")

    if master_process:
        ckpt = {
            "step": max_steps,
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg,
        }
        torch.save(ckpt, os.path.join(out_dir, "ckpt_final.pt"))
        print("training done, final checkpoint saved.")

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()