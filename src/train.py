import os
import math
import time
import glob
import random
import argparse
from contextlib import nullcontext

import yaml
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model import GPT, GPTConfig
from data import DataLoaderLite


def get_lr(step, max_lr, min_lr, warmup_steps, max_steps):
    if warmup_steps > 0 and step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    if max_steps <= warmup_steps:
        return min_lr
    ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    ratio = min(1.0, max(0.0, ratio))
    return min_lr + 0.5 * (1 + math.cos(math.pi * ratio)) * (max_lr - min_lr)


@torch.no_grad()
def evaluate(model, loader, eval_steps, device, use_amp):
    if eval_steps <= 0:
        raise ValueError('eval_steps must be positive')
    was_training = model.training
    model.eval()
    try:
        loader.reset()
        total_loss = torch.zeros((), device=device, dtype=torch.float64)
        for _ in range(eval_steps):
            x, y = loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type='cuda', enabled=use_amp):
                _, loss = model(x, y)
            total_loss += loss.detach().double()
        return (total_loss / eval_steps).item()
    finally:
        model.train(was_training)


def save_checkpoint(path, step, raw_model, optimizer, scaler, cfg, rank, world_size):
    """在训练过程中保存模型的检查点（Checkpoint），以便训练中断后能够恢复，而不必从头开始训练。"""
    checkpoint = {
        'step': step,
        'model': raw_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict(),
        'config': cfg,
        'world_size': world_size,
        'rank0_cpu_rng': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        checkpoint['rank0_cuda_rng'] = torch.cuda.get_rng_state()
    temp_path = path + '.tmp'
    torch.save(checkpoint, temp_path)
    os.replace(temp_path, path)#采用临时文件以此来降低因保存过程中断而损坏已有检查点的风险。


def find_checkpoint(out_dir, resume):
    """根据用户指定的恢复训练方式，查找合适的模型检查点（Checkpoint），并返回检查点文件路径"""
    if resume == 'none':
        return None
    if resume != 'auto':
        if not os.path.isfile(resume):
            raise FileNotFoundError(f'Checkpoint not found: {resume}')
        return resume
    paths = glob.glob(os.path.join(out_dir, 'ckpt_*.pt'))
    if not paths:
        return None

    best_path, best_step = None, -1
    for path in paths:
        try:
            obj = torch.load(path, map_location='cpu', weights_only=True)
            step = int(obj['step'])
            if step > best_step:
                best_path, best_step = path, step
        except Exception as exc:
            print(f'Warning: skipping unreadable checkpoint {path}: {exc}')
    return best_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--resume', type=str, default='auto',
                        help='auto: newest checkpoint; none: fresh run; or explicit path')
    parser.add_argument('--no_eval', action='store_true', help='disable in-training validation')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    tc = cfg['training']
    ddp = int(os.environ.get('RANK', '-1')) != -1
    if ddp:
        if not torch.cuda.is_available():
            raise RuntimeError('DDP with NCCL requires CUDA')
        dist.init_process_group(backend='nccl')
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
    else:
        rank, local_rank, world_size = 0, 0, 1
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    master = rank == 0

    try:
        seed = int(tc['seed'])
        random.seed(seed + rank)
        torch.manual_seed(seed + rank)
        if device.type == 'cuda':
            torch.cuda.manual_seed(seed + rank)
            torch.set_float32_matmul_precision('high')

        B = int(tc['batch_size'])
        T = int(tc['block_size'])
        total_batch_size = int(tc['total_batch_size'])
        max_steps = int(tc['max_steps'])
        eval_interval = int(tc.get('eval_interval', 0))
        eval_steps = int(tc.get('eval_steps', 10))
        save_interval = int(tc['save_interval'])
        log_interval = int(tc['log_interval'])
        grad_clip = float(tc['grad_clip'])
        max_lr = float(tc['max_lr'])
        min_lr = float(tc['min_lr'])
        warmup_steps = int(tc['warmup_steps'])
        do_eval = not args.no_eval and eval_interval > 0

        if min(B, T, total_batch_size, max_steps, save_interval, log_interval) <= 0:
            raise ValueError('batch, block, total_batch, max_steps, save/log intervals must be positive')
        if not (0 <= warmup_steps < max_steps):
            raise ValueError('Require 0 <= warmup_steps < max_steps')
        if not (0 <= min_lr <= max_lr):
            raise ValueError('Require 0 <= min_lr <= max_lr')
        if total_batch_size % (B * T * world_size) != 0:
            raise ValueError('total_batch_size must be divisible by B*T*world_size')
        grad_accum_steps = total_batch_size // (B * T * world_size)

        out_dir = os.path.join(cfg['output']['dir'], cfg['output']['run_name'])
        if master:
            os.makedirs(out_dir, exist_ok=True)
        if ddp:
            dist.barrier()#dist.barrier() 是 PyTorch 分布式通信中的一个同步操作，称为进程屏障（Barrier）。它的作用是：让所有参与该通信组的进程在这里等待，直到所有进程都到达这个位置，才允许继续执行。
        log_path = os.path.join(out_dir, 'log.csv')

        train_loader = DataLoaderLite(B=B, T=T, process_rank=rank,
                                      num_processes=world_size, split='train',
                                      data_dir=args.data_dir)

        val_loader = (DataLoaderLite(B=B, T=T, process_rank=0,
                                     num_processes=1, split='val',
                                     data_dir=args.data_dir)
                      if master and do_eval else None)

        model = GPT(GPTConfig(**cfg['model'])).to(device)
        raw_model = model
        optimizer = raw_model.configure_optimizers(
            weight_decay=tc['weight_decay'],
            learning_rate=max_lr,
            device=str(device),
        )
        use_amp = device.type == 'cuda' and bool(tc.get('use_amp', True))
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

        ckpt_path = find_checkpoint(out_dir, args.resume) if master else None
        #find_checkpoint(out_dir,resume)返回的是最新ckpt路径，如果每个rank各自去glob找，可能出现不一致，导致各rank加载了不同checkpoint,DDP直接崩，所以只让Rank0找，然后广播给所有人
        if ddp:
            holder = [ckpt_path]
            dist.broadcast_object_list(holder, src=0)
            ckpt_path = holder[0]

        start_step = 0
        if ckpt_path:
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
            raw_model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            if 'scaler' in ckpt:
                scaler.load_state_dict(ckpt['scaler'])
            start_step = int(ckpt['step']) + 1
            if master:
                print(f'Resumed {ckpt_path}, next step={start_step}')
                if ckpt.get('world_size', world_size) != world_size:
                    print('WARNING: world_size changed; data/optimizer trajectory may differ')
                print('NOTE: DataLoader cursor and per-rank RNG are not restored; this is not exact replay.')
        if start_step >= max_steps:
            if master:
                print(f'Already completed {start_step} steps; max_steps={max_steps}')
            return

        if ddp:
            model = DDP(raw_model, device_ids=[local_rank])
        model.train()
        if master:
            if not os.path.exists(log_path) or os.path.getsize(log_path) == 0:
                with open(log_path, 'w', encoding='utf-8') as f:
                    f.write('step,train_loss,val_loss,lr,grad_norm,dt_ms,tokens_per_sec\n')
            print(f'device={device}, ddp={ddp}, world_size={world_size}, amp={use_amp}, '
                  f'grad_accum_steps={grad_accum_steps}')

        for step in range(start_step, max_steps):
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss_sum = torch.zeros((), device=device, dtype=torch.float64)

            for micro_step in range(grad_accum_steps):
                x, y = train_loader.next_batch()
                x, y = x.to(device), y.to(device)
                sync_context = (model.no_sync() if ddp and micro_step < grad_accum_steps - 1
                                else nullcontext())
                with sync_context:
                    with torch.autocast(device_type='cuda', enabled=use_amp):
                        _, loss = model(x, y)
                        scaled_mean_loss = loss / grad_accum_steps
                    loss_sum += loss.detach().double() / grad_accum_steps
                    scaler.scale(scaled_mean_loss).backward()

            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            lr = get_lr(step, max_lr, min_lr, warmup_steps, max_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = lr
            scaler.step(optimizer)
            scaler.update()

            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            dt = time.perf_counter() - t0
            if ddp:
                dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
                loss_sum /= world_size
            train_loss = loss_sum.item()
            tokens_per_sec = total_batch_size / dt
            val_loss = float('nan')

            if do_eval and (step % eval_interval == 0 or step == max_steps - 1):
                if ddp:
                    dist.barrier()
                if master:
                    val_loss = evaluate(raw_model, val_loader, eval_steps, device, use_amp)
                    print(f'[eval] step {step}, val_loss {val_loss:.4f}')
                if ddp:
                    dist.barrier()

            if master and (step % log_interval == 0 or step == max_steps - 1):
                print(f'step {step:5d} | train_loss {train_loss:.4f} | '
                      f'val_loss {val_loss:.4f} | lr {lr:.2e} | '
                      f'norm {norm.item():.3f} | dt {dt*1000:.1f}ms | '
                      f'tok/s {tokens_per_sec:,.0f}')
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(f'{step},{train_loss:.6f},{val_loss:.6f},{lr:.6e},'
                            f'{norm.item():.4f},{dt*1000:.2f},{tokens_per_sec:.2f}\n')

            if step > 0 and step % save_interval == 0:
                if ddp:
                    dist.barrier()
                if master:
                    path = os.path.join(out_dir, f'ckpt_{step:06d}.pt')
                    save_checkpoint(path, step, raw_model, optimizer, scaler, cfg, rank, world_size)
                    print(f'saved checkpoint: {path}')
                if ddp:
                    dist.barrier()

        if ddp:
            dist.barrier()
        if master:
            path = os.path.join(out_dir, 'ckpt_final_1.pt')
            save_checkpoint(path, max_steps - 1, raw_model, optimizer, scaler, cfg, rank, world_size)
            print(f'training done; final checkpoint: {path}')
        if ddp:
            dist.barrier()
    finally:
        if ddp and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
