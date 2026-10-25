"""
Data loading utilities: tokenize text, produce (x, y) batches.
Supports single-process and DDP-sharded loading.
"""
import os
import torch
import tiktoken

class DataLoaderLite:
    def __init__(self, B, T, process_rank=0, num_processes=1,
                 split="train", data_dir="data"):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        path = os.path.join(data_dir, f"{split}.txt")
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        enc = tiktoken.get_encoding("gpt2")
        # 【修复点】：加入 disallowed_special=() 防止 TinyStories 中的特殊 token 报错
        tokens = enc.encode(text, disallowed_special=())
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        print(f"[{split}] loaded {len(self.tokens):,} tokens from {path}")

        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T

        # 【修复点】：如果当前指针已经快到底部了，立刻重置到起点，防止切片长度不够导致 view 报错
        if self.current_position + B * T + 1 > len(self.tokens):
            self.current_position = self.B * self.T * self.process_rank

        buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        self.current_position += B * T * self.num_processes
        return x, y

    def reset(self):
        self.current_position = self.B * self.T * self.process_rank