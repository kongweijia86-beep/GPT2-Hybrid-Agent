import os
import subprocess
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"
)


# ==========================================
# 极简版模型定义（完全匹配你刚下载的权重）
# ==========================================
@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 50257
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([nn.Linear(config.n_embd, config.n_embd) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.wpe(pos) + self.transformer.wte(idx)
        for block in self.transformer.h:
            x = x + block(x)
        logits = self.lm_head(self.transformer.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx


# ==========================================
# 加载本地权重
# ==========================================
print("加载本地 GPT-2 权重...")
current_dir = os.path.dirname(os.path.abspath(__file__))
ckpt_path = os.path.join(current_dir, "outputs", "gpt2-medium", "ckpt_final.pt")
ckpt = torch.load(ckpt_path, map_location="cpu")

gpt2_model = GPT(GPTConfig())
gpt2_model.load_state_dict(ckpt["model"])
gpt2_model.eval()
enc = tiktoken.get_encoding("gpt2")


# ==========================================
# Agent 核心逻辑
# ==========================================
def run_code(code_str):
    with open("temp_script.py", "w", encoding="utf-8") as f:
        f.write(code_str)
    try:
        result = subprocess.run(["python", "temp_script.py"], capture_output=True, text=True, timeout=5)
        return result.stdout if result.returncode == 0 else result.stderr
    except Exception as e:
        return str(e)


def hybrid_agent(user_request):
    print(f"\n👤 用户需求: {user_request}\n")
    print("🧠 [阶段 1] 本地 GPT-2 正在尝试生成代码...")

    prompt = f"Write Python code for: {user_request}\nCode:\n"
    idx = torch.tensor(enc.encode(prompt), dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        out = gpt2_model.generate(idx, max_new_tokens=100, temperature=0.8, top_k=50)

    gpt2_code = enc.decode(out[0].tolist()).replace(prompt, "").strip()
    print(f"GPT-2 生成的代码:\n{gpt2_code}\n")

    exec_result = ""
    if len(gpt2_code) < 10:
        print("❌ 本地 GPT-2 没能生成有效代码，直接触发救援机制。")
        exec_result = "Empty code generated."
    else:
        print("⚙️ 正在本地执行...")
        exec_result = run_code(gpt2_code)
        print(f"执行结果: {exec_result}\n")
        if "Traceback" not in exec_result and "Error" not in exec_result and len(exec_result) > 0:
            print("✅ 奇迹！本地 GPT-2 一次跑通！")
            return gpt2_code

    print("🚨 触发 Agent 兜底策略，呼叫云端 DeepSeek 进行救援修复...")
    messages = [
        {"role": "system",
         "content": "你是一个高级 Python 程序员。请输出纯粹的可执行 Python 代码，不要任何 Markdown 格式。"},
        {"role": "user",
         "content": f"用户需求：{user_request}\n\n之前的小模型尝试了这段代码但报错了：\n{gpt2_code}\n\n报错信息：\n{exec_result}\n\n请修复它并直接输出完整代码。"}
    ]
    response = client.chat.completions.create(
        model="deepseek-coder",
        messages=messages,
        temperature=0.1
    )
    ds_code = response.choices[0].message.content.strip()
    if ds_code.startswith("```python"): ds_code = ds_code[9:-3].strip()
    print(f"DeepSeek 修复后的代码:\n{ds_code}\n")
    print("⚙️ 再次执行...")
    final_result = run_code(ds_code)
    print(f"执行结果: {final_result}\n")
    if "Traceback" not in final_result and "Error" not in final_result and len(final_result) > 0:
        print("✅ DeepSeek 救援成功！")


if __name__ == "__main__":
    hybrid_agent("写一个函数计算 1 到 100 的累加，并打印结果")