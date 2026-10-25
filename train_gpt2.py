from dataclasses import dataclass  #装饰器，不需要手动编写 __init__()，就可以创建对象
import math
import torch
import os
import torch.nn as nn
import time
import inspect
from torch.nn import functional as F
class CausalSelfAttention(nn.Module):          # ✅ Causal
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.MANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )  #注册一个 causal mask（因果掩码），作为模型的 buffer，用来在注意力里屏蔽未来 token。
    def forward(self, x):
        B, T, C = x.size()
        # ① QKV 投影：C → 3C
        qkv = self.c_attn(x)
        # ② 拆成 Q、K、V：3C → 3 个 C
        q, k, v = qkv.split(self.n_embd, dim=2)   # 各 (B, T, C)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  #交换是因为接下来要对每个注意力头分别计算序列中各个 token 之间的注意力。将头维度提前，可以让矩阵乘法对不同批次、不同头进行并行计算。
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)  #对最后一维做softmax
        # y = att @ v
        y = F.scaled_dot_product_attention(q,k,v,is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)   # 把形状为 (B, C, T) 的张量，先交换维度变成 (B, T, C)，再保证内存连续，最后整理成 (B, T, C) 的形状。
        y = self.c_proj(y)
        return y
class TanhGELU(nn.Module):
    def forward(self,input):
        return 0.5 * input * (1.0+torch.tanh(math.sqrt(2.0/math.pi)*(input+0.04471)*torch.pow(input,3.0)))
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)  #升维把每个 token 的表示映射到一个更高维的空间，让后续的非线性变换有更大的"表达空间"。
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.MANOGPT_SCALE_INIT = 1
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)    # ✅ Causal
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)                     # ✅ mlp
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))             # ✅ self.mlp
        return x
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),   #token embedding，把 token ID 变成向量
            wpe=nn.Embedding(config.block_size, config.n_embd),   #position embedding，把位置 ID 变成向量
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),  #n_layer 个 Transformer block
            ln_f=nn.LayerNorm(config.n_embd),#所有 Transformer block 处理完之后，先过一个 LayerNorm，再进 lm_head 输出 logits
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)  #这是输出层
        #weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight  #输入层的词嵌入矩阵（wte）和输出层的分类矩阵（lm_head）的维度是互逆的（一个是 vocab_size -> n_embd，一个是 n_embd -> vocab_size）。这里强制让它们共享同一个权重矩阵。能大幅减少模型参数量，并提高性能，因为语义相近的词在输入和输出空间中的表示应当是一致的，共享权重有助于模型学习这种一致性
        self.apply(self._init_weights)
    def _init_weights(self,module):
        if isinstance(module,nn.Linear):
            std = 0.02
            if hasattr(module,'MANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** (-0.5)  #根据方差传播公式，如果我们要让多次残差相加后的总方差保持不变，那么每次相加的方差需要缩小为原来的 1/根号N
            torch.nn.init.normal_(module.weight,mean=0.0,std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module,nn.Embedding):
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)

    def forward(self,idx,targets=None): #token ID序列idx
        B,T = idx.size()
        assert T <= self.config.block_size,f"Cannot forward sequence of length{T},block size is..."
        #创建位置编号
        pos = torch.arange(0,T,dtype=torch.long,device=idx.device)
        #位置embedding 形如(T,n_embd)
        pos_emb = self.transformer.wpe(pos)
        #token embedding 形如 (B,T,n_embd)
        tok_emb = self.transformer.wte(idx)
        x = pos_emb + tok_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss =None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1,logits.size(-1)),targets.view(-1))
        return logits,loss

    @classmethod
    def from_pretrained(cls, model_type):   #加载权重
        """loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt:%s" % model_type)
        config_args = {
            'gpt2':        dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large':  dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl':     dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = [k for k in sd.keys() if not k.endswith('.attn.bias')]
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)  #下载并返回hugging face的GPT-2模型
        sd_hf = model_hf.state_dict()

        sd_keys_hf = [
            k for k in sd_hf.keys()
            if not k.endswith('.attn.masked_bias')
               and not k.endswith('.attn.bias')
        ]  #attn.bias	作用->causal mask，下三角矩阵,attn.masked_bias	被 mask 的位置填充的常量但它们 不是可训练参数，只是前向计算用的固定张量。

        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight',
                      'mlp.c_fc.weight', 'mlp.c_proj.weight']

        assert len(sd_keys_hf) == len(sd_keys), \
            f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        #这四类权重在 HF 和你的实现里存储方向相反，所以复制前要转置
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):   # ✅ if
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:                                        # ✅ else
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    def configure_optimizers(self,weight_decay,learning_rate,device):
        param_dict = {pn:p for pn,p in self.named_parameters()}
        param_dict = {pn:p for pn,p in param_dict.items() if p.requires_grad}
        decay_params = [p for n,p in param_dict.items() if p.dim()>=2]
        nodecay_params = [p for n,p in param_dict.items() if p.dim() < 2] #二维权重矩阵做衰减，才是真正起到正则化、防止过拟合的作用。
        optim_groups = [
            {'params':decay_params,"weight_decay":weight_decay},
            {'params':nodecay_params,'weight_decay':0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors:{len(decay_params)},with{num_decay_params:,} parameters")
        print(f"num non_decayed parameter tensors:{len(nodecay_params)},with{num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f"using fused AdamW:{use_fused}")
        optimizer = torch.optim.AdamW(optim_groups,lr = learning_rate,betas=(0.9,0.95),eps=1e-8,
                                      **({'fused':True} if use_fused else{}))
        return optimizer

import tiktoken
class DataLoaderLite:
    def __init__(self,B,T,process_rank,num_processes):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        with open(r'input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        #state
        self.current_position = self.B *self.T * self.process_rank
    def next_batch(self):
        B,T =self.B,self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B,T)
        y = (buf[1:]).view(B,T)
        self.current_position += B*T*self.num_processes
        if self.current_position + (B*T+1) > len(self.tokens):
            self.current_position = 0
        return x,y
#attempt to autodetect the device
device = "cpu"
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends,"mps") and torch.backends.mps.is_available():
    device = 'mps'
print(f"using device {device}")
from torch.distributed import init_process_group,destroy_process_group
ddp = int(os.environ.get('RANK',-1)) != -1
if ddp:
    assert torch.cuda.is_available(),"for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)
total_batch_size = 524288
B = 8
T = 32
assert total_batch_size % (B*T) ==0,"make sure total_batch_size is divisible by B*T*ddp_world_size"
grad_accum_steps = total_batch_size // (B*T*ddp_world_size)
if master_process:
   print(f"total desired batch size:{total_batch_size}")
   print(f"-> calculated gradient accumulation steps:{grad_accum_steps}")
train_loader = DataLoaderLite(B=B,T=T,process_rank=ddp_rank,num_processes=ddp_world_size)
torch.set_float32_matmul_precision('high')
#create model
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
#model = torch.compile(model)
if ddp:
    model = DDP(model,device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps= 10 #热身步骤数
max_steps = 50#优化的最大步骤数
def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it+1)/warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it-warmup_steps) / (max_steps-warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 *(1.0+math.cos(math.pi*decay_ratio))
    return min_lr + coeff*(max_lr-min_lr)

#logits,loss = model(x,y)
#optimizer
#optimizer = torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(0.9,0.95),eps=1e-8)
optimizer = raw_model.configure_optimizers(weight_decay=0.1,learning_rate=6e-4,device=device)
for step in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x,y = train_loader.next_batch()
        x,y = x.to(device),y.to(device)
        # with torch.autocast(device_type=device,dtype=torch.bfloat16):
        logits,loss = model(x,y)
        loss = loss/grad_accum_steps
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        loss.backward()
    if ddp:
        dist.all_redue(loss_accum,op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    #torch.cuda.synchronize()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1-t0)
    tokens_processed = train_loader.B * train_loader.T*grad_accum_steps
    tokens_per_sec = tokens_processed/dt
    if master_process:
         print(f"step{step:4d},loss:{loss_accum.item():.6f},norm:{norm:.4f}dt:{dt*1000:.2f}ms")
if ddp:
    destroy_process_group()
import sys;sys.exit(0)





import sys;sys.exit(0)  #import sys; sys.exit(0) 是一行 Python 代码，作用是立即结束程序，退出码为 0。
num_return_sequences = 5 #一次生成5个不同的续写
max_length = 30  #最多生成30个token
model = GPT(GPTConfig())
model.eval()
model.to(device)
#prefix tokens
import tiktoken
enc = tiktoken.get_encoding('gpt2')  #获取GPT-2的BPE编码器
tokens = enc.encode("Hello,I'm a language model,")
tokens = torch.tensor(tokens,dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences,1)  #unsqueeze(0)在第零维加一个维度，（T，）变成（1，T）形式，repeat(num_return_sequences,1):沿第零维复制五次,(1,T)变成(5,T)
x = tokens.to(device)
torch.manual_seed(42)
torch.cuda.manual_seed(42)  #两个都要设：CPU 和 CUDA 各自有随机数生成器
while x.size(1) < max_length: #x.size(1)是当前序列长度
    #forward the model to get the logits
    with torch.no_grad():
        logits,_ = model(x)
        logits = logits[:,-1,:]  #只取最后一个位置的logits,因为要预测的是下一个token
        probs = F.softmax(logits,dim=1)
        topk_probs,topk_indices = torch.topk(probs,50,dim=-1)  #从最后一维里取概率最大的50个
        ix = torch.multinomial(topk_probs,1)   #从top-k概率里采样一个
        xcol = torch.gather(topk_indices,-1,ix)
        x = torch.cat((x,xcol),dim=1)
for i in range(num_return_sequences):
    tokens = x[i,:max_length].tolist()
    decoded = enc.decode(tokens)
    print(">",decoded)

