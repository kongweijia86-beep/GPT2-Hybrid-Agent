#  GPT-2 From Scratch & Hybrid AI Agent

本项目是一个从底层到应用层的全链路大模型实战项目，包含两个核心模块：
1. **底层算法**：不依赖 HuggingFace 封装，纯 PyTorch 手写 GPT-2 Transformer 架构，并在云端完成预训练。
2. **应用工程**：基于自研的小模型与云端大模型（DeepSeek），手写 ReAct 循环，构建了一套“本地推理 + 云端兜底”的混合路由 Agent 架构。

---

##  模块一：GPT-2 从零预训练 (From Scratch)

- **模型架构**：纯手写 Causal Self-Attention、MLP、LayerNorm，实现权重共享（Weight Tying）与缩放初始化策略。
- **数据工程**：编写 `DataLoaderLite` 实现文本 Tokenization、流式加载以及 `(x, y)` 错位切片（`x[:-1]`, `y[1:]`）。
- **云端训练**：在 Kaggle T4 GPU 环境下，基于 TinyStories 数据集（57M tokens）完成 5000 步预训练。
- **优化策略**：使用 Cosine 学习率衰减、Warmup 预热与梯度裁剪防止梯度爆炸。
- **训练成果**：验证集 Cross-Entropy Loss 降至 **2.00**，Perplexity 达 **7.42**。

**  Loss 收敛曲线**
> *<img width="579" height="378" alt="image" src="https://github.com/user-attachments/assets/526b88c7-4fa0-438e-a45f-34e52d3f580f" />
*

---

##  模块二：本地与云端混合 Agent (Hybrid Agent)

- **痛点解决**：本地自研的 10M 小模型缺乏代码生成能力，而直接调用云端大模型 API 成本较高。
- **架构设计**：手写 Intent Router（意图路由）。将“童话故事生成”路由给本地 GPT-2（零成本）；将“代码生成”路由给云端 DeepSeek。
- **自我修正 (Self-Correction)**：手写 ReAct 循环，利用 `subprocess` 在本地沙盒执行代码。若本地模型生成的代码报错，Agent 会自动捕获 `stderr`，并将报错信息与上下文抛给云端 DeepSeek 进行“救援修复”。

** Agent 运行演示**
> *<img width="1020" height="827" alt="image" src="https://github.com/user-attachments/assets/6139ead9-3af9-45f0-aa4a-fb1f3f17d1f2" />
<img width="710" height="522" alt="image" src="https://github.com/user-attachments/assets/9a190965-22bf-4c22-981a-b892d135fa5d" />
*

---

## 🚀 快速开始 (Quick Start)

### 1. 安装依赖
```bash
pip install -r requirements.txt
