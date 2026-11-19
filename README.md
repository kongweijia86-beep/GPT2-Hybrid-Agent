# GPT2 Hybrid Agent

**基于本地 GPT-2 与 DeepSeek API 的混合智能体系统**

## 1. 项目介绍

本项目旨在探索本地小模型与云端大模型协作的可行方案，针对本地模型生成质量不稳定、直接调用云端模型存在 API 成本的问题，设计并实现了一套 Hybrid Agent。

系统采用 **Local First + Quality Evaluation + Cloud Fallback** 的工作模式：

- 优先使用本地 GPT-2 完成文本生成任务。
- 通过质量评估机制判断本地生成结果是否达到预设标准。
- 对未通过质量评估的结果，调用 DeepSeek API 进行兜底生成。
- 通过实验记录评估生成质量、响应时间及 API 调用情况。

项目重点是探索本地模型与云端模型之间的任务分配策略，而不是单纯追求模型生成能力。

## 2. 系统架构

```text
             User Request
                   |
                   v
              Task Router
                   |
          +--------+--------+
          |                 |
        Story              Code
          |                 |
          v                 v
      Local GPT-2        DeepSeek
          |
          v
     Quality Evaluation
          |
     +----+----+
     |         |
   Passed   Rejected
     |         |
     v         v
   Return   DeepSeek API
               |
               v
          Final Output
```

## 3. 核心功能

### 3.1 本地模型生成

- 基于 GPT-2 Medium 架构构建本地语言模型。
- 使用 TinyStories 数据集进行故事生成训练。
- 支持 Temperature、Top-k、Top-p 等采样策略。
- 实现 EOS Token 终止机制。

### 3.2 多维度质量评估

针对儿童故事生成任务，建立五维质量评价体系：

| 评价维度 | 评价内容 |
|---|---|
| Relevance | 是否符合用户主题 |
| Coherence | 故事情节是否连贯 |
| Completeness | 是否具有完整的故事结构 |
| Language | 语言表达是否自然 |
| Diversity | 表达是否具有多样性 |

使用 DeepSeek 进行质量评价，根据评分结果决定是否采用本地生成结果。

### 3.3 混合模型路由

- 本地模型优先生成。
- 质量不达标时调用 DeepSeek 兜底。
- 对代码生成任务使用 DeepSeek。
- 使用 Python AST 进行代码语法校验。

### 3.4 实验记录

记录以下指标：

- 本地模型接受率
- DeepSeek 兜底率
- 故事质量评分
- 本地生成耗时
- 云端生成耗时
- API 调用次数

使用 JSONL 记录实验过程，并通过 CSV 汇总实验结果。

## 4. 实验结果

### 4.1 实验设置

- 测试主题：20 个
- 每个主题生成次数：3 次
- 总测试次数：60 次
- 本地模型：GPT-2 Medium
- 数据集：TinyStories
- 云端模型：DeepSeek

### 4.2 实验结果

| 指标 | 实验结果 |
|---|---:|
| 总测试次数 | 60 |
| 本地模型接受次数 | 15 |
| 本地模型接受率 | 25% |
| DeepSeek 兜底次数 | 45 |
| DeepSeek 兜底率 | 75% |
| 最终故事平均质量评分 | 8.28/10 |
| 本地平均生成耗时 | 19.76s |
| DeepSeek 兜底平均耗时 | 3.66s |
| API 调用次数 | 150 |

注：质量评分来自项目中的自动评价机制，并非独立人工盲评。实验结果仅代表当前模型、参数、测试集及评分标准下的表现。

## 5. 实验分析与不足

实验表明，本地 GPT-2 已能够独立完成部分故事生成任务，但在复杂主题遵循、情节连贯性及故事完整性方面仍存在不足。

当前主要问题：

1. 本地模型接受率较低，生成质量仍有提升空间。
2. CPU 环境下本地模型推理耗时较长。
3. 本地模型与云端模型之间仍存在明显的质量差异。
4. 当前实验规模有限，尚不足以证明该路由策略具有普遍优势。

可能的优化方向：

- 扩大训练数据规模，研究数据规模对生成质量的影响。
- 优化 Prompt 和采样参数。
- 探索模型推理加速方案。
- 优化质量评价标准与路由阈值。
- 增加不同主题和重复实验，进一步验证策略稳定性。

## 6. 技术栈

- **Programming:** Python
- **Deep Learning:** PyTorch
- **Language Model:** GPT-2 Medium
- **Dataset:** TinyStories
- **API:** DeepSeek API
- **Evaluation:** LLM-as-a-Judge
- **Code Validation:** Python AST
- **Data Analysis:** JSONL / CSV
- **Version Control:** Git / GitHub

## 7. 项目结构

```text
GPT2-Hybrid-Agent/
│
├── src/
│   ├── model.py
│   ├── train.py
│   ├── agent.py
│   ├── generate.py
│   └── analyze_generation.py
│
├── outputs/
│   └── gpt2-medium/
│
├── experiment_results/
│   ├── log.jsonl
│   └── summary.csv
│
├── .env.example
├── .gitignore
└── README.md
```

实际文件结构以仓库当前版本为准。

## 8. 运行方式

### 8.1 安装依赖

```bash
pip install torch tiktoken openai python-dotenv
```

### 8.2 配置 API Key

在项目根目录创建 `.env` 文件：

```text
DEEPSEEK_API_KEY=your_api_key
```

### 8.3 运行 Agent

```bash
python src/agent.py
```

运行前需要准备训练好的模型 checkpoint，并根据本地环境配置模型路径。

## 9. 后续计划

- 扩大 TinyStories 训练数据规模。
- 提升本地模型故事生成质量。
- 优化本地推理速度。
- 研究更合理的模型路由和质量评估机制。
- 对不同路由策略开展对照实验。

## 10. 项目地址

GitHub: https://github.com/kongweijia86-beep/GPT2-Hybrid-Agent

