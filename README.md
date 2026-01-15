# English

# CrystalLM ✨

A Chinese **Word-level** Language Model Implemented from Scratch (GRU-based)

CrystalLM is an **educational and experimental** Chinese language modeling project.
Starting from the most fundamental text processing, it incrementally implements:

* Chinese word segmentation (jieba)
* Vocabulary construction, pruning, and governance (`<UNK>` mechanism)
* Word-level recurrent language model (**GRU**)
* Text generation with temperature / sampling control
* Model checkpoint saving and loading (directly reusable for inference)

This project **intentionally avoids high-level wrapper frameworks**.
All core logic is written from scratch, making it ideal for understanding the *entire lifecycle* of a language model—from raw text to generated output.

---

## ✨ Feature Overview

* 📖 **Word-level language modeling (Chinese)**
* 🧠 **GRU-based sequence modeling (2–3 layers supported)**
* ✂️ **Vocabulary pruning + `<UNK>` fallback**
* 🔥 **Temperature / top-p style generation control**
* 💾 **Model checkpoint save & load**
* 🧪 Designed for learning, inspection, and modification

---

## 📂 Recommended Project Structure

```
CrystalLM/
├── train.py               # Training script
├── generate.py            # Load model & generate text
├── model.py               # Word-level GRU model definition
├── train.txt              # Training corpus
├── crystallm_gru.pt       # Trained model (optional)
└── README.md
```

> Using a single-file script is perfectly acceptable at early stages.
> Modularization can be done later when the design stabilizes.

---

## ⚙️ Environment Dependencies

* Python 3.10+
* PyTorch
* jieba

Install dependencies:

```bash
pip install torch jieba
```

---

## 🚀 Quick Start

### 1️⃣ Train the Model

Prepare a Chinese corpus in `train.txt`, then run:

```bash
python train.py
```

After training, a model checkpoint will be generated, for example:

```text
crystallm_gru.pt
```

---

### 2️⃣ Generate Text (No Retraining Required)

```bash
python generate.py
```

Or invoke generation directly in code:

```python
print(generate("你", temperature=0.8))
```

---

## 🧠 Model Specifications

* **Model Type**: Word-level GRU Language Model
* **Embedding Dim**: ~128
* **Hidden Dim**: ~256
* **GRU Layers**: 2–3
* **Context Window**: ~50 tokens (configurable)
* **Loss Function**: CrossEntropyLoss
* **Optimizer**: Adam

During training, loss converges steadily.
Generated text gradually reflects recognizable Chinese structure and rhythm.

---

## 📌 About `<UNK>` (Vocabulary Governance)

To control vocabulary size and training stability, low-frequency tokens are mapped to `<UNK>`.

This is a **standard and necessary practice** in classical language modeling.

To reduce `<UNK>` frequency:

* Increase corpus size
* Adjust `min_freq`
* Actively clean low-quality or noisy tokens

CrystalLM treats vocabulary management as a *first-class modeling concern*, not an afterthought.

---

## 🎛️ Temperature Explanation

* `temperature < 1.0` → Conservative, stable output
* `temperature = 1.0` → Neutral sampling
* `temperature > 1.0` → More diverse, higher entropy

Example:

```python
generate("你", temperature=0.7)
generate("你", temperature=1.2)
```

---

## 🎯 Project Goal

CrystalLM does **not** aim for SOTA performance.

Its purpose is explicit and pragmatic:

> To walk through *how a language model actually works*,
> using readable code and explicit design trade-offs.

Best suited for:

* NLP / deep learning beginners
* Learners transitioning from RNNs to Transformers
* Anyone who wants to truly understand *vocabulary, context, and generation*

---

## 📢 Why CrystalLM Is a Serious Educational Project

CrystalLM is not a toy project, nor a demo assembled from black-box APIs.

It is a **deliberately constrained, explicitly designed educational system**.

In an era dominated by large-scale Transformers and opaque training pipelines, CrystalLM takes a different stance:

> Understanding precedes scaling.
> Clarity precedes performance.

### What makes CrystalLM *serious*:

* **End-to-end transparency**
  Every stage—from tokenization to vocabulary pruning, from sequence modeling to sampling—is explicit and inspectable.

* **Intentional architectural choices**
  GRU is not used because it is “simpler”, but because it exposes:

  * temporal dependency modeling
  * hidden state dynamics
  * the real cost of context length
    …without abstraction leakage.

* **Vocabulary governance as a first-class problem**
  `<UNK>` handling, low-frequency pruning, and corpus quality are treated as modeling decisions, not preprocessing trivia.

* **No reliance on pretrained magic**
  All learning emerges from the provided data and code paths, making failure modes *observable and instructive*.

CrystalLM is designed for learners who want to **reason about language models**, not just run them.

It prioritizes:

* conceptual correctness over benchmark chasing
* reproducibility over scale
* understanding over illusion of intelligence

If modern LLMs are skyscrapers,
CrystalLM is the **structural engineering course** that explains why they stand.

---

## 🔮 Future Plans (Roadmap)

* [x] GRU-based implementation
* [ ] GRU vs LSTM comparative experiments
* [ ] Transformer / Mini-GPT variant
* [ ] CLI-based interactive generation
* [ ] Vocabulary statistics & evaluation logging

---

## 📜 License

MIT License

---

## 🙌 Acknowledgments

* PyTorch
* jieba Chinese word segmentation
* Everyone curious enough to build models from first principles

---

# 简体中文

# CrystalLM ✨

一个从零实现的中文**词级**语言模型（GRU）

CrystalLM 是一个**教学导向 + 实验导向**的中文语言模型项目。
它从最基础的文本处理开始，完整实现了：

* 中文分词（jieba）
* 词表构建、裁剪与治理（`<UNK>` 机制）
* 基于 **GRU** 的词级语言模型
* 可控文本生成（temperature / 采样策略）
* 模型 checkpoint 保存与直接复用

本项目**刻意不依赖高阶封装框架**，
核心逻辑全部手写，适合系统性理解语言模型的工作原理。

---

## ✨ 特性一览

* 📖 **中文词级建模**
* 🧠 **GRU 序列模型（支持多层）**
* ✂️ **低频词裁剪 + `<UNK>` 兜底**
* 🔥 **Temperature / 采样风格控制**
* 💾 **模型保存 / 加载**
* 🧪 适合学习、拆解与二次开发

---

## 📂 项目结构（推荐）

```
CrystalLM/
├── train.py               # 训练脚本
├── generate.py            # 加载模型并生成文本
├── model.py               # 词级 GRU 模型定义
├── train.txt              # 训练语料
├── crystallm_gru.pt       # 已训练模型（可选）
└── README.md
```

> 当前使用单文件结构也完全没问题，
> 在模型稳定后再拆分是更务实的路线。

---

## ⚙️ 环境依赖

* Python 3.10+
* PyTorch
* jieba

```bash
pip install torch jieba
```

---

## 🚀 快速开始

### 1️⃣ 训练模型

准备 `train.txt`（中文文本），运行：

```bash
python train.py
```

训练结束后会生成模型文件，例如：

```text
crystallm_gru.pt
```

---

### 2️⃣ 生成文本（无需重新训练）

```bash
python generate.py
```

或在代码中调用：

```python
print(generate("你", temperature=0.8))
```

---

## 🧠 模型说明

* **模型类型**：Word-level GRU Language Model
* **Embedding 维度**：约 128
* **Hidden 维度**：约 256
* **GRU 层数**：2–3 层
* **上下文窗口**：约 50 个词（可调）
* **损失函数**：CrossEntropyLoss
* **优化器**：Adam

随着训练推进，loss 稳定下降，
生成文本逐渐呈现出中文的节奏与结构。

---

## 📌 关于 `<UNK>`（词表治理）

为控制词表规模并提升训练稳定性，
低频或噪声词会被映射为 `<UNK>`。

这是**经典语言模型中的标准做法**，而非缺陷。

降低 `<UNK>` 的方式包括：

* 扩充训练语料
* 调整 `min_freq`
* 主动清理异常与低质量词条

在 CrystalLM 中，**词表治理是核心设计之一**。

---

## 🎛️ Temperature 说明

* `temperature < 1.0`：更稳、更保守
* `temperature = 1.0`：标准采样
* `temperature > 1.0`：更发散、更具随机性

```python
generate("你", temperature=0.7)
generate("你", temperature=1.2)
```

---

## 🎯 项目目标

CrystalLM 的目标**从来不是 SOTA**，而是：

> 用尽可能透明的代码，
> 走完一遍「语言模型是如何工作的」。

非常适合：

* NLP / 深度学习初学者
* 从 RNN 过渡到 Transformer 的学习者
* 想真正理解“词表、上下文、生成机制”的人

---

## 📢 为什么 CrystalLM 是一个“严肃的”教学项目

CrystalLM 不是玩具，也不是拼接 API 的演示脚本。

它是一个**有意识设限、设计目标明确的教学型语言模型系统**。

在当下 Transformer 与超大模型主导的语境中，CrystalLM 选择了一条不同的路线：

> 先理解，再扩展。
> 先透明，再性能。

### CrystalLM 的“严肃性”体现在：

* **端到端的可解释性**
  从分词、词表裁剪，到序列建模、采样生成，每一步都是显式、可追踪、可修改的。

* **有意识的架构取舍**
  使用 GRU 不是因为“简单”，
  而是因为它能真实暴露：

  * 时间依赖如何建模
  * hidden state 如何演化
  * 上下文长度的真实代价

* **将词表治理视为核心问题**
  `<UNK>`、低频词、语料噪声不是“预处理细节”，
  而是直接影响模型行为的建模决策。

* **拒绝预训练黑箱**
  模型能力完全来自数据与代码本身，
  因而所有失败模式都是可观察、可学习的。

CrystalLM 面向的是**想真正“想明白语言模型”的学习者**，
而不是只想“跑起来”的使用者。

它优先考虑：

* 概念正确性，而非榜单成绩
* 可复现性，而非规模崇拜
* 理解深度，而非智能幻觉

如果说现代大模型是摩天大楼，
那么 CrystalLM 就是**解释它们为何能站立的结构力学课**。

---

## 🔮 后续计划

* [x] GRU 实现
* [ ] GRU vs LSTM 对比实验
* [ ] Transformer / Mini-GPT
* [ ] CLI 交互式生成工具
* [ ] 更系统的评估与日志体系

---

## 📜 License

MIT License

---

## 🙌 致谢

* PyTorch
* jieba 中文分词
* 所有愿意从底层理解 NLP 的人

---

