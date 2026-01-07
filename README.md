# English

# CrystalLM ✨  
A Chinese Word-level Language Model Implemented from Scratch 

CrystalLM is an **educational and experimental** Chinese language modeling project.  
Starting from the most fundamental text processing, it progressively implements:

- Chinese word segmentation (jieba)
- Vocabulary construction and pruning (<UNK> mechanism)
- Word-level LSTM language model
- Word-level GRU language model
- Text generation (temperature sampling)
- Model saving and loading (can be directly reused for generation)

This project **does not rely on high-level wrapper frameworks**. The core logic is entirely hand-written, making it suitable for learning and understanding the complete workflow of a language model.

---

## ✨ Feature Overview

- 📖 **Word-level Modeling**
- 🧠 **LSTM sequence modeling, supports hidden state passing**
- ✂️ **Low-frequency word pruning + `<UNK>` fallback**
- 🔥 **Temperature-controlled generation style**
- 💾 **Supports model checkpoint saving / loading**
- 🧪 Suitable for NLP / deep learning beginners to read and modify

---

## 📂 Recommended Project Structure

```
CrystalLM/
├── train.py          # Training script (train + save model)
├── generate.py       # Load model and generate text
├── model.py          # WordLSTM model definition
├── train.txt         # Training text
├── crystallm_wordlstm.pt  # Trained model (optional)
└── README.md
```

> If you are currently using a single script file, that's perfectly fine. You can split it later.

---

## ⚙️ Environment Dependencies

- Python 3.10+
- PyTorch
- jieba

Install dependencies:

```bash
pip install torch jieba
```

---

## 🚀 Quick Start

### 1️⃣ Train the Model

Prepare `train.txt` (Chinese text), then run:

```bash
python train.py
```

After training, a model file will be generated, for example:

```text
crystallm_wordlstm.pt
```

---

### 2️⃣ Generate Text (No Retraining Needed)

```bash
python generate.py
```

Or call directly in code:

```python
print(generate("你", temperature=0.8))
```

---

## 🧠 Model Specifications

* **Model Type**: Word-level LSTM Language Model
* **Embedding dim**: 128
* **Hidden dim**: 256
* **Context window**: 20 (configurable)
* **Loss Function**: CrossEntropyLoss
* **Optimizer**: Adam

During training, the loss will steadily decrease, and the generated text will gradually exhibit Chinese sentence structures and style.

---

## 📌 About `<UNK>`

To control vocabulary size, low-frequency words are mapped to `<UNK>`.
This is a **standard practice in language modeling**, effectively improving overall coherence.

To reduce `<UNK>` occurrences:
* Increase training corpus
* Or lower `min_freq`

---

## 🎛️ Temperature Explanation

* `temperature < 1.0`: More conservative and stable
* `temperature = 1.0`: Standard sampling
* `temperature > 1.0`: More diverse and "creative"

Example:

```python
generate("你", temperature=0.7)
generate("你", temperature=1.2)
```

---

## 🎯 Project Goal

The goal of CrystalLM is **NOT to pursue SOTA**, but rather:

> To walk through "how a language model works" with the clearest code possible.

It is especially suitable for:
* NLP / deep learning beginners
* Learners transitioning from LSTM to Transformer
* Those who want to truly understand "vocabulary, context, generation"

---

## 🔮 Future Plans (TODO)

* [ ] GRU version for comparison
* [ ] Transformer / Mini-GPT version
* [ ] CLI command-line generation tool
* [ ] More comprehensive evaluation and logging system

---

## 📜 License

MIT License

---

## 🙌 Acknowledgments

* PyTorch
* jieba Chinese word segmentation
* Everyone curious about NLP

---

# 简体中文

# CrystalLM ✨  
一个从零实现的中文词级语言模型（Word-level LSTM）

CrystalLM 是一个**教学向 + 实验向**的中文语言模型项目，  
从最基础的文本处理开始，逐步实现了：

- 中文分词（jieba）
- 词表构建与裁剪（<UNK> 机制）
- 词级 LSTM 语言模型
- 文本生成（temperature sampling）
- 模型保存与加载（可直接复用生成）

本项目**不依赖高阶封装框架**，核心逻辑全部手写，适合学习和理解语言模型的完整流程。

---

## ✨ 特性一览

- 📖 **词级建模（Word-level）**
- 🧠 **LSTM 序列建模，支持 hidden state 传递**
- ✂️ **低频词裁剪 + `<UNK>` 兜底**
- 🔥 **Temperature 控制生成风格**
- 💾 **支持模型 checkpoint 保存 / 加载**
- 🧪 适合 NLP / 深度学习初学者阅读与修改

---

## 📂 项目结构（推荐）

```

CrystalLM/
├── train.py          # 训练脚本（训练 + 保存模型）
├── generate.py       # 加载模型并生成文本
├── model.py          # WordLSTM 模型定义
├── train.txt         # 训练文本
├── crystallm_wordlstm.pt  # 训练好的模型（可选）
└── README.md

````

> 如果你目前还在用单文件脚本，也完全没问题，后续可再拆分。

---

## ⚙️ 环境依赖

- Python 3.10+
- PyTorch
- jieba

安装依赖：

```bash
pip install torch jieba
````

---

## 🚀 快速开始

### 1️⃣ 训练模型

准备好 `train.txt`（中文文本），然后运行：

```bash
python train.py
```

训练完成后会生成模型文件，例如：

```text
crystallm_wordlstm.pt
```

---

### 2️⃣ 生成文本（无需重新训练）

```bash
python generate.py
```

或在代码中直接调用：

```python
print(generate("你", temperature=0.8))
```

---

## 🧠 模型说明

* **模型类型**：Word-level LSTM Language Model
* **Embedding dim**：128
* **Hidden dim**：256
* **Context window**：20（可调）
* **损失函数**：CrossEntropyLoss
* **优化器**：Adam

训练过程中 loss 会稳定下降，生成文本逐渐呈现出中文句式与风格。

---

## 📌 关于 `<UNK>`

为控制词表规模，低频词会被映射为 `<UNK>`。
这是**语言模型中的标准做法**，可以有效提升整体连贯性。

如果希望减少 `<UNK>`：

* 增加训练语料
* 或降低 `min_freq`

---

## 🎛️ Temperature 说明

* `temperature < 1.0`：更保守、更稳定
* `temperature = 1.0`：标准采样
* `temperature > 1.0`：更发散、更有“创意”

示例：

```python
generate("你", temperature=0.7)
generate("你", temperature=1.2)
```

---

## 🎯 项目目标

CrystalLM 的目标**不是追求 SOTA**，而是：

> 用最清晰的代码，走完一遍「语言模型是如何工作的」。

它非常适合：

* NLP / 深度学习初学者
* 想从 LSTM 过渡到 Transformer 的学习者
* 想真正理解“词表、上下文、生成”的人

---

## 🔮 后续计划（TODO）

* [ ] GRU 版本对比
* [ ] Transformer / Mini-GPT 版本
* [ ] CLI 命令行生成工具
* [ ] 更完善的评估与日志系统

---

## 📜 License

MIT License

---

## 🙌 致谢

* PyTorch
* jieba 中文分词
* 所有对 NLP 抱有好奇心的人
