# from collections import defaultdict
# import random

# # 训练文本
# texts = ["你好", "你好", "你好啊", "你好你"]

# # 统计 bigram
# counts = defaultdict(lambda: defaultdict(int))

# for text in texts:
#     chars = list(text)
#     for a, b in zip(chars, chars[1:]):
#         counts[a][b] += 1

# # 根据当前字，生成下一个字
# def predict_next(char):
#     options = counts.get(char)
#     if not options:
#         # 没见过这个字，当场“失忆”，重新开始
#         char = random.choice(list(counts.keys()))
#         options = counts[char]

#     total = sum(options.values())
#     r = random.randint(1, total)
#     s = 0
#     for ch, c in options.items():
#         s += c
#         if s >= r:
#             return ch


# # 测试“对话”
# cur = "你"
# result = cur
# for _ in range(10):
#     nxt = predict_next(cur)
#     result += nxt
#     cur = nxt

# print(result)


# from collections import defaultdict
# import random

# # 训练文本
# with open("train.txt", "r", encoding="utf-8") as f:
#     texts = [line.strip() for line in f if line.strip()]


# # trigram 统计
# counts = defaultdict(lambda: defaultdict(int))

# for text in texts:
#     chars = list(text)
#     for a, b, c in zip(chars, chars[1:], chars[2:]):
#         counts[(a, b)][c] += 1

# # 预测下一个字
# def predict_next(prev2):
#     options = counts.get(prev2)
#     if not options:
#         # 兜底：随机选一个已知状态
#         prev2 = random.choice(list(counts.keys()))
#         options = counts[prev2]

#     total = sum(options.values())
#     r = random.randint(1, total)
#     s = 0
#     for ch, c in options.items():
#         s += c
#         if s >= r:
#             return ch

# # 初始化
# start = random.choice(list(counts.keys()))
# result = [start[0], start[1]]

# # 生成文本
# for _ in range(50):
#     nxt = predict_next((result[-2], result[-1]))
#     result.append(nxt)

# print("".join(result))


# import torch
# import torch.nn as nn
# import random
# import torch.nn.functional as F

# # 1. 读取训练文本，保留换行
# with open("train.txt", "r", encoding="utf-8") as f:
#     text = f.read()  # 直接读，不去掉换行

# # 打印信息
# print(f"训练文本总长度（字符数）：{len(text)}")
# print(f"训练文本总行数：{len(text.splitlines())}")
# vocab = sorted(list(set(text)))
# print(f"训练文本独立字符数（vocab size）：{len(vocab)}")

# chars = sorted(set(text))
# stoi = {ch: i for i, ch in enumerate(chars)}
# itos = {i: ch for ch, i in stoi.items()}
# vocab_size = len(chars)

# # 2. 转成 ID
# data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)

# # 3. 构造 (当前字 → 下一个字) 训练对
# N = 15  # 上下文长度
# x = []
# y = []

# for i in range(len(data) - N):
#     x.append(data[i:i+N])  # 前 N 个字符作为输入
#     y.append(data[i+N])    # 第 N+1 个字符作为输出

# x = torch.stack(x)  # shape: (样本数, N)
# y = torch.tensor(y) # shape: (样本数,)

# # 4. 定义模型
# class TinyMLPLM(nn.Module):
#     def __init__(self, vocab_size, embed_dim=32, context_size=15, hidden_dim=256):
#         super().__init__()
#         self.context_size = context_size

#         # 1️⃣ embedding
#         self.embed = nn.Embedding(vocab_size, embed_dim)

#         # 2️⃣ MLP（核心升级点）
#         self.fc1 = nn.Linear(embed_dim * context_size, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, vocab_size)

#     def forward(self, x):
#         # x: (batch, context_size)
#         x = self.embed(x)                  # (batch, context, embed)
#         x = x.view(x.size(0), -1)          # ✅ 保留顺序，直接展平
#         x = F.relu(self.fc1(x))            # ✅ 非线性
#         logits = self.fc2(x)               # (batch, vocab)
#         return logits
    
# model = TinyMLPLM(
#     vocab_size=vocab_size,
#     embed_dim=32,
#     context_size=15,
#     hidden_dim=256
# )

# loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

# # 5. 训练
# for epoch in range(45):
#     logits = model(x)
#     loss = loss_fn(logits, y)

#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     if epoch % 5 == 0:
#         print(f"epoch {epoch}, loss {loss.item():.4f}")


# # 6. 生成文本
# def generate(start_chars, length=300, temperature=1.0):
#     model.eval()
#     result = list(start_chars)
#     context_size = model.context_size

#     # 用第一个字符做 padding（也可以用空格、换行等）
#     pad_char = result[0]

#     for _ in range(length):
#         context = result[-context_size:]

#         # ⭐ 不足 context_size 就左补齐
#         if len(context) < context_size:
#             context = [pad_char] * (context_size - len(context)) + context

#         idx = torch.tensor([[stoi[ch] for ch in context]])
#         logits = model(idx)
#         probs = torch.softmax(logits / temperature, dim=-1)
#         next_idx = torch.multinomial(probs, 1).item()
#         result.append(itos[next_idx])

#     return "".join(result)

# print(generate("你", temperature=1.0)[:100])


import torch
import torch.nn as nn
import random
import torch.nn.functional as F

import jieba

from collections import Counter

import time
from datetime import datetime

# 1. 读取训练文本，保留换行


with open("train_en.txt", "r", encoding="utf-8") as f:
    text = f.read().lower()

tokens = list(jieba.cut(text))

print("词数：", len(tokens))
# print("示例：", tokens[:30])

# 打印信息
print(f"训练文本总长度（字符数）：{len(text)}")
print(f"训练文本总行数：{len(text.splitlines())}")
vocab = sorted(list(set(text)))
print(f"训练文本独立字符数(vocab size)：{len(vocab)}")



word_counts = Counter(tokens)
min_freq = 2
vocab = ["<UNK>"] + [w for w, c in word_counts.items() if c >= min_freq]
stoi = {w: i for i, w in enumerate(vocab)}
data = torch.tensor(
    [stoi.get(w, stoi["<UNK>"]) for w in tokens],
    dtype=torch.long
)

itos = {i: w for w, i in stoi.items()}
vocab_size = len(vocab)
print("词表大小:", vocab_size)

# 2. 转成 ID
data = torch.tensor([stoi.get(w, stoi["<UNK>"]) for w in tokens], dtype=torch.long)

# 3. 构造 (当前字 → 下一个字) 训练对
context_size = 20

x, y = [], []
for i in range(len(data) - context_size):
    x.append(data[i:i+context_size])
    y.append(data[i+context_size])

x = torch.stack(x)
y = torch.tensor(y)

# 4. 定义模型
class WordLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embed(x)                  # (batch, seq, embed)
        out, hidden = self.lstm(x, hidden) # 接收并返回 hidden
        out = out[:, -1, :]
        logits = self.fc(out)
        return logits, hidden

model = WordLSTM(vocab_size)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

# 5. 训练
for epoch in range(30):
    logits, _ = model(x)
    loss = loss_fn(logits, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 5 == 0:
        print(f"epoch {epoch}, loss {loss.item():.4f}, [{datetime.now().strftime('%H:%M:%S')}]")

checkpoint = {
    "model_state": model.state_dict(),
    "stoi": stoi,
    "itos": itos,
    "vocab_size": vocab_size,
    "context_size": context_size,
    "embed_dim": 128,
    "hidden_dim": 256,
}

torch.save(checkpoint, "crystallm_wordlstm.pt")
print("✅ 模型已保存")



# 6. 生成文本
def generate(start_text, length=100, temperature=1.0):
    model.eval()

    # 1. 起始文本 → 词
    start_tokens = list(jieba.cut(start_text))
    result = start_tokens.copy()

    hidden = None

    # 2. 先把起始词喂进模型，建立 hidden state
    for w in start_tokens[:-1]:
        idx = torch.tensor([[stoi.get(w, stoi["<UNK>"])]])
        _, hidden = model(idx, hidden)

    cur_word = start_tokens[-1]

    # 3. 正式生成
    for _ in range(length):
        idx = torch.tensor([[stoi.get(cur_word, stoi["<UNK>"])]])
        logits, hidden = model(idx, hidden)

        probs = torch.softmax(logits / temperature, dim=-1)
        next_idx = torch.multinomial(probs, 1).item()

        cur_word = itos[next_idx]
        result.append(cur_word)

    # 4. 词 → 文本
    return "".join(result)



print(generate("你", temperature=1.0))
