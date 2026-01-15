import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import jieba
from collections import Counter
import time
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# 1. 读取训练文本，保留换行
with open("train.txt", "r", encoding="utf-8") as f:
    text = f.read().lower()

tokens = []
for line in text.splitlines():
    line = line.strip()
    if not line:
        continue
    tokens.extend(jieba.cut(line))
    tokens.append("<END>")


print("词数：", len(tokens))
# print("示例：", tokens[:30])

# 打印信息
print(f"训练文本总长度（字符数）：{len(text)}")
print(f"训练文本总行数：{len(text.splitlines())}")



COMMON_TOKENS = [
    "minecraft", "ai", "cpu", "gpu",
    "ctrl", "shift", "alt", "cmd",
    "+", "-", "*", "/", "=", "==",
    "(", ")", "[", "]", "{", "}",
    "（", "）", "【", "】", "「", "」",
    "《", "》", "<", ">", "?", "!",
    "@", "#", "$", "%", "^", "&",
    "*", "_", "¥", "、", "“", "”",
    "≠", "±", "：", "；", "‘", "’"
]

word_counts = Counter(tokens)
min_freq = 2
SPECIAL_TOKENS = ["<PAD>", "<END>", "<UNK>"]

vocab = SPECIAL_TOKENS + COMMON_TOKENS + [
    w for w, c in word_counts.items()
    if c >= min_freq and w not in COMMON_TOKENS
]
print("词表大小（含特殊符号）：", len(vocab))
stoi = {w: i for i, w in enumerate(vocab)}
data = torch.tensor(
    [stoi.get(w, stoi["<UNK>"]) for w in tokens],
    dtype=torch.long
)

itos = {i: w for w, i in stoi.items()}
vocab_size = len(vocab)
print("词表大小:", vocab_size)