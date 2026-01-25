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
    #tokens.append("<END>")
    tokens.append("\n")


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
min_freq = 3
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

# 2. 转成 ID
data = torch.tensor([stoi.get(w, stoi["<UNK>"]) for w in tokens], dtype=torch.long)

# 3. 使用 Dataset + DataLoader（省内存）
context_size = 64

class WordDataset(torch.utils.data.Dataset):
    def __init__(self, data, context_size):
        self.data = data
        self.context_size = context_size

    def __len__(self):
        return len(self.data) - self.context_size

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.context_size]
        y = self.data[idx + self.context_size]
        return x, y

dataset = WordDataset(data, context_size)

loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=128,          # ← 直接翻倍
    shuffle=True,
    drop_last=True,
    # num_workers=2,           # ← 不要 3，Mac 上 2 最稳
    # pin_memory=True
)

# 4. 定义模型
# 4. 定义 Transformer 模型
class WordTransformer(nn.Module):
    def __init__(self, vocab_size, 
                 embed_dim=128, 
                 num_heads=6, 
                 num_layers=4, 
                 ff_hidden=512, 
                 max_len=512, 
                 dropout=0.2):
        super().__init__()
        self.embed_dim = embed_dim
        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=stoi["<PAD>"])
        self.pos_embed = nn.Embedding(max_len, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_hidden,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        """
        x: (batch, seq_len)
        """
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = self.token_embed(x) + self.pos_embed(positions)

        # mask: Transformer 默认看整个序列，我们这里不需要 causal mask 因为训练时我们只预测下一词
        out = self.transformer(x)  # (batch, seq, embed_dim)
        # 取最后一个 token 的输出作为预测
        logits = self.fc(out[:, -1, :])
        return logits


model = WordTransformer(vocab_size).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 5. 训练
for epoch in range(12):
    model.train()
    total_loss = 0.0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        hidden = None
        logits, _ = model(xb)
        loss = loss_fn(logits, yb)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    if epoch % 1 == 0:
        print(
            f"epoch {epoch}, loss {total_loss / len(loader):.4f}, "
            f"[{datetime.now().strftime('%H:%M:%S')}]"
        )

checkpoint = {
    "model_state": model.state_dict(),
    "stoi": stoi,
    "itos": itos,
    "vocab_size": vocab_size,
    "context_size": context_size,
    "embed_dim": 128,
    "hidden_dim": 256,
}

torch.save(checkpoint, "crystallm_wordgru.pt")
print("✅ 模型已保存")

# 6. 生成文本
def generate(start_text, length=300, temperature=1.0):
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
        idx = torch.tensor([[stoi.get(cur_word, stoi["<UNK>"])]], device=device)
        logits, hidden = model(idx, hidden)
        probs = torch.softmax(logits / temperature, dim=-1)
        next_idx = torch.multinomial(probs, 1).item()

        cur_word = itos[next_idx]
        if cur_word == "<UNK>" and random.random() < 0.8:
            continue
        if cur_word == "<END>":
            break

        result.append(cur_word)

    # 4. 词 → 文本
    return "".join(result)

print(generate("来", temperature=1.0))
