import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import time

import generate_GPT as generate

# ======================
# 配置区
# ======================
TEXT_PATH = "train.txt"
SAVE_PATH = "crystallm_gpt_lowmem.pt"

CONTEXT_SIZE = 256
BATCH_SIZE = 16          # 小 batch 保持低内存
GRAD_ACCUM = 4           # 累积 4 个 batch，相当于 batch=64
EPOCHS = 10
LR = 3e-4

EMBED_DIM = 256          # 低内存版，减半
LAYERS = 4               # 低内存版
HEADS = 4
FF_DIM = 1024
DROPOUT = 0.1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BOS = "<|bos|>"
EOS = "<|eos|>"
PAD = "<|pad|>"

# ======================
# 读取文本 & tokenizer
# ======================
print("📖 Loading text...")
text = Path(TEXT_PATH).read_text(encoding="utf-8")

chars = sorted(list(set(text)))
specials = [PAD, BOS, EOS]

itos = specials + chars
stoi = {s: i for i, s in enumerate(itos)}
vocab_size = len(itos)
print("🔤 Vocab size:", vocab_size)

def encode(s):
    return [stoi[BOS]] + [stoi[c] for c in s] + [stoi[EOS]]

data = torch.tensor(
    [i for line in text.splitlines() for i in encode(line)],
    dtype=torch.long
)

# ======================
# Dataset
# ======================
class LMDataset(Dataset):
    def __len__(self):
        return len(data) - CONTEXT_SIZE - 1

    def __getitem__(self, idx):
        x = data[idx : idx + CONTEXT_SIZE]
        y = data[idx + 1 : idx + CONTEXT_SIZE + 1]
        return x, y

loader = DataLoader(
    LMDataset(),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,      # 避免额外内存
    pin_memory=False
)

# ======================
# GPT 模型
# ======================
class GPTBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, HEADS, dropout=DROPOUT, batch_first=True
        )
        self.ln2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, FF_DIM),
            nn.GELU(),
            nn.Linear(FF_DIM, dim),
            nn.Dropout(DROPOUT)
        )

    def forward(self, x, mask):
        h = self.ln1(x)
        attn, _ = self.attn(h, h, h, attn_mask=mask)
        x = x + attn
        x = x + self.ff(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token = nn.Embedding(vocab_size, EMBED_DIM)
        self.pos = nn.Embedding(CONTEXT_SIZE, EMBED_DIM)
        self.blocks = nn.ModuleList([GPTBlock(EMBED_DIM) for _ in range(LAYERS)])
        self.ln_f = nn.LayerNorm(EMBED_DIM)
        self.head = nn.Linear(EMBED_DIM, vocab_size, bias=False)
        self.head.weight = self.token.weight

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        x = self.token(x) + self.pos(pos)

        mask = torch.triu(
            torch.ones(T, T, device=x.device) * float("-inf"),
            diagonal=1
        )

        for blk in self.blocks:
            x = blk(x, mask)

        x = self.ln_f(x)
        return self.head(x)

model = GPT().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# ======================
# 训练
# ======================
scaler = torch.cuda.amp.GradScaler()  # 混合精度训练，降低显存

print("🚀 Training...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    start_time = time.time()
    optimizer.zero_grad()

    for step, (x, y) in enumerate(loader):
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        with torch.cuda.amp.autocast():
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

        scaler.scale(loss / GRAD_ACCUM).backward()

        if (step + 1) % GRAD_ACCUM == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item()

        if (step + 1) % 50 == 0:
            print(f"[{time.strftime('%H:%M:%S')}] step {step+1}, loss {loss.item():.4f}")

    epoch_time = time.time() - start_time
    print(f"Epoch {epoch+1}/{EPOCHS} | avg loss {total_loss/len(loader):.4f} | time {epoch_time:.1f}s")

# ======================
# 保存
# ======================
torch.save(
    {
        "model": model.state_dict(),
        "stoi": stoi,
        "itos": itos,
        "context": CONTEXT_SIZE
    },
    SAVE_PATH
)

print("✅ Saved to", SAVE_PATH)


# ======================
# 加载模型
# ======================
print(generate.DEVICE("你", 100))
