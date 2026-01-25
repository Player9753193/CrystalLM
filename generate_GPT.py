import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *

# ======================
# 载入 checkpoint
# ======================
ckpt = torch.load(SAVE_PATH, map_location=DEVICE)
stoi = ckpt["stoi"]
itos = ckpt["itos"]
context_size = ckpt["context"]
vocab_size = len(itos)

BOS_ID = stoi[BOS]
EOS_ID = stoi[EOS]

# ======================
# GPT 模型
# ======================
class GPTBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, HEADS, dropout=DROPOUT, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, FF_DIM),
            nn.GELU(),
            nn.Linear(FF_DIM, dim)
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
        self.pos = nn.Embedding(context_size, EMBED_DIM)
        self.blocks = nn.ModuleList([GPTBlock(EMBED_DIM) for _ in range(LAYERS)])
        self.ln_f = nn.LayerNorm(EMBED_DIM)
        self.head = nn.Linear(EMBED_DIM, vocab_size, bias=False)
        self.head.weight = self.token.weight

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        x = self.token(x) + self.pos(pos)
        mask = torch.triu(torch.ones(T, T, device=x.device) * float("-inf"), diagonal=1)
        for blk in self.blocks:
            x = blk(x, mask)
        return self.head(self.ln_f(x))

model = GPT().to(DEVICE)
model.load_state_dict(ckpt["model"])
model.eval()

# ======================
# 生成函数
# ======================
@torch.no_grad()
def generate(prompt: str):
    ids = [BOS_ID] + [stoi[c] for c in prompt if c in stoi]

    for _ in range(MAX_NEW_TOKENS):
        x = torch.tensor([ids[-context_size:]], device=DEVICE)
        logits = model(x)[:, -1] / max(0.1, TEMPERATURE)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=100.0, neginf=-100.0)
        probs = torch.softmax(logits, dim=-1)

        # TOP-P
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_probs[cum_probs > TOP_P] = 0.0

        # fallback，如果 TOP-P 屏蔽全零
        if sorted_probs.sum() == 0:
            next_id = torch.multinomial(probs, 1).item()
        else:
            sorted_probs /= sorted_probs.sum()
            next_id = sorted_idx[0, torch.multinomial(sorted_probs, 1)].item()

        if next_id == EOS_ID:
            break
        ids.append(next_id)

    return "".join(itos[i] for i in ids if i > 2)

# ======================
# CLI
# ======================
print("CrystalLM GPT Generator")
while True:
    s = input(">>> ")
    if s in {"exit", "quit"}:
        break
    print(generate(s))
    print("-" * 40)