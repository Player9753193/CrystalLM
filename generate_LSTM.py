import torch
import torch.nn as nn
import jieba

# ===== 模型定义（必须和训练时一模一样）=====
class WordLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embed(x)
        out, hidden = self.lstm(x, hidden)
        out = out[:, -1, :]
        logits = self.fc(out)
        return logits, hidden


# ===== 加载 checkpoint =====
ckpt = torch.load("crystallm_wordlstm.pt", map_location="cpu")

stoi = ckpt["stoi"]
itos = ckpt["itos"]
vocab_size = ckpt["vocab_size"]

model = WordLSTM(
    vocab_size=vocab_size,
    embed_dim=ckpt["embed_dim"],
    hidden_dim=ckpt["hidden_dim"]
)
model.load_state_dict(ckpt["model_state"])
model.eval()

print("✅ 模型加载完成")

def generate(start_text, length=100, temperature=1.0):
    model.eval()
    start_tokens = list(jieba.cut(start_text))
    result = start_tokens.copy()

    hidden = None

    with torch.no_grad():
        for w in start_tokens[:-1]:
            idx = torch.tensor([[stoi.get(w, stoi["<UNK>"])]])
            _, hidden = model(idx, hidden)

        cur_word = start_tokens[-1]

        for _ in range(length):
            idx = torch.tensor([[stoi.get(cur_word, stoi["<UNK>"])]])
            logits, hidden = model(idx, hidden)

            probs = torch.softmax(logits / temperature, dim=-1)
            next_idx = torch.multinomial(probs, 1).item()

            cur_word = itos[next_idx]
            result.append(cur_word)

    return "".join(result)

# print(generate("minecraft", temperature=1.0))
print(generate("人生", temperature=1.0))
print(generate("科学", temperature=1.0))
print(generate("未来", temperature=1.0))
# print(generate("技术", temperature=0.8))
# print(generate("文明", temperature=0.8))
# print(generate("宇宙", temperature=0.8))
# print(generate("时间", temperature=0.8))
# print(generate("生命", temperature=0.8))