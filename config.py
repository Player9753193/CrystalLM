# config.py
import torch

# ======= 文件路径 =======
TEXT_PATH = "train_cn_small.txt"
SAVE_PATH = "crystallm_gpt.pt"

# ======= 模型参数 =======
VOCAB_SIZE = None        # None = 自动字符表
CONTEXT_SIZE = 24
EMBED_DIM = 128
LAYERS = 4
HEADS = 4
FF_DIM = 512
DROPOUT = 0.3

# ======= 训练参数 =======
BATCH_SIZE = 16
EPOCHS = 15
LR = 3e-4

# ======= 设备 =======
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======= 特殊 token =======
BOS = "<|bos|>"
EOS = "<|eos|>"
PAD = "<|pad|>"
SYS = "<|sys|>"
USR = "<|usr|>"
BOT = "<|bot|>"

# ======= 生成参数 =======
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.9
TOP_P = 0.9