import numpy as np
import subprocess

# 난수 시드 설정
seed = 42
np.random.seed(seed)

# 입력 데이터 생성
batch_size = 2
seq_len = 10
d_model = 6
query = np.random.rand(batch_size, seq_len, d_model).astype(np.float32)
key = np.random.rand(batch_size, seq_len, d_model).astype(np.float32)
value = np.random.rand(batch_size, seq_len, d_model).astype(np.float32)

# 입력 데이터를 파일로 저장
np.savetxt("./data/query_input.txt", query.flatten(), fmt="%.6f")
np.savetxt("./data/key_input.txt", key.flatten(), fmt="%.6f")
np.savetxt("./data/value_input.txt", value.flatten(), fmt="%.6f")

# C 프로그램 실행
subprocess.run(["gcc", "03_Attention.c", "-o", "attention"])
subprocess.run(["attention.exe"], check=True)

# C 프로그램 출력 읽기
c_output = np.loadtxt("./data/c_output.txt").reshape(batch_size, seq_len, d_model)

# Python 프로그램 실행
import torch
import math
from torch import nn
from torch.nn import functional as F

def clones(module, N):
    from copy import deepcopy
    return nn.ModuleList([deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

# Python Multi-Head Attention 실행
multi_head_attn = MultiHeadedAttention(h=2, d_model=d_model, dropout=0.1)
query_torch = torch.tensor(query)
key_torch = torch.tensor(key)
value_torch = torch.tensor(value)
output = multi_head_attn(query_torch, key_torch, value_torch)

python_output = output.detach().numpy()

# 출력 비교
print("C Output:\n", c_output)
print("Python Output:\n", python_output)
comparison = np.allclose(c_output, python_output, atol=1e-6)
print("Outputs match:", comparison)
