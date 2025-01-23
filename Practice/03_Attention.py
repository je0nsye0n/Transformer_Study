import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from copy import deepcopy

torch.set_printoptions(precision=4, sci_mode=False)
torch.set_printoptions(linewidth=200)  # 한 줄의 최대 길이를 넉넉하게 설정

# Helper function to create clones of a module
def clones(module, N):
    return nn.ModuleList([deepcopy(module) for _ in range(N)])

# Attention function
def attention(query, key, value):
    d_k = query.size(-1)
    #print(query)
    #print(key)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # Scaled dot-product
    print(math.sqrt(d_k), d_k)
    attn_value = scores.softmax(dim=-1)
    return scores

# Multi-Head Attention class
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
            mask = mask.unsqueeze(1)  # Apply same mask to all heads
        nbatches = query.size(0)

        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

batch_size = 1
seq_len = 10
d_model = 6
h = 1

# Input initialization
query = torch.randn(batch_size, seq_len, d_model)  # (batch_size, seq_len, d_model)
key = torch.randn(batch_size, seq_len, d_model)    # Same shape as query
value = torch.randn(batch_size, seq_len, d_model)  # Same shape as query
mask = torch.ones(batch_size, seq_len).bool()      # Example mask (optional)

# Save the query tensor to a text file
np.savetxt("./data/query.txt", query.detach().numpy().astype(np.float32).reshape(-1, d_model), fmt="%.4f")
np.savetxt("./data/key.txt", key.detach().numpy().astype(np.float32).reshape(-1, d_model), fmt="%.4f")
np.savetxt("./data/value.txt", value.detach().numpy().astype(np.float32).reshape(-1, d_model), fmt="%.4f")
#multi_head_attn = MultiHeadedAttention(h=h, d_model=d_model, dropout=0.1)
output = attention(query, key, value)
# Multi-head attention computation
#output = multi_head_attn(query, key, value)
#print(output)



#output = multi_head_attn(query, key, value, mask=mask)