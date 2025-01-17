import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from copy import deepcopy

# Helper function to create clones of a module
def clones(module, N):
    return nn.ModuleList([deepcopy(module) for _ in range(N)])

# Attention function
def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # Scaled dot-product
    if mask is not None:
        mask = mask.unsqueeze(1).unsqueeze(2)  # Expand dimensions for compatibility
        scores = scores.masked_fill(mask == 0, float('-inf'))
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

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

# Example usage
batch_size = 1
seq_len = 10
d_model = 6
h = 2

multi_head_attn = MultiHeadedAttention(h=h, d_model=d_model, dropout=0.1)

query = torch.randn(batch_size, seq_len, d_model)
key = torch.randn(batch_size, seq_len, d_model)
value = torch.randn(batch_size, seq_len, d_model)
mask = torch.ones(batch_size, seq_len).bool()

print("Q")
print(query)
print("mask",mask)

output = multi_head_attn(query, key, value, mask=mask)

print("Output shape:", output.shape)
print("Output:", output)
