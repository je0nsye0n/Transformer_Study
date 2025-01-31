import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from copy import deepcopy
import os

torch.set_printoptions(precision=4, sci_mode=False)
torch.set_printoptions(linewidth=200) 

# Save weights and biases file
def save_linear_weights_biases(linears, folder="data"):
    os.makedirs(folder, exist_ok=True)
    for i, linear in enumerate(linears):
        weight_file = os.path.join(folder, f"linear{i+1}_weights.txt")
        bias_file = os.path.join(folder, f"linear{i+1}_biases.txt")
        np.savetxt(weight_file, linear.weight.detach().numpy().astype(np.float32), fmt="%.4f")
        np.savetxt(bias_file, linear.bias.detach().numpy().astype(np.float32), fmt="%.4f")

# Load weights and biases
def load_linear_weights_biases(linears, folder="data"):
    for i, linear in enumerate(linears):
        weight_file = os.path.join(folder, f"linear{i+1}_weights.txt")
        bias_file = os.path.join(folder, f"linear{i+1}_biases.txt")
        if os.path.exists(weight_file) and os.path.exists(bias_file):
            weights = np.loadtxt(weight_file, dtype=np.float32)
            biases = np.loadtxt(bias_file, dtype=np.float32)
            linear.weight.data = torch.tensor(weights)
            linear.bias.data = torch.tensor(biases)
        else:
            print(f"Warning: {weight_file} or {bias_file} not found. Skipping load.")


# module clone
def clones(module, N):
    return nn.ModuleList([deepcopy(module) for _ in range(N)])

# attention function
def attention(query, key, value):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # Scaled dot-product
    attn_value = scores.softmax(dim=-1)  # Apply softmax
    return torch.matmul(attn_value, value)  # Weighted sum of values

# MHA module
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        save_linear_weights_biases(self.linears)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value):
        nbatches = query.size(0)
        # Linear projections
        load_linear_weights_biases(self.linears) 
        results = []
        for l, x in zip(self.linears, (query, key, value)):
            x = l(x)  # 선형 변환
            x = x.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)  # view와 transpose
            results.append(x)

        query, key, value = results
        
        # Apply attention
        x = attention(query, key, value)

        # Concatenate heads and apply final linear layer
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


batch_size = 2
seq_len = 10
d_model = 6
h = 2

# Input initialization
query = torch.randn(batch_size, seq_len, d_model)  # (batch_size, seq_len, d_model)
key = torch.randn(batch_size, seq_len, d_model)    # Same shape as query
value = torch.randn(batch_size, seq_len, d_model)  # Same shape as query
mask = torch.ones(batch_size, seq_len).bool()      # Example mask (optional)

# Save the query tensor to a text file
np.savetxt("./data/query.txt", query.detach().numpy().astype(np.float32).reshape(-1, d_model), fmt="%.4f")
np.savetxt("./data/key.txt", key.detach().numpy().astype(np.float32).reshape(-1, d_model), fmt="%.4f")
np.savetxt("./data/value.txt", value.detach().numpy().astype(np.float32).reshape(-1, d_model), fmt="%.4f")

multi_head_attn = MultiHeadedAttention(h=h, d_model=d_model, dropout=0.1)

# Multi-head attention computation
output = multi_head_attn(query, key, value)

print("<output>")
print(output)