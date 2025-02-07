import torch
import torch.nn as nn
import math
import numpy as np
import os

torch.set_printoptions(precision=4, sci_mode=False)
torch.set_printoptions(linewidth=200)

# Save weights and biases file
def save_linear_weights_biases(linears, name, folder="Data"):
    os.makedirs(folder, exist_ok=True)
   
    weight_file = os.path.join(folder, f"{name}_linear_weights.txt")
    bias_file = os.path.join(folder, f"{name}_linear_biases.txt")
    np.savetxt(weight_file, linears.weight.detach().numpy().astype(np.float32), fmt="%.4f")
    np.savetxt(bias_file, linears.bias.detach().numpy().astype(np.float32), fmt="%.4f")

def save_layer_norm_params(layer_norm, name, folder="./data"):
    os.makedirs(folder, exist_ok=True)
    
    gamma_file = os.path.join(folder, f"{name}_gamma.txt")
    beta_file = os.path.join(folder, f"{name}_beta.txt")
    
    np.savetxt(gamma_file, layer_norm.gamma.detach().cpu().numpy().astype(np.float32), fmt="%.4f")
    np.savetxt(beta_file, layer_norm.beta.detach().cpu().numpy().astype(np.float32), fmt="%.4f")

class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()

    def forward(self, q, k, v, mask=None, e=1e-12):
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_value = scores.softmax(dim=-1)
        return torch.matmul(attn_value, v)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)
        self.attention = ScaleDotProductAttention()

    def forward(self, q, k, v, mask=None):
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        q, k, v = self.split(q), self.split(k), self.split(v)
        out = self.attention(q, k, v, mask)
        out = self.concat(out)
        out = self.w_concat(out)
        return out

    def split(self, tensor):
        batch_size, length, d_model = tensor.size()
        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        return tensor

    def concat(self, tensor):
        batch_size, head, length, d_tensor = tensor.size()
        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, -1)
        return tensor

class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, n_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.ReLU(),
            nn.Linear(ffn_hidden, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        _x = x
        x = self.attention(x, x, x, mask)
        x = self.norm1(x + _x)

        _x = x
        x = self.ffn(x)
        x = self.norm2(x + _x)
        return x

class Encoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, n_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, ffn_hidden, n_head)
            for _ in range(n_layers)
        ])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x

if __name__ == "__main__":
    torch.set_printoptions(sci_mode=False)
    
    enc_voc_size = 100
    max_len = 50
    d_model = 16
    ffn_hidden = 128
    n_head = 4
    n_layers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    batch_size = 2
    seq_len = 5
    src = torch.randint(0, enc_voc_size, (batch_size, seq_len, d_model)).float().to(device)
    src_mask = torch.ones((batch_size, 1, seq_len, seq_len)).to(device)  # Masking for attention
    
    encoder = Encoder(d_model, ffn_hidden, n_head, n_layers).to(device)
    output = encoder(src, src_mask)
    print(output)