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

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, device):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False

        pos = torch.arange(0, max_len, device=device).float().unsqueeze(1)
        _2i = torch.arange(0, d_model, step=2, device=device).float()

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        seq_len = x.size(1)
        return self.encoding[:seq_len, :]

class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        k_t = k.transpose(-2, -1)
        score = (q @ k_t) / math.sqrt(q.size(-1))

        if mask is not None:
            score = score.masked_fill(mask == 0, -10000.0)

        score = self.softmax(score)
        v = score @ v
        return v, score

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)
        save_linear_weights_biases(self.w_q,"query")
        save_linear_weights_biases(self.w_k,"key")
        save_linear_weights_biases(self.w_v,"value")
        save_linear_weights_biases(self.w_concat,"concat")
        self.attention = ScaleDotProductAttention()

    def forward(self, q, k, v, mask=None):
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        q, k, v = self.split(q), self.split(k), self.split(v)
        out, attention = self.attention(q, k, v, mask=mask)
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

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        save_layer_norm_params(self, self.__class__.__name__)
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)
        self.linear2 = nn.Linear(hidden, d_model)
        save_linear_weights_biases(self.linear1,"linear1")
        save_linear_weights_biases(self.linear2,"linear2")


    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        #x = self.dropout(x)
        x = self.linear2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, n_head)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, drop_prob)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, src_mask):
        _x = x
        x = self.attention(x, x, x, mask=src_mask)
        #x = self.dropout1(x)
        x = self.norm1(x + _x)

        _x = x
        x = self.ffn(x)
        #x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x

class TransformerEmbedding(nn.Module):
    def __init__(self, d_model, max_len, vocab_size, drop_prob, device):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_len, device)
        self.dropout = nn.Dropout(p=drop_prob)
        self.save_embedding_weights("./Data/embedding_weights.txt")
        
    def save_embedding_weights(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.savetxt(filepath, self.tok_emb.weight.detach().cpu().numpy(), fmt="%.6f")

    def forward(self, x):
        x = self.tok_emb(x) + self.pos_emb(x)
        return x

class Encoder(nn.Module):
    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        self.emb = TransformerEmbedding(d_model, max_len, enc_voc_size, drop_prob, device)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, ffn_hidden, n_head, drop_prob)
            for _ in range(n_layers)
        ])

    def forward(self, x, src_mask):
        x = self.emb(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x

if __name__ == "__main__":
    torch.set_printoptions(sci_mode=False)
    # Hyperparameters
    enc_voc_size = 100
    max_len = 50
    d_model = 16
    ffn_hidden = 128
    n_head = 4
    n_layers = 4
    drop_prob = 0.1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Sample input
    batch_size = 2
    seq_len = 5
    src = torch.randint(0, enc_voc_size, (batch_size, seq_len)).to(device)
    src_mask = torch.ones((batch_size, 1, 1, seq_len)).to(device)  # No masking

    # Initialize encoder
    encoder = Encoder(enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device).to(device)

    # Forward pass
    output = encoder(src, src_mask)
    
    np.savetxt("./Data/input.txt",src.detach().numpy().astype(np.float32),fmt="%.4f")
    np.savetxt("./Data/output.txt",output.detach().numpy().astype(np.float32),fmt="%.4f")