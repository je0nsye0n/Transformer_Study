import torch
import math
from torch import nn


class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        k_t = k.transpose(2, 3)  # Transpose
        score = (q @ k_t) / math.sqrt(k.size(-1))  # Scaled dot product

        if mask is not None:
            mask = mask.unsqueeze(1)  # 크기 맞춤 (batch_size, 1, seq_len, seq_len)
            score = score.masked_fill(mask == 0, -10000)

        score = self.softmax(score)
        v = score @ v
        return v, score


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

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
        return tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)

    def concat(self, tensor):
        batch_size, head, length, d_tensor = tensor.size()
        return tensor.transpose(1, 2).contiguous().view(batch_size, length, head * d_tensor)


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.enc_dec_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm3 = LayerNorm(d_model=d_model)
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, dec, enc, trg_mask, src_mask):
        _x = dec
        x = self.self_attention(q=dec, k=dec, v=dec, mask=trg_mask)
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        if enc is not None:
            _x = x
            x = self.enc_dec_attention(q=x, k=enc, v=enc, mask=src_mask)
            x = self.dropout2(x)
            x = self.norm2(x + _x)

        _x = x
        x = self.ffn(x)
        x = self.dropout3(x)
        x = self.norm3(x + _x)
        return x


class TransformerEmbedding(nn.Module):
    def __init__(self, d_model, drop_prob, max_len, vocab_size, device):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = self.get_positional_encoding(max_len, d_model, device)
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        return self.dropout(self.token_emb(x) + self.positional_encoding[: x.size(1), :])

    @staticmethod
    def get_positional_encoding(max_len, d_model, device):
        pe = torch.zeros(max_len, d_model, device=device)
        position = torch.arange(0, max_len, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)


class Decoder(nn.Module):
    def __init__(self, dec_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        self.emb = TransformerEmbedding(d_model=d_model, drop_prob=drop_prob, max_len=max_len, vocab_size=dec_voc_size, device=device)
        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model, ffn_hidden=ffn_hidden, n_head=n_head, drop_prob=drop_prob) for _ in range(n_layers)])
        self.linear = nn.Linear(d_model, dec_voc_size)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        trg = self.emb(trg)
        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask, src_mask)
        return self.linear(trg)

# 하이퍼파라미터 설정
batch_size = 2
seq_len = 10
d_model = 512
ffn_hidden = 2048
n_head = 8
n_layers = 6
drop_prob = 0.1
vocab_size = 100  # 임의의 단어 집합 크기
device = "cpu"

# 임의의 입력 데이터 생성
trg = torch.randint(0, vocab_size, (batch_size, seq_len))  # (배치, 길이)
enc_src = torch.randn(batch_size, seq_len, d_model)  # 인코더 출력 (랜덤 벡터)
trg_mask = torch.ones(batch_size, seq_len, seq_len)  # 모든 토큰 사용 가능
src_mask = torch.ones(batch_size, seq_len, seq_len)  # 모든 토큰 사용 가능

# 모델 초기화
decoder = Decoder(
    dec_voc_size=vocab_size,
    max_len=seq_len,
    d_model=d_model,
    ffn_hidden=ffn_hidden,
    n_head=n_head,
    n_layers=n_layers,
    drop_prob=drop_prob,
    device=device
)

#input 출력
print("Decdoer Input:",trg)

# 디코더 실행
output = decoder(trg, enc_src, trg_mask, src_mask)

# 출력 확인
print("Decoder Output Shape:", output.shape)
print("Decoder Output:", output)
