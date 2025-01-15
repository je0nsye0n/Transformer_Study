import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from keras.datasets import imdb
import numpy as np

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.embedding_dim = embedding_dim  # d_model
        self.num_heads = num_heads
        assert embedding_dim % self.num_heads == 0

        self.projection_dim = embedding_dim // num_heads

        self.query_dense = nn.Linear(embedding_dim, embedding_dim)
        self.key_dense = nn.Linear(embedding_dim, embedding_dim)
        self.value_dense = nn.Linear(embedding_dim, embedding_dim)
        self.dense = nn.Linear(embedding_dim, embedding_dim)

    def scaled_dot_product_attention(self, query, key, value):
        matmul_qk = torch.matmul(query, key.transpose(-2, -1))
        depth = key.size(-1)
        logits = matmul_qk / torch.sqrt(torch.tensor(depth, dtype=torch.float32))
        attention_weights = F.softmax(logits, dim=-1)
        output = torch.matmul(attention_weights, value)
        return output, attention_weights

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.projection_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self, inputs):
        batch_size = inputs.size(0)

        query = self.split_heads(self.query_dense(inputs), batch_size)
        key = self.split_heads(self.key_dense(inputs), batch_size)
        value = self.split_heads(self.value_dense(inputs), batch_size)

        scaled_attention, _ = self.scaled_dot_product_attention(query, key, value)
        scaled_attention = scaled_attention.permute(0, 2, 1, 3).contiguous()
        concat_attention = scaled_attention.view(batch_size, -1, self.embedding_dim)

        outputs = self.dense(concat_attention)
        return outputs

class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, dff, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(embedding_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, dff),
            nn.ReLU(),
            nn.Linear(dff, embedding_dim)
        )
        self.layernorm1 = nn.LayerNorm(embedding_dim)
        self.layernorm2 = nn.LayerNorm(embedding_dim)
        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, inputs):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(nn.Module):
    def __init__(self, max_len, vocab_size, embedding_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = nn.Embedding(vocab_size, embedding_dim)
        self.pos_emb = nn.Embedding(max_len, embedding_dim)

    def forward(self, x):
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

class TransformerClassifier(nn.Module):
    def __init__(self, max_len, vocab_size, embedding_dim, num_heads, dff, num_classes, rate=0.1):
        super(TransformerClassifier, self).__init__()
        self.embedding = TokenAndPositionEmbedding(max_len, vocab_size, embedding_dim)
        self.transformer1 = TransformerBlock(embedding_dim, num_heads, dff, rate)
        self.transformer2 = TransformerBlock(embedding_dim, num_heads, dff, rate)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(embedding_dim, 50)
        self.fc2 = nn.Linear(50, num_classes)
        self.dropout = nn.Dropout(rate)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer1(x)
        x = self.transformer2(x)
        x = x.permute(0, 2, 1)  # For pooling
        x = self.pooling(x).squeeze(-1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

# Parameters
vocab_size = 20000
max_len = 200
embedding_dim = 64
num_heads = 4
dff = 128
num_classes = 2

# Load IMDB dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)
X_train = torch.tensor([x[:max_len] + [0] * (max_len - len(x)) for x in X_train])
X_test = torch.tensor([x[:max_len] + [0] * (max_len - len(x)) for x in X_test])
y_train = torch.tensor(y_train)
y_test = torch.tensor(y_test)

# DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Model instance
model = TransformerClassifier(max_len, vocab_size, embedding_dim, num_heads, dff, num_classes)

# Training
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 10
model.train()
for epoch in range(epochs):
    for batch in train_loader:
        x_batch, y_batch = batch
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    test_predictions = torch.argmax(test_outputs, dim=1)
    accuracy = (test_predictions == y_test).float().mean().item()

print(f"Test Accuracy: {accuracy:.4f}")

# Example sentences
word_index = imdb.get_word_index()
index_to_word = {value + 3: key for key, value in word_index.items()}
index_to_word[0] = "<PAD>"
index_to_word[1] = "<START>"
index_to_word[2] = "<UNK>"

sentences = [
    "this movie is great",
    "this movie is terrible",
    "a very bad experience"
]

def encode_sentence(sentence, word_index, max_len):
    tokens = [word_index.get(word, 2) for word in sentence.split()]  # 2 is <UNK>
    return tokens[:max_len] + [0] * (max_len - len(tokens))

encoded_inputs = torch.tensor([encode_sentence(sentence, word_index, max_len) for sentence in sentences])

# Make predictions
with torch.no_grad():
    predictions = model(encoded_inputs)
    predicted_classes = torch.argmax(predictions, dim=1)
    class_labels = ["negative", "positive"]
    decoded_predictions = [class_labels[p.item()] for p in predicted_classes]

for sentence, prediction in zip(sentences, decoded_predictions):
    print(f"Sentence: '{sentence}' => Prediction: {prediction}")
