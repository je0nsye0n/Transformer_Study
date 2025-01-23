import torch
import torch.nn as nn
import numpy as np 

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
        #x = self.dropout(x)
        x = self.linear2(x)
        return x

# 모델 초기화
torch.manual_seed(0)  # 랜덤 시드 고정
d_model = 8
hidden = 32
drop_prob = 0.0
model = PositionwiseFeedForward(d_model, hidden, drop_prob)

# Save model weights and biases in float32
np.savetxt("./data2/linear1_weights.txt", model.linear1.weight.detach().numpy().astype(np.float32), fmt="%.4f")
np.savetxt("./data2/linear1_biases.txt", model.linear1.bias.detach().numpy().astype(np.float32), fmt="%.4f")
np.savetxt("./data2/linear2_weights.txt", model.linear2.weight.detach().numpy().astype(np.float32), fmt="%.4f")
np.savetxt("./data2/linear2_biases.txt", model.linear2.bias.detach().numpy().astype(np.float32), fmt="%.4f")

# Generate deterministic input
input_tensor = torch.randn(2, 4, d_model)  # (BATCH_SIZE, SEQ_LEN, D_MODEL)
np.savetxt("./data2/input_data.txt", input_tensor.numpy().astype(np.float32).reshape(-1), fmt="%.4f")

output = model(input_tensor)
print("Output Tensor:")
print(output)