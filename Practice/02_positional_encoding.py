import torch
import numpy as np
from torch import nn

# Define the PositionalEncoding module
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, device):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False

        pos = torch.arange(0, max_len, device=device).float().unsqueeze(dim=1)
        _2i = torch.arange(0, d_model, step=2, device=device).float()

        self.encoding[:, ::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        _, seq_len = x.size()
        return self.encoding[:seq_len, :]

# Example usage
def main():
    # Parameters
    d_model = 8  # Model dimension for easy viewing
    max_len = 20  # Maximum sequence length
    device = "cpu"

    # Create an instance of PositionalEncoding
    pos_enc = PositionalEncoding(d_model=d_model, max_len=max_len, device=device)

    # Input tensor: predefined values
    input_values = torch.tensor([
        [-0.5122, -1.7005, -1.4030, -0.3759, 0.8815, 0.6126, 0.4119, 1.4686, 0.3029, 0.6220],
        [1.3683, -0.3435, -0.1593, -0.2343, -0.1361, -0.3176, -0.3475, 0.0671, 1.4763, 0.4753]
    ], device=device)

    # Generate positional encodings
    encoding = pos_enc(input_values)

    # Print the input and corresponding positional encoding
    print("Input Tensor:")
    print(input_values)
    print("\nPositional Encoding:")
    print(encoding)

    # Combine input with positional encoding (element-wise addition)
    combined = input_values.unsqueeze(2) + encoding.unsqueeze(0)
    print("\nCombined Input with Positional Encoding:")
    print(combined)

if __name__ == "__main__":
    main()
