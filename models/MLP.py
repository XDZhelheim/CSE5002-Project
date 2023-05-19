import torch.nn as nn
import torch
import numpy as np
from torchinfo import summary


class MLP(nn.Module):
    def __init__(self, num_nodes=5298, input_dim=6, output_dim=11, hidden_dim=32):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.mlp(x)


if __name__ == "__main__":
    model = MLP()
    summary(model, [5298, 6], device="cpu")
