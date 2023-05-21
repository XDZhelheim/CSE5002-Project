import torch.nn as nn
import torch
import numpy as np
from torchinfo import summary


class MLP(nn.Module):
    def __init__(
        self,
        input_classes_list=[6, 3, 43, 44, 64, 2506],
        embedding_dim=16,
        output_dim=11,
        hidden_dim=32,
    ):
        super().__init__()

        self.input_classes_list = input_classes_list
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.num_features = len(input_classes_list)

        self.embedding_layers = nn.ModuleList(
            nn.Embedding(num_classes, embedding_dim)
            for num_classes in input_classes_list
        )

        self.mlp = nn.Sequential(
            nn.Linear(self.num_features * embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        x = x.long()

        embeddings = []
        for i in range(self.num_features):
            embedding = self.embedding_layers[i](x[:, i])  # (N, embedding_dim)
            embeddings.append(embedding)
        x = torch.concat(embeddings, dim=-1)  # (N, embedding_dim*num_features)

        return self.mlp(x)


if __name__ == "__main__":
    model = MLP()
    summary(model, [5298, 6], device="cpu")
