"""Node-wise shared predictor with city-local identity embeddings."""
import torch
from torch import nn

class MultiLayerPerceptron(nn.Module):
    """Multi-Layer Perceptron with residual links."""

    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(
            in_channels=input_dim,  out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.fc2 = nn.Conv2d(
            in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.15)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Feed forward of MLP.

        Args:
            input_data (torch.Tensor): input data with shape [B, D, N]

        Returns:
            torch.Tensor: latent repr
        """

        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))      # MLP
        hidden = hidden + input_data                           # residual
        return hidden


class SharedBackbone(nn.Module):
    def __init__(self, lag=12, horizon=6, dim=8):
        super().__init__()
        self.time_series_emb_layer = nn.Conv2d(lag, dim, 1)
        self.encoders = nn.ModuleList([MultiLayerPerceptron(3*dim, 3*dim) for _ in range(3)])
        self.regression_layer = nn.Conv2d(3*dim, horizon, 1)

    def forward(self, history, node_embedding, time_embedding, last):
        batch, length, nodes = history.shape
        features = history.transpose(1, 2).reshape(batch, nodes, length)
        features = self.time_series_emb_layer(features.transpose(1, 2).unsqueeze(-1))
        node = node_embedding.T[None, :, :, None].expand(batch, -1, -1, -1)
        temporal = time_embedding[:, :, None, None].expand(-1, -1, nodes, -1)
        hidden = torch.cat((features, node, temporal), dim=1)
        for encoder in self.encoders:
            hidden = encoder(hidden)
        hidden = hidden + last[:, None, :, None]
        return self.regression_layer(hidden).squeeze(-1).transpose(1, 2)


class FedTriScope(nn.Module):
    """Input [batch, history, nodes], output [batch, nodes, horizon].

    Inputs/outputs are standardized. ``start`` contains calendar ticks for the
    first historical observation; only its last-history daily slot is embedded.
    """
    def __init__(self, num_nodes, steps_per_day, lag=12, horizon=6, dim=8):
        super().__init__()
        self.lag, self.horizon = lag, horizon
        self.steps_per_day = steps_per_day
        self.node_emb = nn.Parameter(torch.empty(num_nodes, dim))
        self.time_emb = nn.Parameter(torch.empty(steps_per_day, dim))
        nn.init.xavier_uniform_(self.node_emb)
        nn.init.xavier_uniform_(self.time_emb)
        self.shared_model = SharedBackbone(lag, horizon, dim)

    def forward(self, speed, start):
        last = speed[:, -1, :]
        temporal = self.time_emb[(start + self.lag - 1) % self.steps_per_day]
        return self.shared_model(speed-last[:, None, :], self.node_emb, temporal, last)
