import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, output_dim=41, hidden_layers=1, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=39,
            hidden_size=hidden_dim,
            num_layers=hidden_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.linear = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1, 39)
        x, h_n = self.gru(x)
        forward_state = h_n[-2, :, :]
        backward_state = h_n[-1, :, :]
        out = self.linear(torch.cat((forward_state, backward_state), dim=-1))
        return out


class BasicBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.block = nn.Sequential(nn.Linear(input_dim, output_dim), nn.ReLU())

    def forward(self, x):
        return self.block(x)
