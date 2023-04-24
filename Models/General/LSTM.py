import torch
import torch.nn as nn


class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layer, output_dim, device):
        super().__init__()

        self.device = device

        self.hidden_dim = hidden_dim  # Hidden dimensions
        self.layer_dim = num_layer  # Number of hidden layers
        self.output_dim = output_dim
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layer,
            batch_first=True
        )  # (batch_dim, seq_dim, feature_dim)

        # self.fc = nn.Linear(hidden_dim, output_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 4096),
            nn.LeakyReLU(),
            nn.Dropout(0.5),

            nn.Linear(4096, 4096),
            nn.LeakyReLU(),
            nn.Dropout(0.5),

            nn.Linear(4096, output_dim)
        )

    def forward(self, x, seq_size):  # (batch, seq, features)
        x = x.float()

        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(
            0), self.hidden_dim).requires_grad_()
        h0 = h0.to(self.device)

        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(
            0), self.hidden_dim).requires_grad_()
        c0 = c0.to(self.device)

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        b, s, f = out.shape
        out = out.reshape(-1, self.hidden_dim)
        classes = self.fc(out)
        classes = classes.reshape(b, s, self.output_dim)

        return classes
