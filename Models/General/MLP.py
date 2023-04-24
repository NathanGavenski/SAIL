import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .Attention import Self_Attn1D


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLP, self).__init__()

        out = max(8, in_dim * 2)
        self.input = nn.Linear(in_dim, out)
        self.fc = nn.Linear(out, out)
        self.fc2 = nn.Linear(out, out)
        self.output = nn.Linear(out, out_dim)

        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = x.float()
        x = self.relu(self.input(x))
        x = self.relu(self.fc(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.output(x))
        return x


class MlpWithAttention(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        # IF we will use batch norm split fake and true data

        out = max(8, in_dim)
        self.layer1 = nn.Linear(in_dim, 32)
        self.layer2 = nn.Linear(32, 32)
        self.output = nn.Linear(32, out_dim)

        self.attention1 = Self_Attn1D(32, nn.LeakyReLU)
        self.attention2 = Self_Attn1D(32, nn.LeakyReLU)

        self.norm1 = nn.LayerNorm(32, elementwise_affine=True)
        self.norm2 = nn.LayerNorm(32, elementwise_affine=True)

        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = x.float()

        x = self.layer1(x)
        x = self.attention1(x)
        x = self.norm1(x)
        x = self.relu(x)

        x = self.layer2(x)
        x = self.attention2(x)
        x = self.norm2(x)
        x = self.relu(x)

        x = self.output(x)

        return x


class MlpAttention(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MlpAttention, self).__init__()

        out = max(8, in_dim * 2)
        self.input = nn.Linear(in_dim, out)
        self.output = nn.Linear(out, out_dim)
        self.attention = Self_Attn1D(out, nn.LeakyReLU)

        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = x.float()
        x = self.input(x)
        x = self.attention(x)
        x = self.relu(self.output(x))
        return x
