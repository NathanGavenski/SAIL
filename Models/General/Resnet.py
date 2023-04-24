from copy import deepcopy

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .Attention import Self_Attn2D


class Resnet(nn.Module):
    r''' ResNet encoder network for image input.
    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
    '''

    def __init__(self, normalize=False):
        super().__init__()
        self.normalize = normalize
        self.features = models.resnet18(pretrained=True)

        self.att = Self_Attn2D(64)
        self.att2 = Self_Attn2D(128)

    def forward(self, x):
        img = deepcopy(x)
        if self.normalize:
            x = normalize_imagenet(x)

        x = self.features.conv1(x)
        x = self.features.bn1(x)
        x = self.features.relu(x)
        x = self.features.maxpool(x)

        x = self.features.layer1(x)
        x = self.features.layer2(x)
        x = self.features.layer3(x)
        x = self.features.layer4(x)

        x = self.features.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class ResnetFirst(nn.Module):
    r''' ResNet encoder network for image input.
    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
    '''

    def __init__(self, normalize=False):
        super().__init__()
        self.normalize = normalize
        self.features = models.resnet18(pretrained=True)

        name = str(len(self.features.layer1))
        self.features.layer1.add_module(name, Self_Attn2D(64))

        name = str(len(self.features.layer2))
        self.features.layer2.add_module(name, Self_Attn2D(128))

    def forward(self, x):
        img = deepcopy(x)
        if self.normalize:
            x = normalize_imagenet(x)
        return self.features(x)


class ResnetLast(nn.Module):
    r''' ResNet encoder network for image input.
    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
    '''

    def __init__(self, normalize=False):
        super().__init__()
        self.normalize = normalize
        self.features = models.resnet18(pretrained=True)

        self.att3 = Self_Attn2D(256)
        self.att4 = Self_Attn2D(512)

        name = str(len(self.features.layer3))
        self.features.layer3.add_module(name, Self_Attn2D(256))

        name = str(len(self.features.layer4))
        self.features.layer4.add_module(name, Self_Attn2D(512))

    def forward(self, x):
        img = deepcopy(x)
        if self.normalize:
            x = normalize_imagenet(x)
        return self.features(x)