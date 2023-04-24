import math
import random

import numpy as np
import torch
import torch.autograd as autograd
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .General.Empty import Empty
from .General.Resnet import *
from .General.MLP import *


class Policy(nn.Module):

    def __init__(self, action_size, net='vgg', pretrained=True, input=4):
        super(Policy, self).__init__()

        self.net = net
        self.action_size = action_size

        if net == 'inception':
            self.model = models.inception_v3(pretrained=pretrained)
            self.model.fc = Empty()
            linear = nn.Linear(2048, 4096)
        elif net == 'vgg':
            self.model = models.vgg19_bn(pretrained=pretrained)
            self.model.classifier = Empty()
            self.fc_layers = nn.Sequential(
                nn.Linear((512 * 7 * 7), 4096),
                nn.LeakyReLU(),
                nn.Dropout(0.5),
                nn.Linear(4096, 4096),
                nn.LeakyReLU(),
                nn.Dropout(0.5),
                nn.Linear(4096, action_size)
            )

            if pretrained:
                print('Freezing weights')
                for params in self.model.parameters():
                    params.requires_grad = False

        elif net in ['attention', 'resnet']:
            if net == 'attention':
                self.model = ResnetFirst(normalize=False)
            else:
                self.model = Resnet(normalize=False)

            self.model.features.fc = nn.Sequential(
                nn.Linear(512, 512),
                nn.LeakyReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 512),
                nn.LeakyReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, action_size)
            )

        elif net == 'vector':
            self.model = MlpWithAttention(input, action_size)
            self.generator = MLP(input + action_size, input)

    def forward(self, state, generator=True, detach=True):
        # output: [b, action_size] - Cartpole: [b, 2]
        action = self.model(state)

        if generator:
            # Output: [b, state_size + action_size] - Cartpole: [b, 6]
            feature_map = torch.hstack((state, action))
            if detach:
                feature_map = feature_map.detach()

            # Output: [b, state_size] - Cartpole: [b, 4]
            new_state = self.generator(feature_map)
            return action, new_state
        else:
            return action, None

    def act(self, state, epsilon):
        if not isinstance(state, torch.Tensor):
            state = torch.Tensor(state)[None]
            state = state.cuda() if torch.cuda.is_available() else state
        q_value = self.forward(state)

        if random.random() > epsilon:
            action = torch.argmax(q_value, 1)
            return action.item(), 0
        else:
            classes = np.arange(self.action_size)
            prob = torch.nn.functional.softmax(
                q_value, dim=1).cpu().detach().numpy()
            action = np.random.choice(classes, p=prob[0])
            return action, 0 if action == torch.argmax(q_value, 1) else 1


def train(
    model,
    idm_model,
    data,
    criterion,
    g_criterion,
    optimizer,
    g_optimizer,
    device,
    args,
    actions=None,
    tensorboard=None
):
    if not model.training:
        model.train()

    if idm_model.training:
        idm_model.eval()

    s, nS, a_gt = data

    #print(device)
    s = s.to(device)
    nS = nS.to(device)

    prediction = idm_model(s, nS)

    if args.choice == 'max':
        action = torch.argmax(prediction, 1)
    elif args.choice == 'weighted':
        classes = np.arange(actions)
        prob = torch.nn.functional.softmax(
            prediction, dim=1).cpu().detach().numpy()

        action = []
        for i in range(s.shape[0]):
            a = np.random.choice(classes, p=prob[i])
            action.append(a)

        action = torch.tensor(action)
        action = action.to(device)

    if tensorboard is not None:
        tensorboard.add_histogram('Train/action distribution', action)
        tensorboard.add_histogram('Train/GT action distribution', a_gt)

    optimizer.zero_grad()
    g_optimizer.zero_grad()
    action_pred, new_state = model(s, generator=True)

    # Action
    action_pred_argmax = torch.argmax(action_pred, 1)
    loss = criterion(action_pred, action.long())
    acc = ((action_pred_argmax == action).sum().item() /
           a_gt.shape[0]) * 100  # \hat{\hat{a}} == \hat{a}
    loss.backward()
    optimizer.step()

    # New State (s_{t+1})
    loss_g = g_criterion(new_state, nS.float())
    loss_g.backward()
    g_optimizer.step()

    return (loss, acc), loss_g


def validation(model, idm_model, data, device, args, actions=None, tensorboard=None):
    if model.training:
        model.eval()

    if idm_model.training:
        idm_model.eval()

    s, nS, a_gt = data

    s = s.to(device)
    nS = nS.to(device)

    prediction = idm_model(s, nS)

    if args.choice == 'max':
        action = torch.argmax(prediction, 1)
    elif args.choice == 'weighted':
        classes = np.arange(actions)
        prob = torch.nn.functional.softmax(
            prediction, dim=1).cpu().detach().numpy()

        action = []
        for i in range(s.shape[0]):
            action.append(np.random.choice(classes, p=prob[i]))

        action = torch.tensor(action)
        action = action.to(device)

    if tensorboard is not None:
        tensorboard.add_histogram('Valid/action distribution', action)
        tensorboard.add_histogram('Valid/GT action distribution', a_gt)

    action_pred, new_state = model(s)
    pred_argmax = torch.argmax(action_pred, 1)

    acc = ((pred_argmax == action).sum().item() / a_gt.shape[0]) * 100
    g_dist = torch.pow((new_state - nS.long()), 2).mean()

    return acc, g_dist
