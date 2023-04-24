# Args should be imported before everything to cover https://discuss.pytorch.org/t/cuda-visible-device-is-of-no-use/10018
from utils.args import args
import numpy as np
import sys
import torch
import torchvision

from progress.bar import Bar
from torch import nn, optim
from torch.utils.data import DataLoader

from datasets.DatasetMaze import balance_dataset
from Models.IDM import IDM
from Models.Policy import Policy
from datasets.DatasetMaze import Policy_Dataset
from datasets.DatasetMaze import split_dataset
from utils.board import Board
from utils.utils import save_policy_model
from utils.utils import policy_infer

from PIL import Image

np.set_printoptions(suppress=True)

# ARGS: GPU and Pretrained
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = './dataset/maze/POLICY/'
path = './dataset/maze/POLICY/maze10/'
train_dataset = Policy_Dataset(config, path, random=False, type='same')
validation_dataset = Policy_Dataset(config, path, train=False, labyrinths_valid=train_dataset.labyrinths_valid)

# Dataloader
print('Creating PyTorch DataLoaders')
batch_size = args.policy_batch_size
dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dataloader_validation = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

# Model and action size
print('Creating Policy Model')
action_dimension = args.actions
policy_model = Policy(action_dimension, pretrained=args.pretrained)
policy_model.load_state_dict(torch.load('./checkpoint/policy/model_443.ckpt'))
policy_model.to(device)
policy_model.eval()

infer = policy_infer(policy_model, transforms=dataloader_train.dataset.transforms, device=device, random=False, size=(10, 10), episodes=100, verbose=True)
