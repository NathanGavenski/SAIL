from collections import defaultdict
import torch
import numpy as np
import pandas as pd

from copy import copy
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from .utils import RASS, RASM, GN, create_dataset
from utils.enjoy import get_environment

np.set_printoptions(suppress=True)


def detect_path(file):
    if '/' in file:
        return True
    else:
        return False


def read_data(dataset_path, env_name, idm=True):
    count = defaultdict(list)
    actions = []
    states = np.ndarray((0, 2), dtype=str)
    with open(f'{dataset_path}{env_name}.txt') as f:
        for idx, line in enumerate(f):
            word = line.replace('\n', '').split(';')
            state = word[0] if detect_path(word[0]) else f'{dataset_path}{word[0]}'
            nState = word[1] if detect_path(word[1]) else f'{dataset_path}{word[1]}'
            action = int(word[-1])
            actions.append(action)
            states = np.append(states, [[state, nState]], axis=0)
            count[action].append(idx)

    return count, states, np.array(actions)


def read_vector(dataset_path, env_name, idm=True):
    state_size = get_environment({'name': env_name}).reset().shape[0]
    count = defaultdict(list)
    actions = []
    states = np.ndarray((0, 2, state_size), dtype=str)
    with open(f'{dataset_path}{env_name}.txt') as f:
        for idx, line in enumerate(f):
            word = line.replace('\n', '').split(';')
            state = np.fromstring(word[0].replace('[', '').replace(']', '').replace(',', ' '), sep=' ', dtype=float)
            nState = np.fromstring(word[1].replace('[', '').replace(']', '').replace(',', ' '), sep=' ', dtype=float)

            s = np.append(state[None], nState[None], axis=0)

            action = int(word[-1])
            actions.append(action)
            states = np.append(states, s[None], axis=0)
            count[action].append(idx)

    return count, states, np.array(actions)


def balance_dataset(dataset_path, env_name, downsample_size=5000, replace=True, sampling=True, vector=False):
    if vector is False:
        data = read_data(dataset_path)
    else:
        data = read_vector(dataset_path, env_name)
    count, states, actions = data

    sizes = []
    dict_sizes = {}
    for key in count:
        sizes.append(len(count[key]))
        dict_sizes[key] = len(count[key])
    print('Size each action:', dict_sizes)

    if sampling is True:
        max_size = np.min(sizes) if downsample_size is not None else None
        downsample_size = downsample_size if downsample_size is not None else np.inf
        downsample_size = min(downsample_size, max_size)

    classes = list(range(3))
    all_idxs = np.ndarray((0), dtype=np.int32)
    if downsample_size is not None:
        for i in classes:
            size = len(count[i])

            try:
                random_idxs = np.random.choice(size, downsample_size, replace=replace)
            except ValueError:
                random_idxs = np.random.choice(size, size, replace=replace)

            idxs = np.array(count[i])[random_idxs]
            all_idxs = np.append(all_idxs, idxs, axis=0).astype(int)

        states = states[all_idxs]
        a = actions[all_idxs]
    else:
        a = actions

    print('Final size action:', np.bincount(a))
    return states, a


def split_dataset(states, actions, stratify=True):
    if stratify:
        return train_test_split(states, actions, test_size=0.3, stratify=actions)
    else:
        return train_test_split(states, actions, test_size=0.3)


class IDM_Dataset(Dataset):

    transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    def __init__(self, path, images, actions, mode='RGB'):
        super().__init__()
        self.previous_images = images[:, 0]
        self.next_images = images[:, 1]
        self.labels = actions
        self.mode = mode
        self.path = path

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        s = Image.open(self.previous_images[idx]).convert(self.mode)
        nS = Image.open(self.next_images[idx]).convert(self.mode)

        s = self.transforms(s)
        nS = self.transforms(nS)

        a = torch.tensor(self.labels[idx])

        return (s, nS, a)


class IDM_Vector_Dataset(Dataset):

    transforms = transforms.Compose([
        torch.from_numpy,
    ])

    def __init__(self, path, images, actions, augment):
        super().__init__()
        self.previous_images = images[:, 0]
        self.next_images = images[:, 1]
        self.labels = actions
        self.augment = augment

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        s = self.previous_images[idx].astype(float)
        nS = self.next_images[idx].astype(float)

        if self.augment and torch.rand(1)[0] > 0.5:
            s = self.transforms(s)
            nS = self.transforms(nS)
        else:
            s = torch.from_numpy(s)
            nS = torch.from_numpy(nS)

        a = torch.tensor(self.labels[idx])

        return (s, nS, a)


class Policy_Dataset(Dataset):

    transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    random_policy = -200
    expert = -125

    def __init__(self, path, images, actions, mode='RGB'):
        super().__init__()
        self.previous_images = images[:, 0]
        self.next_images = images[:, 1]
        self.labels = actions
        self.mode = mode
        self.path = path

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        s = Image.open(self.previous_images[idx]).convert(self.mode)
        nS = Image.open(self.next_images[idx]).convert(self.mode)

        s = self.transforms(s)
        nS = self.transforms(nS)

        a = torch.tensor(self.labels[idx])

        return (s, nS, a)

    def get_performance_rewards(self, *args):
        return self.expert, self.random_policy


class Policy_Vector_Dataset(Dataset):

    transforms = transforms.Compose([
        torch.from_numpy
    ])

    random_policy = -200
    expert = -137.43835616438355

    def __init__(self, path, images, actions, ):
        super().__init__()
        self.dataset = path
        self.labels = actions
        self.previous = images[:, 0].astype(float)
        self.next = images[:, 1].astype(float)

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        s = self.dataset[int(self.previous[idx])]
        nS = self.dataset[int(self.next[idx])]
        s = torch.from_numpy(s)
        nS = torch.from_numpy(nS)
        a = torch.tensor(self.labels[idx])
        return s, nS, a

    def get_performance_rewards(self, *args):
        return self.expert, self.random_policy


def get_idm_dataset(path, batch_size, downsample_size=5000, shuffle=True, replace=True, sampling=True, **kwargs):

    states, actions = balance_dataset(
        path,
        downsample_size=downsample_size,
        replace=replace,
        sampling=sampling,
    )
    train, validation, train_y, validation_y = split_dataset(states, actions)

    train_dataset = IDM_Dataset(path, train, train_y)
    validation_dataset = IDM_Dataset(path, validation, validation_y)
    train = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    validation = DataLoader(validation_dataset, batch_size=batch_size, shuffle=shuffle)
    return train, validation


def get_policy_dataset(path, batch_size, shuffle=True, **kwargs):
    states, actions = balance_dataset(
        path,
        downsample_size=None,
        replace=False,
        sampling=False,
    )

    train, validation, train_y, validation_y = split_dataset(states, actions)
    train_dataset = Policy_Dataset(path, train, train_y)
    validation_dataset = Policy_Dataset(path, validation, validation_y)

    train = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    validation = DataLoader(validation_dataset, batch_size=batch_size, shuffle=shuffle)
    return train, validation


def get_idm_vector_dataset(
    path,
    batch_size,
    env_name,
    downsample_size=5000,
    shuffle=True,
    replace=True,
    sampling=True,
    augment=False,
    **kwargs
):

    states, actions = balance_dataset(
        path,
        env_name,
        downsample_size=downsample_size,
        replace=replace,
        sampling=sampling if downsample_size is not None else False,
        vector=True,
    )
    train, validation, train_y, validation_y = split_dataset(states, actions)

    train_dataset = IDM_Vector_Dataset(path, train, train_y, augment)
    validation_dataset = IDM_Vector_Dataset(path, validation, validation_y, augment)
    train = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    validation = DataLoader(validation_dataset, batch_size=batch_size, shuffle=shuffle)
    return train, validation


def get_policy_vector_dataset(
    path,
    batch_size,
    downsample_size=None,
    shuffle=True,
    replace=True,
    sampling=True,
    augment=False,
    validation=True,
    **kwargs
):

    indexes, reward, dataset = create_dataset(path, size=downsample_size, mode='episode')
    states, actions = indexes[:, :-1], indexes[:, -1]
    print(f'Average Expert Reward: {reward} and size: {indexes.shape}')

    if validation:
        train, validation, train_y, validation_y = split_dataset(states, actions)
        train_dataset = Policy_Vector_Dataset(dataset, train, train_y)
        validation_dataset = Policy_Vector_Dataset(dataset, validation, validation_y)
        train_dataset.expert = reward
        validation_dataset.expert = reward
        train = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        validation = DataLoader(validation_dataset, batch_size=batch_size, shuffle=shuffle)
        return train, validation
    else:
        train_dataset = Policy_Vector_Dataset(dataset, states, actions)
        train_dataset.expert = reward
        #print(len(train_dataset))
        #exit()
        train = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        return train, None