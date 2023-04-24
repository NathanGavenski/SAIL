import random

import torch
import numpy as np


class RASS(object):
    def __init__(self, alpha=(0.6, 0.8), beta=(1.2, 1.4)):
        assert isinstance(alpha, (tuple, list, torch.Tensor))
        assert isinstance(beta, (tuple, list, torch.Tensor))
        self.alpha = alpha
        self.beta = beta

    def __call__(self, sample):
        alpha = random.choice(self.alpha)
        beta = random.choice(self.beta)
        uniform = torch.distributions.uniform.Uniform(alpha, beta)
        z = uniform.sample(sample.shape)
        return sample * z


class RASM(object):
    def __init__(self, alpha=(0.6, 0.8), beta=(1.2, 1.4)):
        assert isinstance(alpha, (tuple, list, torch.Tensor))
        assert isinstance(beta, (tuple, list, torch.Tensor))
        self.alpha = alpha
        self.beta = beta

    def __call__(self, sample):
        alpha = random.choice(self.alpha)
        beta = random.choice(self.beta)
        z = (alpha - beta) * torch.rand(*sample.size()) + beta
        return sample * z


class GN(object):
    def __init__(self, loc=torch.tensor([0.0]), scale=torch.tensor([1.0])):
        assert isinstance(loc, (float, torch.Tensor))
        assert isinstance(scale, (float, torch.Tensor))
        self.dist = torch.distributions.normal.Normal(loc, scale)

    def __call__(self, sample):
        z = self.dist.sample(sample.shape).squeeze(-1)
        return sample + z


def create_dataset(dataset_path: str, size: int = None, mode: str = 'episode'):
    dataset = np.load(dataset_path, allow_pickle=True)
    beggining = np.where(dataset['episode_starts'] == True)[0]
    end = np.array([*np.array(beggining - 1)[1:], dataset['episode_starts'].shape[0] - 1])

    rewards = dataset['episode_returns']


    if size is not None and mode == 'episode':
        beggining = beggining[:size]
        end = end[:size]
        rewards = rewards[:size]

    reward_indexes = []
    controller = False
    trajectories = np.ndarray(shape=(0, 3))
    for idx, (b, e) in enumerate(zip(beggining, end)):
        _b = np.array(list(range(b, e)))
        _e = _b + 1
        actions = dataset['actions'][_b]

        _b = _b[None].swapaxes(0, 1)
        _e = _e[None].swapaxes(0, 1)
        entry = np.hstack((_b, _e, actions.reshape((actions.shape[0],1))))

        if mode == 'pairs':
            reward_indexes.append(idx)
            if entry.shape[0] + trajectories.shape[0] >= size:
                amount = size - (entry.shape[0] + trajectories.shape[0])
                entry = entry[:amount, :]
                controller = True
        trajectories = np.append(trajectories, entry, axis=0)

        if mode == 'pairs' and controller:
            break

    if mode == 'episode':
        return trajectories, rewards.mean(), dataset['obs']
    elif mode == 'pairs':
        return trajectories, rewards[reward_indexes].mean(), dataset['obs']