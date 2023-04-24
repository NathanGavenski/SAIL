import gym
import numpy as np
import torch
import math

from torch import nn, optim
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.enjoy import get_environment
from Models.Policy import Policy
from Models.General.LSTM import LSTM


class ReplayBuffer:
    def __init__(self, observation_space):
        self.observation_space = observation_space
        self.states = torch.Tensor(size=(0, observation_space))
        self.episode = []
        self.actions = []
        self.count = 1
        self.padding = 0

    def insert(self, state, action, count):
        if count > self.padding:
            self.padding = count

        self.states = torch.cat((self.states, state.detach().cpu()), dim=0)
        self.actions.append(action)
        self.episode.append(count)

    def sample(self):
        begin = np.where(np.array(self.episode) == 1)[0]
        end = np.append(begin[1:] - 1, [len(self.episode)], axis=0)

        sequence_size = []
        episode = torch.Tensor(0, self.padding, self.observation_space)
        for b, e in zip(begin, end):
            idxs = [idx for idx in range(b, e)]
            states = self.states[idxs]
            sequence_size.append(states.shape[0])
            padding_size = self.padding - states.shape[0]
            padding = torch.zeros((padding_size, self.observation_space)).detach().to('cpu')
            states = torch.cat((states, padding), dim=0)
            episode = torch.cat((episode, states[None]), dim=0)

        return episode, sequence_size

    def get_by_index(self, indexes):
        begin = np.where(np.array(self.episode) == 1)[0]
        end = np.append(begin[1:] - 1, [len(self.episode) - 1], axis=0)
        begin = begin[indexes]
        end = end[indexes]

        state, next_state, action = [], [], []
        for b, e in zip(begin, end):
            idxs = [idx for idx in range(b, e)]
            _state = self.states[idxs]
            _next_state = self.states[np.array(idxs) + 1]
            _action = np.array(self.actions)[idxs]

            for s, a, nS in zip(_state, _action, _next_state):
                state.append(s.tolist()) # i am a tensor
                next_state.append(nS.tolist())
                action.append(a.tolist())

        return state, next_state, action


class DiscriminatorDataset(Dataset):

    transforms = transforms.Compose([
        transforms.ToTensor()
    ])

    def __init__(self, data, sequence_size):
        super().__init__()
        self.sequence_size = sequence_size
        self.data = data[:, :, :-1]
        self.label = data[:, :, -1]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        data = self.data[index, :, :]
        label = self.label[index, :]
        sequence_size = self.sequence_size[index]

        if not isinstance(data, torch.Tensor):
            data = self.transforms(data)

        if not isinstance(label, torch.Tensor):
            label = self.transforms(label)

        return data, label, sequence_size, index


def get_expert_data(expert_path, amount):
    expert_data = np.load(expert_path, allow_pickle=True)
    begin = np.where(expert_data['episode_starts'] == True)[0]
    end = np.append(
        begin[1:] - 1,
        [expert_data['episode_starts'].shape[0]],
        axis=0
    )

    begin = begin[:amount]
    end = end[:amount]

    max_size = 0
    for b, e in zip(begin, end):
        if max_size < e - b:
            max_size = e - b

    sequence_size = []
    episode = torch.Tensor(0, max_size, expert_data['obs'][0].shape[0])
    for b, e in zip(begin, end):
        idxs = [idx for idx in range(b, e)]
        states = torch.from_numpy(expert_data['obs'][idxs])
        sequence_size.append(states.shape[0])
        padding_size = max_size - states.shape[0]
        padding = torch.zeros((padding_size, expert_data['obs'][0].shape[0]))
        states = torch.cat((states, padding), dim=0)
        episode = torch.cat((episode, states[None]), dim=0)

    return episode, sequence_size


def adversarial(
    policy,
    discriminator,
    episodes,
    device,
    environment,
    expert,
    d_criterion,
    d_optimizer,
    batch_size
):
    if not policy.training:
        policy.train()

    if not discriminator.training:
        discriminator.train()

    env = get_environment(environment, None, None, False)
    replaybuffer = ReplayBuffer(env.reset().shape[0])

    aer = []
    with torch.no_grad():
        for _ in range(episodes):
            done, state = False, env.reset()

            count, total_reward = 0, 0
            while not done:
                state = torch.from_numpy(state)[None].to(device)

                action, _ = policy(state, generator=False)

                replaybuffer.insert(
                    state,
                    torch.argmax(action, 1).item(),
                    count
                )

                action = torch.argmax(action).item()
                next_state, reward, done, info = env.step(action)
                state = next_state
                total_reward += reward
                count += 1
            aer.append(total_reward)

    fake_data, fake_sequence_size = replaybuffer.sample()
    fake_label = torch.zeros((*fake_data.shape[:-1], 1))
    fake = torch.cat((fake_data, fake_label), dim=2)

    # TODO save in memory to not load expert on every episode
    expert_data, expert_sequence_size = get_expert_data(expert, episodes)
    expert_label = torch.ones((*expert_data.shape[:-1], 1))
    expert = torch.cat((expert_data, expert_label), dim=2)

    # FIXME add padding to the dataset with less entries (if fake_data > expert_data)
    fake_size = fake.shape[1]
    expert_size = expert.shape[1]
    padding_size = abs(expert_size - fake_size)

    if expert_size - fake_size > 0:
        padding = torch.zeros((episodes, padding_size, fake.shape[-1]))
        fake = torch.cat((fake, padding), dim=1)
    elif fake_size - expert_size > 0:
        padding = torch.ones((episodes, padding_size, expert.shape[-1]))
        expert = torch.cat((expert, padding), dim=1)

    sequence_size = fake_sequence_size + expert_sequence_size
    data = torch.cat((fake, expert), dim=0)
    data = DiscriminatorDataset(data, sequence_size)
    dataloader = DataLoader(data, shuffle=True, batch_size=batch_size)

    alpha = []
    batch_acc = []
    batch_loss = []

    print('\nStarting mini_batch - fake data, etc')
    for mini_batch in tqdm(dataloader):
        data, label, sequence_size, index = mini_batch
        data = data.to(device)
        label = label.to(device)
        d_optimizer.zero_grad()
        
        fake_data_index = torch.unique(torch.where(label == 0)[0])
        fake_data = data[fake_data_index]
        b, s, f = fake_data.shape

        true_data_index = torch.unique(torch.where(label == 1)[0])
        data = data[true_data_index]
        for g_data, seq in zip(fake_data, sequence_size[fake_data_index]):
            g_data = g_data[:seq]
            pred, new_state = policy(g_data)
            new_state = torch.cat((g_data[0][None], new_state[:-1]), dim=0)

            padding_size = s - seq
            padding = torch.zeros((padding_size, f)).to(device)
            new_state = torch.cat((new_state, padding), dim=0)
            data = torch.cat((data, new_state[None]), dim=0)
        
        pred = discriminator(data, sequence_size)
        pred_argmax = torch.argmax(pred, 2)

        pred = pred.reshape(-1, 2)
        _label = label.reshape(-1)
        loss = d_criterion(pred, _label.long())
        loss.backward(retain_graph=True)
        batch_loss.append(loss.item())
        d_optimizer.step()

        acc = ((torch.argmax(pred, 1) == _label).sum().item() / _label.shape[0]) * 100
        batch_acc.append(acc)

        # TODO plot how many fake trajectories we are using
        # All fake data
        label_index = np.unique(np.where(label.cpu() == 0)[0])

        # All data that the discriminator thinks its real
        data_index = np.unique(np.where(pred_argmax.cpu() == 1)[0])

        # All fake data that the discriminator thinks its real
        alpha_index = np.intersect1d(label_index, data_index)
        alpha += index[alpha_index].tolist()

    return (replaybuffer.get_by_index(alpha)), np.mean(aer), (np.mean(batch_acc), np.mean(batch_loss))
