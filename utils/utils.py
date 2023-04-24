from collections import defaultdict
import os
from os import listdir
from os.path import isfile, join
import random
import shutil

import gym
from gym.wrappers import TimeLimit
import numpy as np
import torch

from datasets.Dataset import get_idm_dataset as idm
from datasets.Dataset import get_policy_dataset as policy
from datasets.Dataset import get_idm_vector_dataset as idm_vector
from datasets.Dataset import get_policy_vector_dataset as policy_vector
from Models.IDM import IDM
from Models.Policy import Policy
from utils.enjoy import delete_alpha
from utils.enjoy import get_environment
from utils.enjoy import performance
from utils.enjoy import play_vector


def load_policy_model(args, environment, device, folder=None):
    parent_folder = './checkpoint/policy'
    path = folder if folder is not None else parent_folder

    model = Policy(
        environment['action'],
        net=args.encoder,
        pretrained=args.pretrained,
        input=environment['input_size']
    )
    model.load_state_dict(torch.load(f'{path}/best_model.ckpt'))
    model = model.to(device)
    model.eval()
    return model


def load_idm_model(args, folder=None):
    parent_folder = './checkpoint/policy'
    path = f'{parent_folder}/{folder}' if folder is not None else parent_folder

    model = IDM(args.actions, pretrained=args.pretrained)
    model.load_state_dict(torch.load(f'{path}/best_model.ckpt'))
    model = model.to(device)
    model.eval()
    return model


def save_gif(gif_images, random, iteration):
    if os.path.exists('./gifs/') is False:
        os.makedirs('./gifs/')

    name = 'Random' if random else 'Sample'
    gif_images[0].save(f'./gifs/{name}_{iteration}.gif',
                       format='GIF',
                       append_images=gif_images[1:],
                       save_all=True,
                       duration=100,
                       loop=0)


def policy_infer(
    model,
    dataloader,
    device,
    domain,
    random=False,
    size=(10, 10),
    episodes=10,
    seed=None,
    bar=None,
    verbose=False,
    dataset=False,
    gif=True,
    alpha_location=None,
    args=None,
):

    _dataset = dataloader.dataset
    transforms = _dataset.transforms

    if model.training:
        model.eval()

    if dataset:
        delete_alpha(alpha_location, domain)

    count = 0
    total_solved = 0
    exploration = 0
    reward_epoch = []
    performance_epoch = []
    states = np.ndarray((0, 4), dtype=str)
    for e in range(episodes):
        _seed = e + 1 if seed is None else seed
        env = get_environment(domain, size, _seed, random)

        play_function = domain['enjoy']
        epoch_data = play_function(
            env,
            model,
            dataset,
            gif,
            count,
            alpha_location,
            transforms,
            device,
            states,
            domain,
            args,
        )
        total_reward, gif_images, count, goal, exploration_ratio = epoch_data
        total_solved += goal
        exploration += exploration_ratio

        if gif is True:
            save_gif(gif_images, random, e)

        if random is False:
            performance_epoch.append(
                performance(total_reward, _seed, _dataset))
        reward_epoch.append(total_reward)

        if verbose:
            print(f'{e}/{episodes} - Total Reward: {total_reward}')

        if bar is not None:
            bar.next()

        env.close()
        del env

    return np.mean(reward_epoch), np.mean(performance_epoch), total_solved / episodes, exploration / episodes


def create_alpha_dataset(idm, alpha, domain, ratio=None):
    try:
        alpha_file = open(f'{alpha}{domain["name"]}.txt', 'r')

        classes_alpha = defaultdict(list)
        for line in alpha_file:
            words = line.replace('\n', '').split(';')
            classes_alpha[words[-1]].append(words)
        alpha_file.close()
    except FileNotFoundError:
        alpha_file = []
        classes_alpha = defaultdict(list)

    idm_file = open(f'{idm}{domain["name"]}.txt', 'r')
    classes_idm = defaultdict(list)
    for line in idm_file:
        words = line.replace('\n', '').split(';')
        classes_idm[words[-1]].append(words)
    idm_file.close()

    ratio = 1 if ratio is None else 1 - ratio
    for key in range(domain['action']):
        k = max(0, int(len(classes_idm[str(key)]) * ratio))
        classes_alpha[str(key)] += random.sample(classes_idm[str(key)], k)

    if os.path.exists(alpha) is False:
        os.makedirs(alpha)

    with open(f'{alpha}{domain["name"]}.txt', 'w') as f:
        for key in classes_alpha:
            for row in classes_alpha[str(key)]:
                if 'previous' in row[0] and 'maze' in domain['name']:
                    f.write(
                        f'{idm}images/{row[0]};{idm}images/{row[1]};{row[-2]};{row[-1]}\n')
                elif 'prev' in row[0]:
                    f.write(
                        f'{idm}{row[0]};{idm}{row[1]};{row[-1]};{row[-1]}\n')
                else:
                    f.write(
                        f'{alpha}images/{row[0]};{alpha}images/{row[1]};{row[-1]};{row[-1]}\n')


def create_alpha_vector_dataset(idm, alpha, domain, ratio=None):
    try:
        alpha_file = open(f'{alpha}{domain["name"]}.txt', 'r')

        classes_alpha = defaultdict(list)
        for line in alpha_file:
            words = line.replace('\n', '').split(';')
            classes_alpha[words[-1]].append(line)
        alpha_file.close()
    except FileNotFoundError:
        alpha_file = []
        classes_alpha = defaultdict(list)

    idm_file = open(f'{idm}{domain["name"]}.txt', 'r')
    classes_idm = defaultdict(list)
    for line in idm_file:
        words = line.replace('\n', '').split(';')
        classes_idm[words[-1]].append(line)
    idm_file.close()

    for key in range(domain['action']):
        if ratio == 0:
            k = int(len(classes_idm[str(key)]))
        else:
            action_size = (len(classes_alpha[str(key)]) * 100) / (ratio * 100)
            k = int(action_size - len(classes_alpha[str(key)]))

        size = len(classes_alpha[str(key)])
        classes_alpha[str(
            key)] += list(np.random.choice(classes_idm[str(key)], k))

        print(key, ratio, k, size, len(classes_alpha[str(key)]))

    if os.path.exists(alpha) is False:
        os.makedirs(alpha)

    with open(f'{alpha}{domain["name"]}.txt', 'w') as f:
        for key in classes_alpha:
            f.write(''.join(classes_alpha[str(key)]))

def read_data(path: str, domain: dict):
    try:
        file = open(f'{path}{domain["name"]}.txt', 'r')
        classes = defaultdict(list)
        for line in file:
            words = line.replace('\n', '').split(';')
            classes[words[-1]].append(line)
        file.close()
    except FileNotFoundError:
        file = []
        classes = defaultdict(list)
    return file, classes


class TimeFeatureWrapper(gym.Wrapper):
    """
    Add remaining time to observation space for fixed length episodes.
    See https://arxiv.org/abs/1712.00378 and https://github.com/aravindr93/mjrl/issues/13.
    :param env: (gym.Env)
    :param max_steps: (int) Max number of steps of an episode
        if it is not wrapped in a TimeLimit object.
    :param test_mode: (bool) In test mode, the time feature is constant,
        equal to zero. This allow to check that the agent did not overfit this feature,
        learning a deterministic pre-defined sequence of actions.
    """

    def __init__(self, env, max_steps=1000, test_mode=False):
        assert isinstance(env.observation_space, gym.spaces.Box)
        # Add a time feature to the observation
        low, high = env.observation_space.low, env.observation_space.high
        low, high = np.concatenate((low, [0])), np.concatenate((high, [1.]))
        env.observation_space = gym.spaces.Box(
            low=low, high=high, dtype=np.float32)

        super(TimeFeatureWrapper, self).__init__(env)

        if isinstance(env, TimeLimit):
            self._max_steps = env._max_episode_steps
        else:
            self._max_steps = max_steps
        self._current_step = 0
        self._test_mode = test_mode

    def reset(self):
        self._current_step = 0
        return self._get_obs(self.env.reset())

    def step(self, action):
        self._current_step += 1
        obs, reward, done, info = self.env.step(action)
        return self._get_obs(obs), reward, done, info

    def _get_obs(self, obs):
        """
        Concatenate the time feature to the current observation.
        :param obs: (np.ndarray)
        :return: (np.ndarray)
        """
        # Remaining time is more general
        time_feature = 1 - (self._current_step / self._max_steps)
        if self._test_mode:
            time_feature = 1.0
        # Optionnaly: concatenate [time_feature, time_feature ** 2]
        return np.concatenate((obs, [time_feature]))


def calculate_amount(epoch, max_epoch, k):
    if epoch == max_epoch:
        return 1
    else:
        return 1 - (1 / (1 + np.power(((max_epoch/epoch) - 1), k)))


def create_alpha_dataset_new(
        idm_dataset: str,
        alpha_dataset: str,
        domain: dict,
        epoch: int,
        max_epoch: int = 100,
        k: int = -2,
        ratio: float = None,
        board=None
):
    _, i_pos = read_data(alpha_dataset, domain)
    _, i_pre = read_data(idm_dataset, domain)

    upper_limit = calculate_amount(epoch, max_epoch, k)
    pos_limit = min(upper_limit, ratio)
    pre_limit = 1 - pos_limit

    ipre_amount = []
    ipos_amount = []
    i = defaultdict(list)
    print('Alpha:')
    for action in range(get_environment(domain).action_space.n):
        amount = int(len(i_pre[str(action)]) * pre_limit)
        action_i_pre = list(np.random.choice(
            i_pre[str(action)], amount, replace=False))
        ipre_amount.append(amount)

        amount = int(len(i_pos[str(action)]) * pos_limit)
        action_i_pos = list(np.random.choice(
            i_pos[str(action)], amount, replace=False))
        ipos_amount.append(amount)

        i[str(action)] += action_i_pre
        i[str(action)] += action_i_pos
        print(
            f'Action {action}: {len(action_i_pre)} (Ipre) {len(action_i_pos)} (Ipos) - {len(i[str(action)])} (total)')

    if os.path.exists(alpha_dataset) is False:
        os.makedirs(alpha_dataset)

    with open(f'{alpha_dataset}{domain["name"]}.txt', 'w') as f:
        for key in i.keys():
            f.write(''.join(i[str(key)]))

    if board is not None:
        board.add_scalars(
            train=False,
            Limit=upper_limit,
            Ipre_limit=pre_limit,
            Ipos_limit=pos_limit,
            Ipre_amount=np.mean(ipre_amount),
            Ipos_amount=np.mean(ipos_amount)
        )


domain = {
    'vector': {
        'idm_dataset': idm_vector,
        'policy_dataset': policy_vector,
        'enjoy': play_vector,
        'alpha': create_alpha_vector_dataset
    }
}
