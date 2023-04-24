from argparse import ArgumentParser
from collections import defaultdict
import os
from multiprocessing import Process
from typing import Callable
import sys

from huggingface_sb3 import load_from_hub
import numpy as np
from stable_baselines3 import DQN, PPO
from tqdm import tqdm
import psutil

import warnings
warnings.filterwarnings("ignore")

from sys import platform
if platform != "win32":
    import ctypes
    ctypes.CDLL('libX11.so.6').XInitThreads()


envs = {
     'mountain-car': {
         'name': 'MountainCar-v0',
         'repo_id': 'sb3/dqn-MountainCar-v0',
         'filename': 'dqn-MountainCar-v0.zip',
         'threshold': -108.,
         'algo': DQN
    },

    'cartpole': {
         'name': 'CartPole-v1',
         'repo_id': 'sb3/ppo-CartPole-v1',
         'filename': 'ppo-CartPole-v1.zip',
         'threshold': 475.,
         'algo': PPO
    },

    'acrobot': {
         'name': 'Acrobot-v1',
         'repo_id': 'sb3/dqn-Acrobot-v1',
         'filename': 'dqn-Acrobot-v1.zip',
         'threshold': -100.,
         'algo': DQN
    },

    'lunar-lander': {
         'name': 'LunarLander-v2',
         'repo_id': 'sb3/dqn-LunarLander-v2',
         'filename': 'dqn-LunarLander-v2.zip',
         'threshold': 200.,
         'algo': DQN
    }
}


class Controller():
    def __init__(self, threads: int = 1, func: Callable = None, amount: int = 100) -> None:
        self.threads = threads
        self.function = func
        self.amount = amount
        self.pbar = tqdm(range(self.amount))
        self.cpus = {}

        for idx in range(self.threads):
            self.cpus[idx] = False

    def __free_cpu(self, idx: int) -> None:
        self.cpus[idx] = False
        return True

    def __lock_cpu(self, idx: int) -> None:
        self.cpus[idx] = True

    def get_first_free_cpu_and_lock(self) -> int:
        for key, value in self.cpus.items():
            if not value:
                self.__lock_cpu(key)
                return key

    def create_folder(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def run(self, opt) -> None:
        self.create_folder(f'./datasets/{opt.game}/')
        self.create_folder(f'./logs/')

        threads, ep = {}, 0
        while ep < self.amount:
            for key in [key for key, t in threads.items() if not t.is_alive()]:
                self.pbar.update(1)
                self.__free_cpu(key)

            if len({key: t for key, t in threads.items() if t.is_alive()}.items()) < self.threads:
                cpu = self.get_first_free_cpu_and_lock()
                if cpu is None:
                    cpu = [key for key, t in threads.items() if not t.is_alive()][0]

                kwargs = {
                    'game': opt.game,
                    'threshold': opt.threshold,
                    'idx': ep,
                    'cpu': cpu
                }

                thread = Process(target=self.function, kwargs=kwargs)
                thread.start()
                threads[cpu] = thread
                ep += 1

        while len([key for key, t in threads.items() if t.is_alive()]) > 0:
            for key in [key for key, t in threads.items() if not t.is_alive()]:
                self.pbar.update(1)
                self.__free_cpu(key)
                threads = {key: t for key, t in threads.items() if t.is_alive()}


def run(game, amount: int = 1, threshold: float = None, idx: int = None, cpu: int = None) -> None:
    import os
    import gym

    if platform != "win32":
        proc = psutil.Process()
        proc.cpu_affinity([int(cpu)])
        os.sched_setaffinity(proc.pid, [int(cpu)])
        sys.stdout = open(f'./logs/{cpu}.txt', 'w')
        os.system("export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so")

    game = envs[game]
    env_name = game['name']

    checkpoint = load_from_hub(
        repo_id=game['repo_id'],
        filename=game['filename'],
    )

    threshold = threshold if threshold is not None else game['threshold']

    custom_objects = {
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0
    }
    policy = game['algo'].load(checkpoint, custom_objects=custom_objects)

    env = gym.make(env_name)
    state_shape = env.reset().shape

    ep_returns = defaultdict(int)

    states = np.ndarray((0, *state_shape))
    next_states = np.ndarray((0, *state_shape))
    actions = []
    ep = 0

    while ep < amount:
        obs, done = env.reset(), False
        tmp_states = np.ndarray((0, *state_shape))
        tmp_next_states = np.ndarray((0, *state_shape))
        tmp_actions = []
        internal_states = None
        while not done:
            tmp_states = np.append(tmp_states, obs[None], axis=0)
            action, internal_states = policy.predict(
                obs,
                state=internal_states,
                deterministic=True,
            )
            tmp_actions.append(action)

            obs, reward, done, _ = env.step(action)
            env.render()

            tmp_next_states = np.append(tmp_next_states, obs[None], axis=0)

            ep_returns[ep] += reward

        if ep_returns[ep] >= threshold:
            states = np.append(states, tmp_states, axis=0)
            next_states = np.append(next_states, tmp_next_states, axis=0)
            actions += tmp_actions
            ep_returns[ep] = ep_returns[ep]
            ep += 1

            dataset = {
                'state': states,
                'actions': actions,
                'next_states': next_states,
            }
            avg_reward = np.mean(ep_returns[ep - 1])
            np.savez(f'./datasets/{opt.game}/tmp_{env_name}_{idx}_{avg_reward}', **dataset)

        else:
            ep_returns[ep] = 0

def create_data_file(game, opt, shape=(0, 84, 84, 4)):
    from os import listdir
    from os.path import isfile, join
    import gym

    game = envs[game]
    env = gym.make(game['name'])
    shape = env.reset().shape
    path = f'./datasets/{opt.game}/'
    files = [join(path, f) for f in listdir(path) if isfile(join(path, f)) and ('npz' in f and 'tmp' in f)]

    states = np.ndarray((0, *shape))
    next_states = np.ndarray((0, *shape))
    actions = np.ndarray((0, *env.action_space.shape))
    starts = []
    rewards = []
    for f in tqdm(files):
        data = np.load(f, allow_pickle=True)
        states = np.append(states, data['state'], axis=0)
        next_states = np.append(next_states, data['next_states'], axis=0)
        actions = np.append(actions, data['actions'], axis=0)
        reward = float(f.split('_')[-1].replace('.npz', ''))
        rewards.append(reward)
        start = np.zeros(data['state'].shape[0])
        start[0] = 1
        starts += start.astype(bool).tolist()

    np.savez(
        f"./datasets/{opt.game}/{game['name']}",
        **{
            'states': states,
            'obs': states,
            'next_states': next_states,
            'actions': actions,
            'episode_starts': starts,
            'expert': np.mean(rewards),
            'episode_returns': rewards,
            'std': np.std(rewards)
        }
    )

    for f in files:
        os.remove(f)


if __name__ == "__main__":
    # xvfb-run --auto-servernum --server-num=1 python create_dataset.py -t 10 -e 100 -g beamrider
    parser = ArgumentParser()

    parser.add_argument(
        "-g", "--game", type=str, help="env name",
    )
    parser.add_argument(
        "-e", "--episodes", default=10, type=int, help="number of episodes"
    )
    parser.add_argument(
        "-t", "--threads", default=1, type=int, help="how many workers should the process execute",
    )
    parser.add_argument(
        "--threshold", type=float, default=None, help="reward threshold for each execution",
    )
    parser.add_argument(
        "--mode", default="play", type=str, help="reward threshold for each execution",
    )

    opt = parser.parse_args()

    if opt.mode == "play":
        controller = Controller(opt.threads, run, opt.episodes)
        controller.run(opt)
    elif opt.mode == "data":
        create_data_file(opt.game, opt)
