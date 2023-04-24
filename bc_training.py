import argparse
from collections import defaultdict
import os

import gym
import numpy as np
from stable_baselines import GAIL, SAC
from stable_baselines.gail import ExpertDataset
from stable_baselines.common.callbacks import EvalCallback
from tqdm import tqdm

import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)


def performance(values, random, expert):
    return (np.array(values) - random) / (expert - random)

def get_args():
    parser = argparse.ArgumentParser(description='Args for general configs')
    parser.add_argument('--env')
    parser.add_argument('--data')
    parser.add_argument('--batch', type=int)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    dataset = ExpertDataset(
        expert_path=args.data,
        batch_size=args.batch,
        verbose=1
    )
    print()

    evals = defaultdict(dict)
    for i in tqdm(range(5)):
        env = gym.make(args.env)

        model = GAIL('MlpPolicy', args.env, dataset, verbose=1)
        model.pretrain(dataset, n_epochs=1000000)

        rewards = []
        for episode in range(100):
            obs, done = env.reset(), False
            total_reward = 0
            
            while not done:
                action, _states = model.predict(obs)
                obs, reward, done, info = env.step(action)
                total_reward += reward
            
            rewards.append(total_reward)
        evals[i]["mean"] = np.mean(rewards)
        evals[i]["std"] = np.std(rewards)
        del model
        del env

        if not os.path.exists("./results/"):
            os.makedirs("./results/")

        if not os.path.exists(f"./results/bc_{args.env}.txt"):
            with open(f"./results/bc_{args.env}.txt", "w") as f:
                f.write("iter;mean;std\n")
                for k, v in evals.items():
                    f.write(f"{k};{v['mean']};{v['std']}\n")
        else:
            with open(f"./results/bc_{args.env}.txt", "a") as f:
                for k, v in evals.items():
                    f.write(f"{k};{v['mean']};{v['std']}\n")