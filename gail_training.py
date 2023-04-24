import argparse
from collections import defaultdict
import logging
import os

logging.getLogger("tensorflow").setLevel(logging.ERROR)

import gym
import numpy as np
from stable_baselines import GAIL
from stable_baselines.gail import ExpertDataset
from stable_baselines.common.callbacks import EvalCallback
from tqdm import tqdm

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
    # # Load the expert dataset
    dataset = ExpertDataset(expert_path=args.data, traj_limitation=100, verbose=0)
    
    evals = defaultdict(dict)
    for i in tqdm(range(5)):
        env = gym.make(args.env)
        eval_callback = EvalCallback(
            env, 
            best_model_save_path=f'./logs/{args.env}',
            log_path=f'./logs/{args.env}',
            eval_freq=500,
            deterministic=True, 
            render=False,
            verbose=0
        )

        model = GAIL('MlpPolicy', env, dataset, verbose=1)
        model.learn(total_timesteps=1000000, callback=eval_callback)

        del model # remove to demonstrate saving and loading

        model = GAIL.load(f"./logs/{args.env}/best_model")

        rewards = []
        for episode in tqdm(range(100)):
            obs, done = env.reset(), False
            total_reward = 0
            
            while not done:
                action, _states = model.predict(obs)
                obs, reward, done, info = env.step(action)
                total_reward += reward

            rewards.append(total_reward)
        evals[i]["mean"] = np.mean(rewards)
        evals[i]["std"] = np.std(rewards)

        if not os.path.exists("./results/"):
            os.makedirs("./results/")

        if not os.path.exists(f"./results/gail_{args.env}.txt"):
            with open(f"./results/gail_{args.env}.txt", "w") as f:
                f.write("iter;mean;std\n")
                for k, v in evals.items():
                    f.write(f"{k};{v['mean']};{v['std']}\n")
        else:
            with open(f"./results/gail_{args.env}.txt", "a") as f:
                for k, v in evals.items():
                    f.write(f"{k};{v['mean']};{v['std']}\n")