from copy import deepcopy
import os

import gym
import numpy as np
import os

from utils.args import get_args

"""
To create a new random dataset you just need to inform env_name (need to be the gym name environment)
and where you want to save the random dataset (e.g., ./dataset/lunarlander/IDM_VECTOR/lunarlander.txt)

Ex: python create_random.py --env_name LunarLander-v2 --data_path ./dataset/lunarlander/IDM_VECTOR/lunarlander.txt
"""

if __name__ == '__main__':
    args = get_args()
    env = gym.make(args.env_name)
    action_size = env.action_space.n
    env.reset()

    samples = []
    state = env.reset()
    for i in range(action_size * 5000):
        action = env.action_space.sample()
        next_state, _, done, _ = env.step(action)
        entry = [state.tolist(), next_state.tolist(), action]
        samples.append(entry)
        
        state = env.reset() if done else deepcopy(next_state)
            
    path = '/'.join(args.data_path.split('/')[:-1])
    if not os.path.exists(path):
        os.makedirs(path)

    with open(args.data_path, 'w') as f:
        for entry in samples:
            f.write(f'{entry[0]};{entry[1]};{entry[2]}\n')
