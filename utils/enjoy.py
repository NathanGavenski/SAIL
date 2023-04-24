from collections import defaultdict
import os
import shutil

import gym
import numpy as np
from PIL import Image
import torch


class EnjoyController():
    def __init__(self, max_count):
        self.prev_state = defaultdict(int)
        self.max_count = max_count

    def append(self, state):
        state = tuple(state)
        self.prev_state[state] += 1
        return True if self.prev_state[state] > self.max_count else False


def get_min_reward():
    return -1


def define_goal(domain, gym_return, total_reward, env):
    location, reward, done, info = gym_return
    if domain['name'] == 'maze':
        return True if reward == 1 else False
    elif domain['name'] == 'cartpole':
        return True if total_reward + reward >= 195 else False
    elif domain['name'] == 'mountaincar':
        return True if location[0] >= 0.5 else False
    elif domain['name'] == 'acrobot':
        return True if reward == 0 else False
    elif domain['name'] == 'lunarlander':
        return not env.lander.awake and not (env.game_over or abs(location[0]) >= 1.0)
    elif domain['name'] == 'reacher':
        pass
    elif domain['name'] == 'ant':
        pass
    else:
        return True


def save_alpha(datas, count, alpha_location, domain):
    path = f'./dataset/{domain["name"]}/POLICY/alpha/' if alpha_location is None else alpha_location

    image_path = f'{path}images/'
    if os.path.exists(image_path) is False:
        os.makedirs(image_path)

    file = f'{path}{domain["name"]}.txt'
    if os.path.exists(file):
        f = open(file, 'a')
    else:
        f = open(file, 'w')

    for i, data in enumerate(datas):
        state, action = data
        s = Image.fromarray(state)
        s.save(f'{image_path}state_{count + i}.png')

        if i + 1 < len(datas):
            f.write(f'state_{count + i}.png;state_{count + (i + 1)}.png;{action.item()};{action.item()}\n')

    f.close()

    return count + len(datas)


def save_vector_alpha(datas, count, alpha_location, domain):
    path = f'./dataset/{domain["name"]}/POLICY/alpha/' if alpha_location is None else alpha_location

    image_path = f'{path}images/'
    if os.path.exists(image_path) is False:
        os.makedirs(image_path)

    file = f'{path}{domain["name"]}.txt'
    if os.path.exists(file):
        f = open(file, 'a')
    else:
        f = open(file, 'w')

    for i, data in enumerate(datas):
        state, action = data
        try:
            nState, nAction = datas[i + 1]

            if isinstance(action, torch.Tensor):
                action = action.item()

            f.write(f'{state.tolist()};{nState.tolist()};{action}\n')
        except IndexError:
            continue

    f.close()
    return count + len(datas)


def delete_alpha(alpha_location, domain):
    path = f'./dataset/{domain["name"]}/POLICY/alpha/' if alpha_location is None else alpha_location
    if os.path.exists(f'{path}images/') is True:
        shutil.rmtree(f'{path}images/', ignore_errors=True)

    if os.path.exists(f'{path}{domain["name"]}.txt'):
        os.remove(f'{path}{domain["name"]}.txt')


def performance(reward, maze_version, dataset):
    sample, random = dataset.get_performance_rewards(maze_version)
    return (reward - random) / (sample - random)


def play(
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
):
    location = env.reset()
    controller = EnjoyController(3)
    done, goal = False, False

    gif_images = []
    total_reward = 0
    run = np.ndarray((0, 2))

    step_count = 0
    exploration_count = 0

    while done is False:
        step_count += 1
        early_stop = controller.append(location)
        s = env.render('rgb_array')
        s_image = Image.fromarray(s).convert('RGB')

        if gif is True:
            gif_images.append(s_image)

        s = transforms(s_image)
        s = s.view(1, *s.shape)
        s = s.to(device)

        prediction = model(s)

        if args.choice == 'max':
            action = torch.argmax(prediction, 1)
        elif args.choice == 'weighted':
            classes = np.arange(domain['action'])
            prob = torch.nn.functional.softmax(prediction, dim=1).cpu().detach().numpy()
            action = np.random.choice(classes, p=prob[0])

            if action != torch.argmax(prediction, 1):
                exploration_count += 1

            action = torch.tensor(action)
            action = action.to(device)

        gym_return = env.step(int(action.double()))
        location, reward, done, info = gym_return
        goal = define_goal(domain, gym_return, total_reward, env)
        total_reward += reward

        if dataset is True:
            data = np.array((np.array(s_image), action))
            run = np.append(run, data.reshape((1, *data.shape)), axis=0)

        if done is True and gif is True or dataset is True:
            nS = env.render('rgb_array')
            nS = Image.fromarray(nS).convert('RGB')

            if gif is True:
                gif_images.append(nS)

            if dataset is True:
                data = np.array((np.array(nS), action))
                run = np.append(run, data.reshape((1, *data.shape)), axis=0)

        if early_stop is True and domain['name'] == 'maze':
            total_reward = get_min_reward()
            done = True
            break

    goal = True
    if dataset is True and goal is True:
        save_alpha(
            run,
            count,
            alpha_location,
            domain,
        )

    count += len(run)
    goal = 1 if goal is True else 0
    return total_reward, gif_images, count, goal, exploration_count / step_count


def play_vector(
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
):
    location = env.reset()
    controller = EnjoyController(3)
    done, goal = False, False

    gif_images = []
    total_reward = 0
    run = np.ndarray((0, 2))

    step_count = 0
    exploration_count = 0

    while done is False:
        step_count += 1
        early_stop = controller.append(location)

        if gif is True:
            s = env.render('rgb_array')
            s_image = Image.fromarray(s).convert('RGB')
            gif_images.append(s_image)

        s = transforms(location)
        s = s.view(1, *s.shape)
        s = s.to(device)

        prediction = model(s)
        
        if args.choice == 'max':
            action = torch.argmax(prediction, 1)
        elif args.choice == 'weighted':
            classes = np.arange(get_environment(domain).action_space.n)
            prob = torch.nn.functional.softmax(prediction, dim=1).cpu().detach().numpy()
            action = np.random.choice(classes, p=prob[0])

            if action != torch.argmax(prediction, 1):
                exploration_count += 1
            
            action = torch.tensor(action)
            action = action.to(device)

        gym_return = env.step(int(action.double()))
        location, reward, done, info = gym_return
        goal = define_goal(domain, gym_return, total_reward, env)
        total_reward += reward

        if dataset is True:
            data = np.array((location, action))
            run = np.append(run, data.reshape((1, *data.shape)), axis=0)

        if done is True and gif is True:
            nS = env.render('rgb_array')
            nS = Image.fromarray(nS).convert('RGB')
            gif_images.append(nS)

        if early_stop is True and domain['name'] == 'maze':
            total_reward = get_min_reward()
            done = True
            break

    if dataset is True and goal is True:
        save_vector_alpha(
            run,
            count,
            alpha_location,
            domain,
        )

    count += len(run)
    goal = 1 if goal is True else 0
    return total_reward, gif_images, count, goal,exploration_count / step_count


def get_environment(domain, size=None, seed=None, random=None):
    if domain['name'] == 'cartpole':
        return gym.make('CartPole-v1')
    elif domain['name'] == 'mountaincar':
        return gym.make('MountainCar-v0')
    elif domain['name'] == 'acrobot':
        return gym.make('Acrobot-v1')
    elif domain['name'] == 'coinrun':
        setup_utils.setup_and_load(set_seed=1, use_cmd_line_args=False)
        env = make('standard', 96)
        return env
    elif domain['name'] == 'lunarlander':
        return gym.make('LunarLander-v2')
    elif domain['name'] == 'maze':
        if random:
            return gym.make(f"maze-random-{size[0]}x{size[1]}-v0")
        else:
            return gym.make(
                f"maze-sample-{size[0]}x{size[1]}-v1",
                version=seed
            )
