# Args should be imported before everything to cover https://discuss.pytorch.org/t/cuda-visible-device-is-of-no-use/10018
from utils.args import get_args
import os

import numpy as np
from progress.bar import Bar
import torch
from torch import nn, optim

from Models.IDM import IDM
from Models.IDM import train as train_idm
from Models.IDM import validation as validate_idm
from Models.General.LSTM import LSTM
from Models.Policy import Policy
from Models.Policy import train as train_policy
from Models.Policy import validation as validate_policy
from utils.adversarial import adversarial
from utils.board import Board
from utils.enjoy import get_environment
from utils.utils import domain
from torch.utils.data import DataLoader

args = get_args()

# ARGS: GPU and Pretrained
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

environment = domain[args.domain]
if args.domain == 'vector':
    environment['name'] = args.env_name


# Tensorboard
parent_folder = "_".join(args.run_name.split('_')[:-1])
st_char = environment['name'][0].upper()
rest_char = environment['name'][1:]
if 'maze' in environment['name']:
    env_name = f'{st_char}{rest_char}{args.maze_size}'
else:
    env_name = f'{st_char}{rest_char}'

name = f'./checkpoint/alpha/{env_name}/{parent_folder}/{args.run_name}'
if os.path.exists(name) is False:
    os.makedirs(name)

parent_folder = "_".join(args.run_name.split('_')[:-1])
path = f'./runs/alpha/{env_name}/{parent_folder}/{args.run_name}'
if os.path.exists(path) is False:
    os.makedirs(path)

board = Board(name, path)

# Datasets
print('\nCreating PyTorch IDM Datasets')
print(f'Using dataset: {args.data_path} with batch size: {args.batch_size}')
get_idm_dataset = environment['idm_dataset']
idm_train, idm_validation = get_idm_dataset(
    args.data_path,
    args.batch_size,
    env_name=args.env_name,
    downsample_size=None,
    reducted=args.reducted,
    no_hit=args.no_hit,
    augment=args.augmented
)

print('\nCreating PyTorch Policy Datasets')
print(
    f'Using dataset: {args.expert_path} with batch size: {args.policy_batch_size}')
get_policy_dataset = environment['policy_dataset']
policy_train, policy_validation = get_policy_dataset(
    args.expert_path,
    args.policy_batch_size,
    downsample_size=args.expert_amount,
    maze_size=args.maze_size,
    maze_type=args.maze_type,
    augment=args.augmented,
    validation=False
)


# Model and action size
print('\nCreating Models')
_env = get_environment(environment)
action_dimension = _env.action_space.n
input_size = _env.reset().shape[0]
policy_model = Policy(
    action_dimension,
    net=args.encoder,
    pretrained=args.pretrained,
    input=input_size
)
idm_model = IDM(
    action_dimension,
    net=args.encoder,
    pretrained=args.pretrained,
    input=input_size * 2
)
discriminator_model = LSTM(
    input_dim=input_size,
    hidden_dim=32,
    num_layer=2,
    output_dim=2,
    device=device
)
del _env

if args.domain == 'vector':
    environment['action'] = action_dimension

if args.idm_pretrained:
    idm_model.load_state_dict(torch.load(args.idm_weights))

idm_model.to(device)
policy_model.to(device)
discriminator_model.to(device)

# Optimizer and loss
print('\nCreating Optimizer and Loss')
print(f'IDM learning rate: {args.lr}\nPolicy learning rate: {args.policy_lr}')
idm_lr = args.lr
idm_criterion = nn.CrossEntropyLoss()
idm_optimizer = optim.Adam(idm_model.parameters(), lr=idm_lr)

# FIXME maybe we need to split the parameters between both optimizers
policy_lr = args.policy_lr
policy_criterion = nn.CrossEntropyLoss()
policy_optimizer = optim.Adam(
    policy_model.model.parameters(),
    lr=policy_lr
)

# FIXME create generator args in utils/args
generator_lr = args.policy_lr
generator_criterion = nn.MSELoss()
generator_optimizer = optim.Adam(
    policy_model.generator.parameters(),
    lr=generator_lr
)

# FIXME create diiscriminator args in utils/args
discriminator_lr = args.policy_lr
discriminator_criterion = nn.CrossEntropyLoss()
discriminator_optimizer = optim.Adam(
    discriminator_model.parameters(),
    lr=discriminator_lr
)

# Train
print('Starting Train\n')

early_stop_count = 0
best_epoch_aer = -np.inf
best_epoch_performance = -np.inf

max_epochs = args.idm_epochs
policy_validation_len = len(policy_validation) if policy_validation is not None else 0
max_iter = len(idm_train) + len(idm_validation) + len(policy_train) + policy_validation_len

for epoch in range(max_epochs):
    print(f'Epoch: {epoch}/{max_epochs}')

    board.add_scalar('IDM Learning Rate', idm_optimizer.param_groups[0]['lr'])
    board.add_scalar('Policy Learning Rate',
                     policy_optimizer.param_groups[0]['lr'])

    # IDM Train
    if args.verbose:
        bar = Bar(f'EPOCH {epoch:3d}', max=max_iter, suffix='%(percent).1f%% - %(eta)ds')

    batch_acc = []
    batch_loss = []
    for itr, mini_batch in enumerate(idm_train):
        loss, acc = train_idm(
            idm_model,
            mini_batch,
            idm_criterion,
            idm_optimizer,
            device,
            board,
        )

        batch_acc.append(acc)
        batch_loss.append(loss.item())

        if args.verbose:
            bar.next()

        if args.debug:
            break

    # IDM Validation
    board.add_scalars(
        train=True,
        IDM_Loss=np.mean(batch_loss),
        IDM_Accuracy=np.mean(batch_acc)
    )

    batch_acc = []
    for itr, sample_batched in enumerate(idm_validation):
        acc = validate_idm(
            idm_model,
            mini_batch,
            device,
            board,
        )
        batch_acc.append(acc)

        if args.verbose is True:
            bar.next()

        if args.debug:
            break

    board.add_scalars(
        train=False,
        IDM_Accuracy=np.mean(batch_acc)
    )

    # Policy Train
    batch_acc = []
    batch_loss = []
    batch_g_loss = []
    for itr, mini_batch in enumerate(policy_train):
        (loss, acc), g_loss = train_policy(
            model=policy_model,
            idm_model=idm_model,
            data=mini_batch,
            criterion=policy_criterion,
            g_criterion=generator_criterion,
            optimizer=policy_optimizer,
            g_optimizer=generator_optimizer,
            device=device,
            args=args,
            actions=action_dimension,
            tensorboard=board
        )

        batch_acc.append(acc)
        batch_loss.append(loss.item())
        batch_g_loss.append(g_loss.item())

        if args.verbose is True:
            bar.next()

        if args.debug:
            break

    board.add_scalars(
        train=True,
        Policy_Loss=np.mean(batch_loss),
        Policy_Accuracy=np.mean(batch_acc),
        Generator_Loss=np.mean(batch_g_loss)
    )

    # Policy Validation
    if policy_validation is not None:
        batch_acc = []
        batch_g_dist = []
        for itr, mini_batch in enumerate(policy_validation):
            acc, g_dist = validate_policy(
                policy_model,
                idm_model,
                mini_batch,
                device,
                args,
                action_dimension,
                board
            )

            batch_acc.append(acc)
            batch_g_dist.append(g_dist)
            bar.next()

            if args.debug:
                break

        board.add_scalars(
            train=False,
            Policy_Accuracy=np.mean(batch_acc),
            Generator_Distance=np.mean(batch_g_dist)
        )

    # Policy Eval
    if args.debug:
        amount = 1
    else:
        amount = 100

    if args.verbose is True:
        bar = Bar(
            f'VALID Sample {epoch:3d}',
            max=amount,
            suffix='%(percent).1f%% - %(eta)ds'
        )
    else:
        bar = None

    alpha_data, infer, (acc, loss) = adversarial(
        policy=policy_model,
        discriminator=discriminator_model,
        episodes=amount,
        device=device,
        environment=environment,
        expert=args.expert_path,
        d_criterion=discriminator_criterion,
        d_optimizer=discriminator_optimizer,
        batch_size=args.adv_batch_size
    )

    board.add_scalars(
        train=False,
        AER_Sample=infer,
    )

    board.add_scalars(
        train=True,
        Discriminator_Loss=loss,
        Discriminator_Acc=acc
    )

    print(f'Sample Infer {infer}\n')

    if args.early_stop is True:
        if infer > best_epoch_aer:
            best_epoch_aer = infer if infer > best_epoch_aer else best_epoch_aer
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count >= 5:
            print('Early stop triggered')
            break

    if infer > best_epoch_aer:
        best_epoch_aer = infer if infer > best_epoch_aer else best_epoch_aer
        for model, name in zip([idm_model, policy_model], ['idm', 'policy']):
            env_name = environment['name']
            torch.save(
                model.state_dict(),
                f'./checkpoint/{env_name}_{name}.ckpt'
            )

    print(f'Using dataset: {args.alpha} with batch size: {args.batch_size}')

    print('Getting IDM dataset')

    alpha_state, alpha_next_state, alpha_action = alpha_data
    val_proportion = int(len(alpha_state) * 0.3)

    idm_train_dataset = idm_train.dataset
    idm_validation_dataset = idm_validation.dataset

    if val_proportion > 0:
        idm_train_dataset.previous_images = np.append(idm_train_dataset.previous_images, alpha_state[:val_proportion], axis=0)
        idm_train_dataset.next_images = np.append(idm_train_dataset.next_images, alpha_next_state[:val_proportion], axis=0)
        idm_train_dataset.labels = np.append(idm_train_dataset.labels, alpha_action[:val_proportion], axis=0)

        idm_validation_dataset.previous_images = np.append(idm_validation_dataset.previous_images, alpha_state[val_proportion:], axis=0)
        idm_validation_dataset.next_images = np.append(idm_validation_dataset.next_images, alpha_next_state[val_proportion:], axis=0)
        idm_validation_dataset.labels = np.append(idm_validation_dataset.labels, alpha_action[val_proportion:], axis=0)

    idm_train = DataLoader(idm_train_dataset, batch_size=args.batch_size, shuffle=True)
    idm_validation = DataLoader(idm_validation_dataset, batch_size=args.batch_size, shuffle=True)

    # Necessary updates
    board.advance()
    policy_validation_len = len(policy_validation) if policy_validation is not None else 0
    max_iter = len(idm_train) + len(idm_validation) + len(policy_train) + policy_validation_len

board.close()
