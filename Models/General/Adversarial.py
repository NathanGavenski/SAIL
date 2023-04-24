import gym
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# env = gym.make('CartPole-v1')
# data = []
# d = []

# for i_episode in range(20):
#     observation = env.reset()
#     #data.append(observation)
#     for t in range(100):
#         env.render()
#         print(observation)
#         action = env.action_space.sample()
#         d = []
#         d = np.append(observation, action)
#         data.append(d)

#         observation, reward, done, info = env.step(action)

#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             break
# env.close()

# with open('CartPole-v1-data.pkl', 'wb') as f:
# 	pickle.dump(data, f)

with open('CartPole-v1-data.pkl', 'rb') as f:
    data = pickle.load(f)

# # train data

print(f'Example of data to be generated: {data[0]}')
print(f'Example of data to be generated: {data[0][-1]}')

# ######################################################################

g_input_size = 5
g_hidden_size = 150
g_output_size = len(data[0])-1

print('\n'+10*'-')
print(f'Generator ' +
      '\n----------' +
      f'\nInput Size: {g_input_size}' +
      f'\nHidden size: {g_hidden_size}' +
      f'\nOutput Size: {g_output_size}')


d_input_size = len(data[0])-1
d_hidden_size = 75
d_output_size = 1

print('\n\n'+10*'-')
print(f'Discriminator ' +
      '\n----------' +
      f'\nInput Size: {d_input_size}' +
      f'\nHidden size: {d_hidden_size}' +
      f'\nOutput Size: {d_output_size}')

print('\n' + 50 * '=')

d_minibatch_size = 15
g_minibatch_size = 10
num_epochs = 1000
print_interval = 100

d_learning_rate = 3e-3
g_learning_rate = 9e-3

data_min_value = np.min(data)
data_max_value = np.max(data)


# print(random.choice(data))
# print(np.min(data))
# print(np.max(data))
#print(random.uniform(np.min(data)-1, np.max(data)+1))

def get_real_sampler(data, batch_size):
    #print(random.choices(data, k=batch_size))
    return random.choices(data, k=batch_size)

# data = [[Ação, -1.5, 0.5, 2.0, 1.3], [Ação, -1.5, 0.5, 2.0, 1.3], [Ação, -1.5, 0.5, 2.0, 1.3], [Ação, -1.5, 0.5, 2.0, 1.3]]
# def get_noise_sampler():
#     #return lambda m, n: torch.rand(m, n).requires_grad_()
#     return random.choice(data)


#noise_data  = get_noise_sampler()

#print(get_real_sampler(data, d_minibatch_size))
# print(noise_data)


class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()

        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        #self.xfer = torch.nn.SELU()
        self.xfer = torch.nn.LeakyReLU()

    def forward(self, x):
        x = self.xfer(self.map1(x))
        x = self.xfer(self.map2(x))

        return self.xfer(self.map3(x))


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()

        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        #self.elu = torch.nn.ELU()
        self.elu = torch.nn.LeakyReLU()

    def forward(self, x):
        x = self.elu(self.map1(x))
        x = self.elu(self.map2(x))
        return torch.sigmoid(self.map3(x))


# Create the networks
G = Generator(input_size=g_input_size, hidden_size=g_hidden_size,
              output_size=g_output_size)
print('Generator network was created.')


D = Discriminator(input_size=d_input_size,
                  hidden_size=d_hidden_size, output_size=d_output_size)
print('Discriminator network was created.')

print(50*'=')

# Define optimizers
criterion = nn.BCELoss()
d_optimizer = optim.SGD(D.parameters(), lr=d_learning_rate)
g_optimizer = optim.SGD(G.parameters(), lr=g_learning_rate)
print('Optimizers have been set.')

print(50*'=')

X, y = [], []
for i in data:
    X.append(i[:-1])
    y.append([i[-1]])

max_epochs = 100
n_batches = 10

# Train model
D.zero_grad()
G.zero_grad()
generator_losses = []
epoch_generator_loss = []

discriminator_losses = []
epoch_discriminator_loss = []

for epoch in range(max_epochs):

    for i in range(n_batches):

        d_optimizer.zero_grad()

        # Local batches and labels
        local_X = X[i*n_batches:(i+1)*n_batches]
        local_y = y[i*n_batches:(i+1)*n_batches]

        # Discriminator Phase -> True Data
        real_decision = D(torch.Tensor(local_X))
        real_error = criterion(
            real_decision, torch.ones(n_batches, 1))  # ones = True
        real_error.backward()

        # Discriminator Phase -> Fake Data
        # Tenho que alterar aqui pra receber o fake e o real data!
        fake_decision = D(torch.Tensor(local_X))
        fake_error = criterion(fake_decision, torch.zeros(
            n_batches, 1))  # zeros = False
        fake_error.backward()
        d_optimizer.step()

        epoch_discriminator_loss.append(real_error.item() + fake_error.item())
        #print(f'|Discriminator| Epoch {epoch} | Batch {i} | Real_error: {real_error} | Fake_error: {fake_error}')

        g_optimizer.zero_grad()
        # here we are going to use the same data
        noise = data[i*n_batches:(i+1)*n_batches]
        fake_generated_data = G(torch.Tensor(noise))
        #print(f'Fake data: {fake_generated_data}')
        # print()

        fake_decision = D(fake_generated_data)
        #print(f'Fake decision: {fake_decision}')
        # print()

        error = criterion(fake_decision, torch.ones(g_minibatch_size, 1))
        error.backward()
        loss, generated = error.item(), fake_generated_data
        g_optimizer.step()

        epoch_generator_loss.append(loss)

        #print(f'|Generator| Epoch {epoch} | Batch {i} | Loss: {loss}')
        # print()

    generator_losses.append(np.mean(epoch_generator_loss))
    epoch_generator_loss = []

    discriminator_losses.append(np.mean(epoch_discriminator_loss))
    epoch_discriminator_loss = []

    print(f'|Discriminator| Epoch {epoch} | Loss: {discriminator_losses[-1]}')
    print(f'|Generator|     Epoch {epoch} | Loss: {generator_losses[-1]}')

    print(50*'=')


# import matplotlib.pyplot as plt

# def draw(data) :
#     plt.figure()
#     plt.plot(range(max_epochs), data)
#     plt.show()


# draw(discriminator_losses)
# draw(generator_losses)
