import network_torch as network
import torch
import numpy as np
import matplotlib.pyplot as plt

# set training parameters
f_etas = [0.05, 0.05, 0.05, 0.05]
b_etas = [0.1, 0.1, 0.1, 0.1]
# b_etas = None
n = [500, 200, 50, network.n_out]
n_epochs = 100

n_trials = 1

losses = np.zeros((n_trials, len(n), int(n_epochs*network.sequence_length/100.0)))

for i in range(n_trials):
    print("Trial {}/{}. ---------------------".format(i+1, n_trials))

    # W_list = []
    # b_list = []
    # Y_list = []
    # c_list = []

    # W_list.append(torch.from_numpy(W_range[0]*np.random.uniform(-1, 1, size=(n[0], network.n_in)).astype(np.float32)))
    # W_list.append(torch.from_numpy(W_range[1]*np.random.uniform(-1, 1, size=(n[1], n[0])).astype(np.float32)))
    # W_list.append(torch.from_numpy(W_range[2]*np.random.uniform(-1, 1, size=(n[2], n[1])).astype(np.float32)))

    # b_list.append(torch.from_numpy(b_range[0]*np.ones((n[0], 1)).astype(np.float32)))
    # b_list.append(torch.from_numpy(b_range[1]*np.ones((n[1], 1)).astype(np.float32)))
    # b_list.append(torch.from_numpy(b_range[2]*np.ones((n[2], 1)).astype(np.float32)))

    # Y_list.append(torch.from_numpy(W_range[0]*np.random.uniform(-1, 1, size=(n[0], n[1])).astype(np.float32)))
    # Y_list.append(torch.from_numpy(W_range[1]*np.random.uniform(-1, 1, size=(n[1], n[2])).astype(np.float32)))

    # c_list.append(torch.from_numpy(b_range[0]*np.ones((n[0], 1)).astype(np.float32)))
    # c_list.append(torch.from_numpy(b_range[1]*np.ones((n[1], 1)).astype(np.float32)))

    # create the network -- this will also load the MNIST dataset files
    net = network.Network(n=n)

    # net.set_weights(W_list, b_list, Y_list, c_list)

    # train the network
    loss = net.train(f_etas, b_etas, n_epochs, save_simulation=False, plot=True)

    losses[i] = loss

plt.ioff()
plt.close('all')

fig, ax = plt.subplots()

colors = ['r', 'g', 'b', 'y']
for m in range(len(n)):
    losses_max = losses[:, m].max(axis=0)
    losses_min = losses[:, m].min(axis=0)

    # ax.fill_between(np.arange(int(n_epochs*network.sequence_length/100.0)), losses_min, losses_max, facecolor=colors[m], alpha=0.5)

    ax.plot(np.arange(int(n_epochs*network.sequence_length/100.0)), np.mean(losses[:, m], axis=0), colors[m], label='Layer {} Loss'.format(m+1), alpha=0.5)
plt.legend()
plt.show()